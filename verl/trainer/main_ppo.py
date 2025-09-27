# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
import os

import ray
import hydra
import random
import asyncio

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import numpy as np

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, qwen_math, code, kklogic
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

import logging

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

CURR_FILE_PATH = os.path.abspath(__file__)
TRAINER_DIR = os.path.dirname(CURR_FILE_PATH)
VERL_DIR = os.path.dirname(TRAINER_DIR)
WORKING_DIR = os.path.dirname(VERL_DIR)
TMP_DIR = os.path.join(WORKING_DIR, "tmp/")


def seed_everything(seed: int = 42):
    """

    Seed everything to ensure reproducibility.

    Args:
        seed (int): The seed number to use.
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(seed)  # PyTorch GPU seed
    torch.cuda.manual_seed_all(seed)  # PyTorch multi-GPU seed


# Example usage
seed_everything(42)

SOURCE_TO_SCORE_FN = {
    "openai/gsm8k": gsm8k.compute_score,
    "math": qwen_math.compute_score,
    "AIME22": qwen_math.compute_score,
    "AIME23": qwen_math.compute_score,
    "AIME24": qwen_math.compute_score,
    "numina_amc_aime": qwen_math.compute_score,
    "math-instruct": qwen_math.compute_score,
    "code": code.compute_score,
    "kk_logic": kklogic.compute_score,
}


class RewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, score_fn_strategy="default") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.score_fn_strategy = score_fn_strategy
        assert self.score_fn_strategy in ["default", "with_format"]

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    async def __call__(self, data: DataProto):
        """
        If there is an rm score in the batch, return that directly. Otherwise,
        compute scores asynchronously via a process pool.
        """
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        # Prepare an empty tensor for the rewards
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        extracted_answers = [None] * len(data)  # Extracted final answers from responses
        answer_correctness = torch.zeros_like(
            data.batch["responses"][:, 0]
        )  # Whether extracted answer is correct

        already_print_data_sources = {}

        results_cache = {}  # A dictionary that stores verified results: (extracted_answer, GT) --> score
        lock = Lock()

        # We collect futures here, along with an index and the length info needed to place the scores
        futures = []

        total_items = len(data)
        step = 8

        with ThreadPoolExecutor(max_workers=12) as executor:
            for start in range(step):
                for i in range(start, total_items, step):
                    data_item = data[i]  # DataProtoItem

                    prompt_ids = data_item.batch["prompts"]
                    prompt_length = prompt_ids.shape[-1]
                    valid_prompt_length = data_item.batch["attention_mask"][
                        :prompt_length
                    ].sum()
                    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                    response_ids = data_item.batch["responses"]
                    valid_response_length = data_item.batch["attention_mask"][
                        prompt_length:
                    ].sum()
                    valid_response_ids = response_ids[:valid_response_length]

                    # Decode
                    sequences = torch.cat((valid_prompt_ids, valid_response_ids))
                    # sequences_str = self.tokenizer.decode(sequences)
                    sequences_str = self.tokenizer.decode(valid_response_ids)

                    ground_truth = data_item.non_tensor_batch["reward_model"][
                        "ground_truth"
                    ]
                    data_source = data_item.non_tensor_batch["data_source"]

                    if self.score_fn_strategy == "default":
                        compute_score_fn = SOURCE_TO_SCORE_FN[data_source]
                    else:
                        raise ValueError()

                    # Handle optional logging
                    if data_source not in already_print_data_sources:
                        already_print_data_sources[data_source] = 0
                    if already_print_data_sources[data_source] < self.num_examine:
                        already_print_data_sources[data_source] += 1
                        self.logger.info(
                            f"[{data_source}] Sample sequence: {sequences_str}"
                        )

                    # Submit the computation to the process pool
                    future = executor.submit(
                        compute_score_fn,
                        sequences_str,
                        ground_truth,
                        results_cache,
                        lock,
                    )
                    futures.append((future, i, valid_response_length))

            # Now gather results asynchronously
            for future, idx, resp_len in futures:
                score, answer = future.result()
                if data.non_tensor_batch["extra_info"][idx]["split"] == "test":
                    score = 1 if score == 1 else 0
                reward_tensor[idx, resp_len - 1] = score
                extracted_answers[idx] = answer
                answer_correctness[idx] = score == 1

        reward_tensor = {
            "token_level_scores": reward_tensor,
            "answer_correctness": answer_correctness,
            "extracted_answers": np.array(extracted_answers, dtype=object),
        }
        if "level" in data.non_tensor_batch:
            reward_tensor["level"] = data.non_tensor_batch["level"]
        if "uid" in data.non_tensor_batch:
            reward_tensor["uid"] = data.non_tensor_batch["uid"]

        return DataProto.from_single_dict(reward_tensor)



@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}
            },
            log_to_driver=True,
        )

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(
        OmegaConf.to_container(config, resolve=True)
    )  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # import pdb; pdb.set_trace()

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            if config.reward_model.use_potential_reward:
                from verl.workers.fsdp_workers import (
                    PotentialRewardModelWoker as RewardModelWorker,
                )
            else:
                from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    score_fn_strategy = config.reward_model.score_fn_strategy
    reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        score_fn_strategy=score_fn_strategy,
    )

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(
        tokenizer=tokenizer, num_examine=3,
        score_fn_strategy=score_fn_strategy,
    )

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    asyncio.run(trainer.fit_async())


if __name__ == "__main__":
    main()
