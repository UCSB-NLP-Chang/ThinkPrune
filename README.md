# ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning

This is the official implementation of the paper:
 ***ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning\***

> ThinkPrune efficiently trims long reasoning chains in LLMs using reinforcement learning.




## **Table of Contents**  

1. [Enviroment & Data Preprocessing](#data-preprocess)  
2. [Training](#training)
   - [Merge Checkpoints](#ckpt-merge)
   - [32B Model Training](#large-model)
3. [Evaluation](#evaluation)  
4. [Analysis](#analyze-reasoning)
5. [Acknowledgement](#acknowledgement)

---


- 🤗 [HF Models](https://huggingface.co/collections/Shiyu-Lab/thinkprune-67f550c57f5baf0521403881)


## Enviroment & Data Preprocessing

<a id="data-preprocess"></a>

We need to first create a virtual environment:

```
conda create -n thinkprune python==3.10
pip install -e ./
```





We use the AIME/AMC subset from [Prime Collection](https://github.com/PRIME-RL/Eurus-2-RL-Data), originally sourced from NuminaMath. Our preprocessing only add an extra system prompt to each question. To generate the data that follows the required format of Verl:

```sh
python examples/data_preprocess/preprocess_past_aime_amc.py \
    --dataset_cache_path=aux_data \
    --model_family=deepseek \
    --max_length=4000 \
    --save_dir=data/past_aime_amc/length4000
```



Since we slightly change the system prompt when training QwQ-32B, please run the following scripts to generate the training data for QwQ:

```
python examples/data_preprocess/preprocess_past_aime_amc.py \
    --dataset_cache_path=aux_data \
    --model_family=qwen \
    --max_length=4000 \
    --save_dir=data/past_aime_amc_qwq/length4000
```





## Training

<a id="training"></a>

Training scripts are in `local_scripts/` and assume usage of a **Slurm** cluster and Ray. Several key steps are:

1. Set the correct working directory and environment

```
cd /path/to/thinkprune
# Activate virtual environment
source ~/.bashrc
conda activate verl_thinkprune
```

2. Set your Wandb api key:

```
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=YOUR_WANDB_KEY
```

3. `Delete /tmp/ray/`

```
srun rm -rf /tmp/ray/
```

This looks a little wired😅. We empirically find this is necessary sometimes (please tell us if you have better solutions, thanks!). The main reasons are:

- when run multi-node training, the ray cache has to be set as the default directory, otherwise we always observe the worker nodes cannot join the ray cluster

- However, if you are on a Slurm cluster and the node allocated to you was used by other people with Ray for LLM training previously, there will often be a directory `/tmp/ray/ray_current_cluster` that is not owned by us and not deleted by the previous user after the experiment finished.
- But Ray need the write access to `/tmp/ray/ray_current_cluster`, which we do not have.

So, we directly delete this folder (of course we cannot delete other files -- this is only effective to delete our ray cache and `/tmp/ray/ray_current_cluster`)

See [this issue](https://github.com/UCSB-NLP-Chang/ThinkPrune/issues/2) for a more elegant solution.

4. choose the model and length limit

```
LENGTH=4000
RUN_NAME=DeepSeek-R1-Distill-Qwen-1.5B-${LENGTH}
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

5. Set the model save path, data path, and gpu usage

```
N_GPUS=8 # GPU per node
TP=1     # tensor parallel for vLLM rollout
MODEL_DIR=/path/to/checkpoints/${RUN_NAME}
DATA_DIR=/path/to/thinkprune/data/past_aime_amc/length${LENGTH}
```

Note: If you are training QwQ, see 

6. First start ray on the master node then on the working nodes

```
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port  --block &

sleep 10

# Start worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" --block &
    sleep 5
done
```



7. Launch training on the master node

```
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ...
```



To run iterative pruning, you only need to change the length limit and switch the model path from Huggingface model name to a local checkpoint:

```
LENGTH=3000
RUN_NAME=DeepSeek-R1-Distill-Qwen-1.5B-${LENGTH}-from4k
MODEL=checkpoints-merged/DeepSeek-R1-Distill-Qwen-1.5B-4000/global_step_180
```

`Note`: the Verl framework will save checkpoints in a sharded manner. We need to manually merge the sharded checkpoints into one before we can load them using `.from_pretrained()` function in Huggingface Transformers. See [Merge Checkpoints](#ckpt-merge)

`Note`: for 32B model training, see [32B Model Training](#large-model)



### Merge Checkpoints

<a id="ckpt-merge"></a>
To merge the sharded checkpoints saved in `checkpoints/{model_name}/actor/global_step_{step}`, use the following scripts from [Verl](https://github.com/volcengine/verl/blob/main/scripts/model_merger.py)

```
python examples/model_merger.py \
    --local_dir checkpoints/DeepSeek-R1-Distill-Qwen-1.5B-4000/actor/global_step_100 \
    --save_dir checkpoints-merged/DeepSeek-R1-Distill-Qwen-1.5B-4000/global_step_100
```


### 32B Model Training

<a id="large-model"></a>

This is most the same as the 1.5B models. But the following changes might be useful:

1. Change the data path to `past_aime_amc_qwq`, which use the system prompt for QwQ-32B. Also, we set TP=8

```
N_GPUS=8 # GPU per node
TP=8     # tensor parallel for vLLM rollout
MODEL_DIR=/path/to/checkpoints/${RUN_NAME}
DATA_DIR=/path/to/thinkprune/data/past_aime_amc_qwq/length${LENGTH}
```

2. Enable offload parameters:

```
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=${LENGTH} \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$ROLLOUT_BS \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    +actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$ROLLOUT_BS \
    reward_model.enable=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$MODEL_DIR \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_math' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=4 \
    trainer.multisample_val=True \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    trainer.num_keep_checkpoint=10 \
    trainer.resume_checkpoint=True
```



## Evaluation

<a id="evaluation"></a>

We use a naive resource allocation strategy to run parallel LLM inference:

```
NUM_GPUS=8  # Total number of GPUs available
BATCH_SIZE=4  # Adjust batch size if needed
MODEL_NAME="qwen1.5B"

for RANK in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$RANK python tools/run_generation.py \
        --rank=$RANK \
        --world_size=$NUM_GPUS \
        --batch_size=$BATCH_SIZE \
        --dataset_name=math500 \
        --save_name=$MODEL_NAME \
        --model_name=$MODEL_NAME &
done

wait
```

It split the evaluation dataset into $NUM_GPUS subsets and evaluate each on one GPU. You may replace `MODEL_NAME` with the path to the huggingface checkpoint saved in your local disk



Similarly, for budget forcing:

```
NUM_GPUS=8  # Total number of GPUs available
BATCH_SIZE=4  # Adjust batch size if needed
GPU_OFFSET=0  # Starting GPU index

MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
DATASET=amc23

BUDGETS=("500" "1000" "2000" "4000" "8000")
for BUDGET in ${BUDGETS[@]}
do
    SAVE_NAME="qwen1_5B-budget${BUDGET}"
    for RANK in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$((GPU_OFFSET + RANK)) python tools/generation_tools/budget_forcing_gen.py \
            --rank=$RANK \
            --world_size=$NUM_GPUS \
            --batch_size=$BATCH_SIZE \
            --dataset_name=$DATASET \
            --max_tokens_thinking=$BUDGET \
            --orig_model_name=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
            --save_name=$SAVE_NAME \
            --model_name=$MODEL_NAME &
    done
    wait
done
```

Here we need one extra argument, `orig_model_name`. It should be the original huggingface name of the given model (e.g., if we train `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` and get a new model, `checkpoints-merged/length4000/global_step180`), then we need to set `model_name` as  `checkpoints-merged/length4000/global_step180` and `orig_model_name` as `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B ` to evaluate it.

Options for `orig_model_name`: `simplescaling/s1`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `agentica-org/DeepScaleR-1.5B-Preview`, `Qwen/QwQ-32B`.



To compute the metrics, run:

```
python tools/eval_accuracy.py --data_path_prefix=logs/math500_qwen1.5B
```

where `data_path_prefix` is the directory that stores the generation results (for each GPU).

## Use Another LLM to Analyze the Reasoning Chains

<a id="analyze-reasoning"></a>


To do this, you need to first setup your OpenAI/AzureOpenAI API key in the `query_api` function in `tools/analyze_tools/llm_summary.py`. Then you can run

```
python tools/analyze_tools/llm_summary.py --data_path_prefix=logs/math500/orig_qwen1_5B
```

Similarly, `data_path_prefix` is the same as the path when we evaluate the accuracy.

It will save the logs to `logs/llm_sum/{savename}.json`. Then we can visualize the proportion of each reasoning behavior via:

```
python tools/analyze_tools/analyze_summary.py --summary_path=logs/llm_sum/{savename}.json
```

## Acknowledgement

We use [Verl](https://github.com/volcengine/verl/tree/main) as our training framework. Since we use a older version, we did not fork the upstream github but instead made a copy without git dependency. 

```
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

## Citation

If you find our work useful, please kindly cite

```
@article{hou2025thinkprune,
  title={ThinkPrune: Pruning Long Chain-of-Thought of LLMs via Reinforcement Learning},
  author={Hou, Bairu and Zhang, Yang and Ji, Jiabao and Liu, Yujian and Qian, Kaizhi and Andreas, Jacob and Chang, Shiyu},
  journal={arXiv preprint arXiv:2504.01296},
  year={2025}
}
```
