#!/bin/bash
#SBATCH --job-name=grpo                 # Sets the name of the job.
#SBATCH --output=logs/%x_%j.log         # Specifies the file to which standard output and error are written. %x is the job name, %j is the job ID.
#SBATCH --open-mode=append              # Appends the output to the log file instead of overwriting it.
#SBATCH --partition=batch               # Requests a specific partition (queue) for the job.
#SBATCH --qos=high_snlp                 # Requests a specific Quality of Service for the job.
#SBATCH --nodes=1                       # Requests a specific number of nodes.
#SBATCH --ntasks-per-node=1             # Specifies the number of tasks to be launched on each node.

set -x -e

(( SLURM_GPUS_ON_NODE > 0 )) && export OMP_NUM_THREADS=$((SLURM_CPUS_ON_NODE / SLURM_GPUS_ON_NODE)) || export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98

source ~/miniconda3/etc/profile.d/conda.sh

conda activate r1qa

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=$(shuf -i 49152-65535 -n 1)
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"
export RAY_ADDRESS="$ip_head"

# Define and create a job-specific temporary directory for Ray
RAY_TEMP_DIR="/tmp/ray_${USER}/${SLURM_JOB_ID}"
mkdir -p "$RAY_TEMP_DIR"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --temp-dir="$RAY_TEMP_DIR" \
    --num-cpus "${SLURM_CPUS_ON_NODE}" --num-gpus "${SLURM_GPUS_ON_NODE}" --block &

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --temp-dir="$RAY_TEMP_DIR" \
        --num-cpus "${SLURM_CPUS_ON_NODE}" --num-gpus "${SLURM_GPUS_ON_NODE}" --block &
    sleep 5
done

EXPERIMENT_NAME="Qwen3-1.7B-GRPO"
MODEL_PATH="Qwen/Qwen3-1.7B"
TRAIN_FILES="dgslibisey/MuSiQue"
VAL_FILES="dgslibisey/MuSiQue"
ROLLOUT_N=8
TOTAL_EPOCHS=2

srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    env -u ROCR_VISIBLE_DEVICES \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='r1qa' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$SLURM_GPUS_ON_NODE \
    trainer.nnodes=$SLURM_NNODES \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.max_actor_ckpt_to_keep=6 \
    trainer.total_epochs=$TOTAL_EPOCHS
    # actor_rollout_ref.model.use_shm=True \

ray stop
