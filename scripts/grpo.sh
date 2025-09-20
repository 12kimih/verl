set -x -e

export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=1

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh

conda activate r1qa

N_GPU=4
TOTAL_EPOCHES=3
# GLOBAL_BATCH_SIZE=128
# ROLLOUT_BATCH_SIZE=384
# VAL_BATCH_SIZE=512

EXP_NAME="Qwen3-0.6B-SFT-GRPO"
MODEL_PATH="12kimih/Qwen3-0.6B-R1QA-SFT-M"

TRAIN_FILE="dgslibisey/MuSiQue"
VAL_FILE="dgslibisey/MuSiQue"
ROLLOUT_N=8

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPU \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='r1qa' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$N_GPU \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.max_actor_ckpt_to_keep=6 \
    trainer.total_epochs=$TOTAL_EPOCHES
    # actor_rollout_ref.model.use_shm=True \
