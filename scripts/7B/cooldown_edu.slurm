#!/bin/bash
#SBATCH --account=cad14911
#SBATCH --job-name="lm_train"
#SBATCH --constraint=MI250
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.out
#SBATCH --nodes=8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --signal=SIGUSR1@60

module purge

module load cpe/23.12
module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-gnu
module load cray-python/3.11.5
module load amd-mixed/6.0.0
module load aws-ofi-rccl

source fast-venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/lus/work/CT10/cad14911/ngodey/data/hf_cache
export HF_HOME=/lus/work/CT10/cad14911/ngodey/data/hf_cache


export MPICH_GPU_SUPPORT_ENABLED=1


srun --ntasks-per-node=8 --cpus-per-task 8 --cpu-bind cores \
python train.py --config=configs/cooldown_8b.json --num_nodes 8 \
--grad_clip_val 1 --global_bs 1024 --train_bs 2 --val_bs 2 --precision "bf16-true" --attn_type flash_attention_2 \
--hf_path ${SCRATCHDIR}/ckpts/hama8b-bf16-edu20B-bs1024-n32_1458655	/hf_4k \
--hf_model_cls "LlamaForCausalLM" \
--run_name "hama8b-bf16-edu20B-bs1024-n32-4k-cooldown1k" \
--dataset /lus/work/CT10/cad14911/ngodey/data/processed/fineweb_edu_350BT_llama3_sl2049_opt \
--strategy fsdp_grad_op --hf_tokenizer 'meta-llama/Meta-Llama-3-8B' --model_max_seq_len 2048 \
--saved_ckpt_path "${SCRATCHDIR}/ckpts" --ckpt_every 500 \
--max_val_batches 32