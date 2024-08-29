#!/bin/bash
#SBATCH --account=cad14911
#SBATCH --job-name="s2_hl"
#SBATCH --constraint=MI250
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.out
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --hint=nomultithread
#SBATCH --time=10:00:00
#SBATCH --signal=SIGUSR1@120

module purge

module load cpe/23.12
module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-gnu
module load cray-python/3.11.5
module load amd-mixed/6.0.0
module load torch/2.3.0
module load aws-ofi-rccl


source rocm6-venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/lus/work/CT10/cad14911/ngodey/data/hf_cache
export HF_HOME=/lus/work/CT10/cad14911/ngodey/data/hf_cache


export MPICH_GPU_SUPPORT_ENABLED=1


srun --ntasks-per-node=8 --cpus-per-task 8 --cpu-bind cores \
python train.py --config=configs/hama_1b_step2.json --num_nodes 4 \
--grad_clip_val 1 --global_bs 512 --train_bs 4 --val_bs 8 --precision "bf16-true" --attn_type flash_attention_2 \
--hf_path ${SCRATCHDIR}/ckpts/hama1b-bf16-fa-ddp-bs512-wu200-n4_1417364/hf_10k --hf_model_cls "LlamaForCausalLM" \
--run_name "hama1b-s2-10k-bf16-fa-ddp-bs512-wu200-n4" \
--dataset /lus/work/CT10/cad14911/ngodey/data/processed/fineweb_10BT_llama3_sl2049_opt \
--strategy ddp --hf_tokenizer 'meta-llama/Meta-Llama-3-8B' --model_max_seq_len 2048 \
--saved_ckpt_path "${SCRATCHDIR}/ckpts" --ckpt_every 500 --val_check_every 20