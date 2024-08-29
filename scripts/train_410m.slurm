#!/bin/bash
#SBATCH --account=cad14911
#SBATCH --job-name="lm_train"
#SBATCH --constraint=MI250
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --signal=SIGUSR1@120

module purge

module load cpe/23.05
module load craype-accel-amd-gfx90a craype-x86-trento
module load cray-python/3.11.5

source rocm6-venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/lus/work/CT10/cad14911/ngodey/data/hf_cache
export HF_HOME=/lus/work/CT10/cad14911/ngodey/data/hf_cache


export OMP_NUM_THREADS=1
export OMP_PROC_PLACES=cores
export OMP_PROC_BIND=close
export MPICH_GPU_SUPPORT_ENABLED=1

export GPU_ARCHS="gfx90a"

srun --ntasks-per-node=8 --cpus-per-task 1 --cpu-bind cores \
python train.py --config=configs/pythia_410m.json --num_nodes 1 \
--grad_clip_val 0 --global_bs 64 --train_bs 8 --val_bs 8 --precision "bf16-mixed" --attn_type sdpa \
--hf_path 'EleutherAI/pythia-410m' --hf_model_cls "GPTNeoXForCausalLM" \
--run_name "pythia-410m-bf16-sdpa-comp" \
--dataset /lus/work/CT10/cad14911/ngodey/data/processed/redpajama_1B_2048_pythia \
--strategy ddp --hf_tokenizer 'EleutherAI/pythia-70m' --model_max_seq_len 2048 --saved_ckpt_path "${SCRATCHDIR}/ckpts" \
--ckpt_every 100