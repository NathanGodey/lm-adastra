#!/bin/bash
#SBATCH --account=cad14911
#SBATCH --job-name="pp_headless"
#SBATCH --constraint=MI250
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --time=01:00:00
#SBATCH --signal=SIGUSR1@120

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


export OMP_NUM_THREADS=1
export OMP_PROC_PLACES=cores
export OMP_PROC_BIND=close
export MPICH_GPU_SUPPORT_ENABLED=1

python prepare_headless.py \
--ckpt_path "${SCRATCHDIR}/ckpts/llama8b-bf16-edu20B-bs1024-n32_1458619/epoch=0-step=4000.ckpt" \
--save_path ${SCRATCHDIR}/ckpts/llama8b-bf16-edu20B-bs1024-n32_1458619/hf_4k