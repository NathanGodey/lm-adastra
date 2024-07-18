#!/bin/bash
#SBATCH --account=cad14911
#SBATCH --job-name="lm_train"
#SBATCH --constraint=MI250
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --signal=SIGUSR1@120

module purge
# A CrayPE environment version
module load cpe/23.12
# An architecture
module load craype-accel-amd-gfx90a craype-x86-trento
# A compiler to target the architecture
module load PrgEnv-cray
# Some architecture related libraries and tools
module load amd-mixed

source venv/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/lus/work/CT10/cad14911/ngodey/data/hf_cache
export HF_HOME=/lus/work/CT10/cad14911/ngodey/data/hf_cache


export OMP_NUM_THREADS=1
export OMP_PROC_PLACES=cores
export OMP_PROC_BIND=close
export MPICH_GPU_SUPPORT_ENABLED=1

srun --ntasks-per-node=8 --cpus-per-task=8 --threads-per-core=1 --gpu-bind=closest \
python manta_dmc.py --config=configs/pythia_70m.json --num_nodes 1 \
--grad_clip_val 1 --global_bs 64 --train_bs 8 --val_bs 8 --precision "32" --attn_type eager \
--hf_path '/lus/scratch/CT10/cad14911/ngodey/ckpts/pythia_70m-prep0' --hf_model_cls "GPTNeoXForCausalLM" --run_name "pythia-70m" \
--dataset path_to_processed --strategy ddp \
--hf_tokenizer 'EleutherAI/pythia-70m' --model_max_seq_len 2048 --saved_ckpt_path "${SCRATCHDIR}/ckpts" --ckpt_every 100