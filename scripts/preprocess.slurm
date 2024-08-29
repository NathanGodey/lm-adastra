#!/bin/bash
#SBATCH --account=cad14911
#SBATCH --job-name="preproc"
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.out
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=24:00:00

module purge

module load cpe/23.12
module load craype-accel-amd-gfx90a craype-x86-trento
module load PrgEnv-gnu
module load cray-python/3.10.10

source dataproc-venv/bin/activate

echo -e "Host name : $SLURM_JOB_NODELIST \n"


export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/lus/work/CT10/cad14911/ngodey/data/hf_cache
export HF_HOME=/lus/work/CT10/cad14911/ngodey/data/hf_cache
export TOKENIZERS_PARALLELISM=true

python preprocess.py --num_proc 2 --batch_size 10000 --max_seq_len 2049 \
--tokenizer 'meta-llama/Meta-Llama-3-8B' \
--dataset_path /lus/work/shared/dataset/HuggingFaceFW--fineweb-edu/sample/350BT \
--output_path /lus/work/CT10/cad14911/ngodey/data/processed/fineweb_edu_350BT_llama3_sl2049_opt