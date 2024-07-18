#!/bin/bash
#SBATCH --account=cad14911
#SBATCH --job-name="preproc"
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.out
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1:00:00


module load cray-python/3.10.10
source /lus/work/CT10/cad14911/ngodey/new-headless-lm/venv/bin/activate

echo -e "Host name : $SLURM_JOB_NODELIST \n"


export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE=/lus/work/CT10/cad14911/ngodey/data/hf_cache
export HF_HOME=/lus/work/CT10/cad14911/ngodey/data/hf_cache

python preprocess.py --num_proc 40 --batch_size 10000 --max_seq_len 2049 \
--tokenizer meta-llama/Meta-Llama-3-8B --dataset_name DATASET_PATH \
--output_path OUTPUT_DATASET_PATH
