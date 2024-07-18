from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import psutil
import functools
import operator
import argparse
import torch


parser = argparse.ArgumentParser()
parser.add_argument("--num_proc")
parser.add_argument("--batch_size")
parser.add_argument("--max_seq_len")
parser.add_argument("--modulo", default=1)
parser.add_argument("--tokenizer")
parser.add_argument("--dataset_name")
parser.add_argument("--dataset_config", default="default")
parser.add_argument("--output_path")


args = parser.parse_args()

num_proc = int(args.num_proc)
max_seq_len = int(args.max_seq_len)
batch_size = int(args.batch_size)
modulo = int(args.modulo)
tokenizer = args.tokenizer
dataset_name = args.dataset_name
dataset_config = args.dataset_config
output_path = args.output_path

NUM_CPU = psutil.cpu_count()
num_proc = min(NUM_CPU//2, num_proc)

print(f"Using {num_proc} CPUs...")

tokenizer = AutoTokenizer.from_pretrained(tokenizer, token="MY_HF_TOKEN")

def tokenize_and_pack(batch, max_seq_len=max_seq_len):
    tokenized_batch = tokenizer(batch["text"]).input_ids
    tokenized_batch_flat = functools.reduce(operator.iconcat, tokenized_batch, [])
    packed_batch = torch.tensor(tokenized_batch_flat[:-(len(tokenized_batch_flat)%max_seq_len)]).reshape(-1, max_seq_len)
    return list(packed_batch)

print("Loading dataset...")
ds = load_dataset(dataset_name, dataset_config, num_proc=num_proc)

print("Packing dataset...")
ds = ds.map(lambda x: {"packed":tokenize_and_pack(x)}, remove_columns=ds['train'].column_names, batched=True, batch_size=batch_size, num_proc=num_proc)

print("Saving dataset...")
ds.save_to_disk(output_path, num_proc=num_proc)
