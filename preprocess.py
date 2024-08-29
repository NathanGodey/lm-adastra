from pathlib import Path
import pyarrow.parquet as pq
from litdata import optimize, TokensLoader
from transformers import AutoTokenizer
from functools import partial
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--num_proc")
parser.add_argument("--batch_size")
parser.add_argument("--max_seq_len")
parser.add_argument("--node_id", default=0)
parser.add_argument("--num_nodes", default=1)
parser.add_argument("--tokenizer")
parser.add_argument("--dataset_path")
parser.add_argument("--dataset_config", default="default")
parser.add_argument("--output_path")


args = parser.parse_args()

num_proc = int(args.num_proc)
max_seq_len = int(args.max_seq_len)
batch_size = int(args.batch_size)
node_id = int(args.node_id)
num_nodes = int(args.num_nodes)
tokenizer = args.tokenizer
dataset_path = args.dataset_path
dataset_config = args.dataset_config
output_path = args.output_path

if num_nodes > 1:
    output_path = output_path + f"/node_{node_id}"

tokenizer = AutoTokenizer.from_pretrained(tokenizer, token="MY_HF_TOKEN")

# 1. Define a function to convert the text within the parquet files into tokens
def tokenize_fn(filepath, tokenizer=None, batch_size=8192):
    parquet_file = pq.ParquetFile(filepath)
    # Process per batch to reduce RAM usage
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["text"]):
        for text in batch.to_pandas()["text"]:
            tokens = tokenizer.encode(text)
            yield torch.tensor(tokens+[tokenizer.eos_token_id])

# 2. Generate the inputs
inputs = [str(file) for i, file in enumerate(Path(dataset_path).rglob("*.parquet")) if i%num_nodes == node_id]

if __name__ == "__main__":
# 3. Store the optimized data wherever you want under "/teamspace/datasets" or "/teamspace/s3_connections"
    print(f"Node {node_id} processing {len(inputs)} files: \n"+"\n".join(inputs))
    outputs = optimize(
        fn=partial(tokenize_fn, tokenizer=tokenizer, batch_size=batch_size), # Note: Use HF tokenizer or any others
        inputs=inputs,
        num_workers=num_proc if num_proc > 0 else None,
        output_dir=output_path,
        chunk_size=(max_seq_len * 8012), # Number of tokens to store by chunks. This is roughly 64MB of tokens per chunk.
        item_loader=TokensLoader(),
    )