import os
from engine.datamodule import TextDataModule
from engine.gpt_training import LmPretraining

import psutil
import argparse
import torch
import json
import shortuuid

import lightning as L
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies import FSDPStrategy

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer




from lightning.pytorch.profilers import PyTorchProfiler

from lightning.pytorch.callbacks import LearningRateMonitor

L.seed_everything(42)

cpu_max = psutil.cpu_count()//2
print("CPU count: ", cpu_max)

print(f"Torch version: {torch.__version__}")

cuda_device_name = torch.cuda.get_device_name()
is_a100 = "A100" in cuda_device_name or "MI250" in cuda_device_name

if is_a100:
  torch.set_float32_matmul_precision('high')


parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--num_nodes", type=int)
parser.add_argument("--global_bs", type=int)
parser.add_argument("--train_bs", type=int)
parser.add_argument("--val_bs", type=int)

parser.add_argument("--dataset")
parser.add_argument("--hf_tokenizer")
parser.add_argument("--grad_clip_val", type=float)

parser.add_argument("--run_name")
parser.add_argument("--hf_path")
parser.add_argument("--hf_model_cls", default="LlamaForCausalLM")

parser.add_argument("--precision", default="16-mixed")
parser.add_argument("--attn_type", default="eager")
parser.add_argument("--strategy", default="ddp")


parser.add_argument('--ckpt_path', nargs='?', const=None, type=str)

parser.add_argument('--model_max_seq_len', type=int, default=2048)

parser.add_argument('--saved_ckpt_path')
parser.add_argument("--ckpt_every", type=int, default=10000)
parser.add_argument("--val_check_every", type=int, default=250)
parser.add_argument("--max_val_batches", type=int, default=10)
parser.add_argument("--log_every_n_steps", type=int, default=25)

parser.add_argument("--num_val_samples", type=int, default=1000)

parser.add_argument('--use_profiler', action='store_true')


args = parser.parse_args()

locals().update(vars(parser.parse_args()))

gpus_by_node = torch.cuda.device_count()

if ((gpus_by_node * num_nodes) % global_bs) == 0:
  raise argparse.ArgumentError(f"Requested a batch size of {global_bs} on {train_bs}x{gpus_by_node} GPUs : not a multiple!")
accu_grad_batches = global_bs // (gpus_by_node * num_nodes * train_bs)
print(f"Train BS: {train_bs}; Grad. accumulating factor: {accu_grad_batches}; Val BS: {val_bs}")


datamodule = TextDataModule(
  dataset, max_seq_len=model_max_seq_len, num_val_samples=num_val_samples, train_batch_size=train_bs, val_batch_size=val_bs, num_proc=1
)

config = json.load(open(config, "rb"))

print(config)

lightning_module = LmPretraining(
    hf_tokenizer, hf_path, hf_model_cls, attn_type=attn_type, **config
)


version_name = f"{run_name}_{os.environ.get('SLURM_JOB_ID', shortuuid.uuid()[:8])}"
logger = pl_loggers.TensorBoardLogger(save_dir="lightning_logs", version=version_name)

callbacks = [
  ModelCheckpoint(every_n_train_steps=ckpt_every, dirpath=f'{saved_ckpt_path}/{version_name}', save_top_k=-1),
  ModelCheckpoint(every_n_train_steps=1000, dirpath=f'{saved_ckpt_path}/{version_name}_last', save_top_k=1)
]

profiler=None
if use_profiler:
  profiler = PyTorchProfiler(dirpath="profiler_out", filename="profiling", row_limit=-1, export_to_chrome=True, sort_by_key="cuda_time_total")

lr_monitor = LearningRateMonitor(logging_interval='step')
callbacks.append(lr_monitor)

if strategy.startswith("fsdp"):
  sharding_strategy = "SHARD_GRAD_OP" if strategy=="fsdp_grad_op" else "FULL_SHARD"
  layers = {LlamaDecoderLayer, GPTNeoXLayer}
  strategy = FSDPStrategy(
    auto_wrap_policy=layers,
    sharding_strategy=sharding_strategy
  )

trainer = L.Trainer(
  num_nodes=num_nodes,
  precision=precision,
  accumulate_grad_batches=accu_grad_batches,
  logger=logger,
  callbacks=callbacks,
  strategy=strategy,
  # plugins=[SLURMEnvironment()],
  max_steps=config["total_nb_steps"],
  limit_val_batches=max_val_batches,
  log_every_n_steps = log_every_n_steps,
  val_check_interval=val_check_every * accu_grad_batches,
  gradient_clip_val=grad_clip_val,
  benchmark=True,
  default_root_dir=f'{saved_ckpt_path}/{version_name}',
  profiler=profiler,
)

trainer.fit(
  lightning_module,
  datamodule,
  ckpt_path=ckpt_path,
)
