from engine.gpt_training import LmPretraining
import argparse
import torch.nn as nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path")
parser.add_argument("--save_path")
parser.add_argument("--add_head", action='store_true')



args = parser.parse_args()

ckpt_path = args.ckpt_path
save_path = args.save_path
add_head = args.add_head



module = LmPretraining.load_from_checkpoint(ckpt_path, device_map="auto")
lm_model = module.lm_model.lm_model

# print(lm_model.get_input_embeddings().weight.data)
# print(lm_model.lm_head.weight.data)

if add_head:
    print("Adding head...")
    embs = lm_model.get_input_embeddings()
    embed_in = embs.weight.data.clone()

    lm_head = nn.Linear(*embed_in.T.shape, bias=False, device=embed_in.device, dtype=embed_in.dtype)
    lm_head.weight = torch.nn.Parameter(embed_in.contiguous())
    lm_model.lm_head = lm_head


module.tokenizer.save_pretrained(save_path) 
lm_model.save_pretrained(save_path) 