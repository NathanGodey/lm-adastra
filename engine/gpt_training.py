import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

import transformers as T
import models as M
import optimizers as O

from transformers import get_constant_schedule_with_warmup
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from functools import partial


TEST_SENTENCES = [
            "My name is Nathan, I am 25 and",
            "Question: What is 3+5? Answer: 8. Question: What is 4+8? Answer:",
            "A notorious protester convicted of wilfully promoting hatred against Muslims and criminally harassing a"
        ]
        

class LmPretraining(L.LightningModule):
    def __init__(
        self, hf_tokenizer, hf_path, hf_model_cls, attn_type, text_key = "text", tokenized = True, max_seq_len = 2048, learning_rate = 6e-4,
        weight_decay = 1e-1, min_lr_rate = 0.1, warmup_steps = 1000, total_nb_steps = 1e6, optimizer_cls = "AdamW",
        optim_kwargs = None, schedule_type = "cosine", hf_load_weights=False, model_kwargs=None, torch_compile=False):
        super().__init__()
        self.tokenizer = T.AutoTokenizer.from_pretrained(hf_tokenizer)
        self.lm_model = None

        self.model_cls = getattr(M, hf_model_cls)
        self.hf_path = hf_path
        self.attn_type = attn_type
        self.hf_load_weights = hf_load_weights
        self.torch_compile = torch_compile

        self.text_key = text_key
        self.tokenized = tokenized

        self.max_seq_len = max_seq_len

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.min_lr_rate = min_lr_rate
        self.warmup_steps = warmup_steps
        self.total_nb_steps = total_nb_steps

        self.model_kwargs = model_kwargs if model_kwargs is not None else dict()

        default_optim_kwargs = {
            "eps": 1e-8,
            "betas": [0.9, 0.95]
        }
        self.optim_kwargs = default_optim_kwargs if optim_kwargs is None else optim_kwargs

        self.schedule_type = schedule_type

        self.save_hyperparameters(ignore=['tokenizer', 'lm_model'])
    
    def configure_model(self):
        if self.lm_model is not None:
            return
        if self.hf_load_weights:
            lm_model = self.model_cls.from_pretrained(self.hf_path, attn_implementation=self.attn_type)
        else:
            lm_config = T.AutoConfig.from_pretrained(self.hf_path, attn_implementation=self.attn_type)
            lm_config.vocab_size = len(self.tokenizer.vocab)
            lm_config.max_position_embeddings = self.max_seq_len
            lm_model = self.model_cls(lm_config)
        
        if self.torch_compile:
            self.lm_model = torch.compile(lm_model)
        else:
            self.lm_model = lm_model


    def common_step(self, inputs):
        inputs_ids = inputs[self.text_key]
        labels = inputs_ids[..., 1:]

        model_out = self.lm_model(
            inputs_ids[..., :-1],
        )

        logits = model_out.logits
        lm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels
        )

        with torch.no_grad():
            lm_predictions = torch.argmax(logits.detach(), dim=-1)
            lm_acc = (labels == lm_predictions).float().mean()
        
        return lm_loss, lm_acc


    def training_step(self, inputs):
        lm_loss, lm_acc = self.common_step(inputs)
        
        result_dict = {
            "train/lm_loss": lm_loss,
            "train/lm_accuracy": lm_acc
        }

        self.log_dict(result_dict)

        return result_dict["train/total_loss"]

    def validation_step(self, inputs):
        lm_loss, lm_acc = self.common_step(inputs)
        
        result_dict = {
            "train/lm_loss": lm_loss,
            "train/lm_accuracy": lm_acc
        }

        self.log_dict(result_dict, sync_dist=True)
    
    def on_validation_epoch_end(self):
        generations = []
        with torch.autocast(device_type=self.device.type):
            for sent in TEST_SENTENCES:
                tokenized_test_sent = self.tokenizer(sent, return_tensors="pt")
                tokenized_test_sent = {k: v.to(self.device) for k, v in tokenized_test_sent.items()}
                generate_out = self.lm_model.generate(**tokenized_test_sent, max_new_tokens=50)
                tokens_generated = self.tokenizer.decode(generate_out[0])
                generations.append(tokens_generated)
        
        generations_str = "\n\n".join(generations)
        
        self.logger.experiment.add_text("test_samples", generations_str, global_step=self.global_step)


    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(),
                                     lr=self.learning_rate,
                                     **self.optim_kwargs)

        schedule_type = getattr(self, "schedule_type", "cosine")
        if schedule_type == "cosine":
            lr_scheduler_func = get_cosine_with_min_lr_schedule_with_warmup(
                optimizer,
                self.warmup_steps,
                self.total_nb_steps,
                min_lr_rate = self.min_lr_rate
            )
        elif schedule_type == "constant":
            lr_scheduler_func = get_constant_schedule_with_warmup(
                optimizer,
                self.warmup_steps,
            )
        else:
            raise NotImplementedError(f"LR schedule named '{schedule_type}' is not implemented.")
        lr_scheduler = {
            'scheduler': lr_scheduler_func,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]
