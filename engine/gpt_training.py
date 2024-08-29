import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

import transformers as T
import models as M
from engine.loss_fns import cwt_loss
import optimizers as O

from transformers import get_constant_schedule_with_warmup
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from engine.lr_schedulers import get_linear_with_min_lr_schedule_with_warmup
from functools import partial
from lightning.pytorch.strategies import FSDPStrategy



TEST_SENTENCES = [
            "My name is Nathan, I am 25 and",
            "Question: What is 3+5? Answer: 8. Question: What is 4+8? Answer:",
            "A notorious protester convicted of wilfully promoting hatred against Muslims and criminally harassing a"
        ]
    
class LmWithLoss(nn.Module):
    def __init__(self, lm_model):
        super().__init__()
        self.lm_model = lm_model
    
    def get_input_embeddings(self):
        return self.lm_model.get_input_embeddings()

    def forward(self, inputs, is_headless=True):
        labels = inputs[..., 1:]
        lm_result = self.lm_model(inputs[..., :-1], output_hidden_states=True)
        logits = lm_result.logits

        if is_headless:
            embs = self.lm_model.get_input_embeddings()
            
            target_input_embeddings = embs(labels).flatten(0, 1)
            lm_loss = cwt_loss(
                logits.flatten(0, 1).to(target_input_embeddings.dtype),
                positive=target_input_embeddings,
                negative=target_input_embeddings
            )

            # embs = self.lm_model.get_input_embeddings()
            # unique_labels, label_count = torch.unique(labels, sorted=False, return_counts=True)

            # target_input_embeddings = embs(labels)
            # neg_input_embeddings = embs(unique_labels)
            # lm_loss = cwt_loss(
            #     logits.flatten(0, 1).to(target_input_embeddings.dtype),
            #     positive=target_input_embeddings.flatten(0, 1),
            #     negative=neg_input_embeddings,
            #     weights=label_count
            # )
        else:
            lm_loss = F.cross_entropy(
                logits.transpose(1, 2),
                labels
            )
        
        return lm_loss, logits, labels
        

class LmPretraining(L.LightningModule):
    def __init__(
        self, hf_tokenizer, hf_path, hf_model_cls, attn_type, max_seq_len = 2048, learning_rate = 6e-4, generate_samples=True, is_headless=False,
        weight_decay = 1e-1, min_lr_rate = 0.1, warmup_steps = 1000, total_nb_steps = 1e6, optimizer_cls = "AdamW",
        optim_kwargs = None, schedule_type = "cosine", hf_load_weights=False, model_kwargs=None, torch_compile=False,
        device_map_mode = 0):
        super().__init__()
        self.tokenizer = T.AutoTokenizer.from_pretrained(hf_tokenizer)
        self.lm_model = None

        self.model_cls = getattr(M, hf_model_cls)
        self.hf_path = hf_path
        self.attn_type = attn_type
        self.hf_load_weights = hf_load_weights
        self.torch_compile = torch_compile
        self.device_map_mode = device_map_mode

        self.is_headless = is_headless

        self.max_seq_len = max_seq_len

        self.optimizer_cls = optimizer_cls
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.min_lr_rate = min_lr_rate
        self.warmup_steps = warmup_steps
        self.total_nb_steps = total_nb_steps

        self.generate_samples = generate_samples

        self.model_kwargs = model_kwargs if model_kwargs is not None else dict()

        default_optim_kwargs = {
            "eps": 1e-8,
            "betas": [0.9, 0.95]
        }
        self.optim_kwargs = default_optim_kwargs if optim_kwargs is None else optim_kwargs

        self.schedule_type = schedule_type

        self.save_hyperparameters(ignore=['tokenizer', 'lm_model'])
    
    def configure_model(self):
        print("Loading model on ", self.device, flush=True)
        if self.lm_model is not None:
            return
        if self.hf_load_weights:
            lm_model = self.model_cls.from_pretrained(
                self.hf_path, attn_implementation=self.attn_type, use_cache=False,
                # device_map=self.device, low_cpu_mem_usage=True,
                # torch_dtype=self.dtype
            )
        else:
            lm_config = T.AutoConfig.from_pretrained(
                self.hf_path, attn_implementation=self.attn_type, use_cache=False,
                # device_map=self.device, low_cpu_mem_usage=True,
                # torch_dtype=self.dtype
            )
            lm_config.vocab_size = len(self.tokenizer.vocab)
            lm_config.max_position_embeddings = self.max_seq_len
            lm_model = self.model_cls(lm_config)

            if self.is_headless:
                lm_model.lm_head = torch.nn.Identity()
        
        print(f"Model dtype:", lm_model.get_input_embeddings().weight.dtype)

        lm_model = LmWithLoss(lm_model)
        if self.torch_compile:
            if self._trainer is not None and isinstance(self.trainer.strategy, FSDPStrategy):
                self.lm_model = torch.compile(lm_model)
            else:
                self.lm_model = torch.compile(lm_model, mode="reduce-overhead")
        else:
            self.lm_model = lm_model

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        if isinstance(self.trainer.strategy, FSDPStrategy):
            if self.trainer.strategy.sharding_strategy.name in ["SHARD_GRAD_OP", "NO_SHARD"]:
                torch.nn.utils.clip_grad_norm_(self.lm_model.parameters(), gradient_clip_val)
            else:
                raise ValueError("Gradient clipping cannot be used with FSDP when model is sharded.")
        else:
            self.clip_gradients(
                optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm
            )

    def common_step(self, inputs):

        lm_loss, logits, labels = self.lm_model(inputs, is_headless = self.is_headless)

        lm_acc = 0.
        if not self.is_headless:
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

        return result_dict["train/lm_loss"]

    def validation_step(self, inputs):
        lm_loss, lm_acc = self.common_step(inputs)
        
        result_dict = {
            "val/lm_loss": lm_loss,
            "val/lm_accuracy": lm_acc
        }

        self.log_dict(result_dict, sync_dist=True)
    
    def on_validation_epoch_end(self):
        if not self.generate_samples:
            return 
        generations = []
        with torch.autocast(device_type=self.device.type):
            for sent in TEST_SENTENCES:
                tokenized_test_sent = self.tokenizer(sent, return_tensors="pt")
                test_sent_tokens = tokenized_test_sent.input_ids.to(self.lm_model.device)
                for _ in range(50):
                    model_out = self.lm_model(test_sent_tokens)
                    new_token = model_out.logits[:, -1:, :].argmax(-1)
                    test_sent_tokens = torch.cat((test_sent_tokens, new_token), -1)

                tokens_generated = self.tokenizer.decode(test_sent_tokens[0])
                generations.append(tokens_generated)
        
        generations_str = "\n\n".join(generations)
        
        self.logger.experiment.add_text("test_samples", generations_str, global_step=self.global_step)


    def configure_optimizers(self):

        optimizer_fn = getattr(O, self.optimizer_cls)
        optimizer = optimizer_fn(
            self.parameters(),
            lr=self.learning_rate,
            **self.optim_kwargs
        )

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
        elif schedule_type == "linear":
            lr_scheduler_func = get_linear_with_min_lr_schedule_with_warmup(
                optimizer,
                self.warmup_steps,
                self.total_nb_steps,
                min_lr_rate = self.min_lr_rate
            )
        else:
            raise NotImplementedError(f"LR schedule named '{schedule_type}' is not implemented.")
        lr_scheduler = {
            'scheduler': lr_scheduler_func,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]
