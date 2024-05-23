import evaluate
import math
import numpy as np
import os
import re
import torch
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from transformers import BartForConditionalGeneration

from utils.scheduler import LinearWarmupLR

class PatentPretrain(pl.LightningModule) :
    def __init__(self, args, device, tokenizer) :
        super().__init__()
        self._device = device
        # self.ckpt_dir = args.ckpt_dir
        # self.num_warmup_steps = args.num_warmup_steps

        self.batch_size = args.batch_size

        self.min_learning_rate = args.min_learning_rate
        # self.max_learning_rate = args.max_learning_rate
        # self.num_training_steps = math.ceil((args.num_data - args.skiprows) / self.batch_size)
        
        self.tokenizer = tokenizer
        self.model = BartForConditionalGeneration.from_pretrained(args.model)

        self.metric = evaluate.load("sacrebleu")

        self.save_hyperparameters(
            {
                **self.model.config.to_dict(),
                # "total_steps" : self.num_training_steps,
                # "max_learning_rate": args.max_learning_rate,
                # "min_learning_rate": args.min_learning_rate,
            }
        )
        
        self.preds = []
        self.trgts = []

    def forward(
        self,
        encoder_input_ids,
        encoder_attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        labels=None
    ) :
        outputs = self.model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

        return outputs
    
    def training_step(self, batch, batch_idx, state='train') :
        outputs = self(
            encoder_input_ids=batch['input_ids'].to(self._device),
            encoder_attention_mask=batch['attention_mask'].to(self._device),
            decoder_input_ids=batch['decoder_input_ids'].to(self._device),
            decoder_attention_mask=batch['decoder_input_ids'].to(self._device),
            labels=batch['labels'].to(self._device)
        )

        loss = outputs.loss
        logits = outputs.logits

        preds = np.array(torch.argmax(logits.cpu(), dim=2))
        targets = np.array(batch['labels'].cpu())
        targets = np.where(targets != -100, targets, self.tokenizer.pad_token_id) # -100 to pad tokens

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # [re.sub(r"</s>[\w\W]*", "", pred) for pred in self.tokenizer.batch_decode(preds)]
        decoded_targets = self.tokenizer.batch_decode(targets, skip_special_tokens=True)
        bleu = self.metric.compute(predictions=decoded_preds, references=decoded_targets)['score']

        self.log(f"{state.upper()}_STEP_LOSS", loss, prog_bar=True)
        self.log(f"{state.upper()}_STEP_BLEU", bleu, prog_bar=True)

        return {
            'loss' : loss,
            'bleu' : torch.tensor(bleu)
        }
    
    def validation_step(self, batch, batch_idx, state='valid') :
        outputs = self(
            encoder_input_ids=batch['input_ids'].to(self._device),
            encoder_attention_mask=batch['attention_mask'].to(self._device),
            decoder_input_ids=batch['decoder_input_ids'].to(self._device),
            decoder_attention_mask=batch['decoder_input_ids'].to(self._device),
            labels=batch['labels'].to(self._device)
        )

        loss = outputs.loss
        logits = outputs.logits

        preds = np.array(torch.argmax(logits.cpu(), dim=2))
        targets = np.array(batch['labels'].cpu())
        targets = np.where(targets != -100, targets, self.tokenizer.pad_token_id) # -100 to pad tokens

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # [re.sub(r"</s>[\w\W]*", "", pred) for pred in self.tokenizer.batch_decode(preds)]
        decoded_targets = self.tokenizer.batch_decode(targets, skip_special_tokens=True)
        bleu = self.metric.compute(predictions=decoded_preds, references=decoded_targets)['score']

        self.log(f"{state.upper()}_STEP_LOSS", loss, prog_bar=True)
        self.log(f"{state.upper()}_STEP_BLEU", bleu, prog_bar=True)
        
        self.preds += decoded_preds
        self.trgts += decoded_targets

        return {
            'loss' : loss,
            'bleu' : torch.tensor(bleu)
        }
        # generated = self.model.generate(batch['input_ids'].to(self.device))
        
        # targets = self.tokenizer.batch_decode(batch['decoder_input_ids'], skip_special_tokens=True)
        # preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        # self.preds += preds
        # self.trgts += targets
        
        # bleu = self.metric.compute(predictions=preds, references=targets)['score']
        
        # self.log(f"{state.upper()}_STEP_BLEU", bleu, prog_bar=True)
        
        # return {
        #     'bleu' : torch.tensor(bleu)
        # }

    def validation_epoch_end(self, outputs, state='valid') :
        loss = torch.mean(torch.tensor([output["loss"] for output in outputs]))
        bleu = torch.mean(torch.tensor([output["bleu"] for output in outputs]))
        self.log(f"{state.upper()}_EPOCH_LOSS", loss, prog_bar=True)
        self.log(f"{state.upper()}_EPOCH_BLEU", bleu, prog_bar=True)
        df_dict = {
            'trgts' : self.trgts,
            'preds' : self.preds
        }
        self.df = pd.DataFrame(df_dict)
        self.df.to_csv(f'/home/ailab/Desktop/NY/2023_ipactory/pretrain/09_valid_generation/{self.global_step}_steps.csv')
        
        self.trgts.clear()
        self.preds.clear()
    
    def make_csv(self, state) :
        os.path.join("/home/ailab/Desktop/NY/2023_ipactory/pretrain/07_valid_output/", f"{self.global_step}.csv")
        self.df.to_csv()
    
    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint) :
        save_path = os.path.join("/home/ailab/Desktop/NY/2023_ipactory/pretrain/10_models", f"{self.global_step}_steps")
        # self.model.config.save_step = self.step_count 
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        # self.make_csv(state="valid")
    
    def configure_optimizers(self) :
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.min_learning_rate)
        return {
            'optimizer' : optimizer
        }
        
    

    
    

