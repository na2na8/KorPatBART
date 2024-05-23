import re

import torch
import torch.nn as nn

from rouge import Rouge
import pandas as pd
import pytorch_lightning as pl
from transformers import BartForConditionalGeneration, AdamW

class Translate(pl.LightningModule) :
    def __init__(self, args, tokenizer) :
        super().__init__()
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        
        self.model = BartForConditionalGeneration.from_pretrained(args.model)
        self.tokenizer = tokenizer
        
        self.rouge = Rouge()
        self.save_hyperparameters()
        
        self.preds = []
        self.trgts = []
        self.rouge1 = []
        self.rouge2 = []
        self.rougel = []
        
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
    
    def get_rouge(self, preds, targets) :
        scores = []
        for idx in range(len(preds)) :
            try :
                scores.append(self.rouge.get_scores(preds[idx], targets[idx])[0])
            except ValueError :
                scores.append(
                    {
                        'rouge-1' : {'f' : 0.0},
                        'rouge-2' : {'f' : 0.0},
                        'rouge-l' : {'f' : 0.0}
                    }
                )
        
        rouge1 = torch.tensor([score['rouge-1']['f'] for score in scores])
        rouge2 = torch.tensor([score['rouge-2']['f'] for score in scores])
        rougel = torch.tensor([score['rouge-l']['f'] for score in scores])
        
        return rouge1, rouge2, rougel
    
    def default_step(self, batch, batch_idx, state=None) :
        outputs = self(
            encoder_input_ids=batch['encoder_input_ids'].to(self.device),
            encoder_attention_mask=batch['encoder_attention_mask'].to(self.device),
            decoder_input_ids=batch['decoder_input_ids'].to(self.device),
            decoder_attention_mask=batch['decoder_attention_mask'].to(self.device),
            labels=batch['labels'].to(self.device)
        )

        loss = outputs.loss
        logits = outputs.logits # shape : (batch, len_decoder_inputs, vocab)
        
        preds = torch.argmax(logits, dim=2).to(self.device) # shape : (batch, len_decoder_inputs)
        targets = batch['decoder_input_ids'].to(self.device)
        trgts = self.tokenizer.batch_decode(targets, skip_special_tokens=True)
        preds = self.tokenizer.batch_decode(preds)
        preds = [re.sub(r"</s>[\w\W]*", "", pred) for pred in preds]
        rouge1, rouge2, rougel = self.get_rouge(preds, trgts)
        
        
        
        self.preds += preds
        self.trgts += trgts
        self.rouge1 += [round(r,4) for r in rouge1.tolist()]
        self.rouge2 += [round(r,4) for r in rouge2.tolist()]
        self.rougel += [round(r,4) for r in rougel.tolist()]
        
        rouge1, rouge2, rougel = torch.mean(rouge1), torch.mean(rouge2), torch.mean(rougel)
        
        self.log(f"[{state} loss]", loss, prog_bar=True)
        self.log(f"[{state} rouge1]", rouge1, prog_bar=True)
        self.log(f"[{state} rouge2]", rouge2, prog_bar=True)
        self.log(f"[{state} rougel]", rougel, prog_bar=True)
        
        
        return {
            'loss' : loss,
            'rouge1' : rouge1,
            'rouge2' : rouge2,
            'rougel' : rougel
        }
    
    def default_epoch_end(self, outputs, state=None) :
        loss = torch.mean(torch.tensor([output['loss'] for output in outputs]))
        rouge1 = torch.mean(torch.tensor([output['rouge1'] for output in outputs]))
        rouge2 = torch.mean(torch.tensor([output['rouge2'] for output in outputs]))
        rougel = torch.mean(torch.tensor([output['rougel'] for output in outputs]))
        
        self.log(f'{state}_loss', loss, on_epoch=True, prog_bar=True)
        self.log(f'{state}_rouge1', rouge1, on_epoch=True, prog_bar=True)
        self.log(f'{state}_rouge2', rouge2, on_epoch=True, prog_bar=True)
        self.log(f'{state}_rougel', rougel, on_epoch=True, prog_bar=True)      
        
    def training_step(self, batch, batch_idx, state='train') :
        result = self.default_step(batch, batch_idx, state='train')
        return result
    
    def validation_step(self, batch, batch_idx, state='valid') :
        result = self.default_step(batch, batch_idx, state='valid')
        return result
    
    def training_epoch_end(self, outputs, state='train') :
        self.default_epoch_end(outputs, state)
    
    def validation_epoch_end(self, outputs, state='valid') :
        self.default_epoch_end(outputs, state)
        # df = {
        #     'trgts' : self.trgts,
        #     'preds' : self.preds,
        #     'r1' : self.rouge1,
        #     'r2' : self.rouge2,
        #     'rl' : self.rougel
        # }
        
        # df = pd.DataFrame(df)
        # df.to_csv(f'/home/nykim/23_ipactory/01_downstream/aihub_sum/02_bart_output/valid_{self.current_epoch}.csv')
        
        self.trgts.clear()
        self.preds.clear()
        self.rouge1.clear()
        self.rouge2.clear()
        self.rougel.clear()
        
         
    def test_step(self, batch, batch_idx, state='test') :
        generated = self.model.generate(batch['encoder_input_ids'].to(self.device), max_length=512)
        
        targets = self.tokenizer.batch_decode(batch['decoder_input_ids'], skip_special_tokens=True)
        preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        
        rouge1, rouge2, rougel = self.get_rouge(preds, targets)
        
        self.preds += preds
        self.trgts += targets
        self.rouge1 += [round(r,4) for r in rouge1.tolist()]
        self.rouge2 += [round(r,4) for r in rouge2.tolist()]
        self.rougel += [round(r,4) for r in rougel.tolist()]
        
        rouge1, rouge2, rougel = torch.mean(rouge1), torch.mean(rouge2), torch.mean(rougel)
        
        self.log(f"[{state} rouge1]", rouge1, prog_bar=True)
        self.log(f"[{state} rouge2]", rouge2, prog_bar=True)
        self.log(f"[{state} rougeL]", rougel, prog_bar=True)
        
        return {
            'rouge1' : rouge1,
            'rouge2' : rouge2,
            'rougel' : rougel
        }
        
    def test_epoch_end(self, outputs, state='test') :
        rouge1 = torch.mean(torch.tensor([output['rouge1'] for output in outputs]))
        rouge2 = torch.mean(torch.tensor([output['rouge2'] for output in outputs]))
        rougel = torch.mean(torch.tensor([output['rougel'] for output in outputs]))
        
        self.log(f'{state}_rouge1', rouge1, on_epoch=True, prog_bar=True)
        self.log(f'{state}_rouge2', rouge2, on_epoch=True, prog_bar=True)
        self.log(f'{state}_rougel', rougel, on_epoch=True, prog_bar=True)
        df = {
            'trgts' : self.trgts,
            'preds' : self.preds,
            'r1' : self.rouge1,
            'r2' : self.rouge2,
            'rl' : self.rougel
        }
        
        df = pd.DataFrame(df)
        df.to_csv(f'/home/nykim/23_ipactory/01_downstream/translate/02_bart_output/test_{self.current_epoch}.csv')
        
        self.trgts.clear()
        self.preds.clear()
        self.rouge1.clear()
        self.rouge2.clear()
        self.rougel.clear()
        
    def configure_optimizers(self) :
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=150)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        return [optimizer], [lr_scheduler]