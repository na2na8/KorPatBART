import os
import numpy as np
import pandas as pd
import re

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

class BARTDataset(Dataset) :
    def __init__(self, path, stage, tokenizer, args) :
        self.tokenizer = tokenizer
        self.max_length = args.max_length
        self.encoder_column = args.encoder_column
        self.decoder_column = args.decoder_column
        
        csv_path = os.path.join(path, stage + '.csv')
        
        self.dataset = pd.read_csv(csv_path)
        self.dataset.drop_duplicates(subset=['claim_eng'], inplace=True)
        self.dataset = self.dataset.dropna(axis=0)
        
    def __len__(self) :
        return len(self.dataset)
    
    def add_padding_data(self, inputs) :
        if len(inputs) < self.max_length :
            pad = np.array([self.tokenizer.pad_token_id] * (self.max_length - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else :
            inputs = inputs[:self.max_length]
        return inputs
    
    def add_ignored_data(self, inputs) :
        if len(inputs) < self.max_length :
            pad = np.array([-100]*(self.max_length - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else :
            inputs = inputs[:self.max_length]
        return inputs
    
    def __getitem__(self, idx) :
        encoder_tokens = self.tokenizer.bos_token + re.sub(r'\([^)]*\)', '', self.dataset[self.encoder_column].iloc[idx])
        encoder_input_ids = self.add_padding_data(self.tokenizer.encode(encoder_tokens))
        
        decoder_tokens =self.tokenizer.bos_token + re.sub(r'\([^)]*\)', '', self.dataset[self.decoder_column].iloc[idx])
        labels = self.tokenizer.encode(decoder_tokens)
        labels.append(self.tokenizer.eos_token_id)
        decoder_input_ids = [self.tokenizer.eos_token_id]
        decoder_input_ids += labels[:-1]
        decoder_input_ids = self.add_padding_data(decoder_input_ids)
        labels = self.add_ignored_data(labels)
        
        encoder_input_ids = torch.from_numpy(np.array(encoder_input_ids, dtype=np.int_))
        encoder_attention_mask = encoder_input_ids.ne(self.tokenizer.pad_token_id).float()
        decoder_input_ids = torch.from_numpy(np.array(decoder_input_ids, dtype=np.int_))
        decoder_attention_mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).float()
        labels = torch.from_numpy(np.array(labels, dtype=np.int_))
        
        return {
            'encoder_input_ids' : encoder_input_ids,
            'encoder_attention_mask' : encoder_attention_mask,
            'decoder_input_ids' : decoder_input_ids,
            'decoder_attention_mask' : decoder_attention_mask,
            'labels' : labels
        }
        
class BARTDataModule(pl.LightningDataModule) :
    def __init__(self, path, args, tokenizer) :
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        
        self.path = path
        
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
        self.setup()
        
    def setup(self, stage=None) :
        self.set_train = BARTDataset(self.path, stage='train', tokenizer=self.tokenizer, args=self.args)
        self.set_valid = BARTDataset(self.path, stage='valid', tokenizer=self.tokenizer, args=self.args)
        self.set_test = BARTDataset(self.path, stage='test', tokenizer=self.tokenizer, args=self.args)
        
    def train_dataloader(self) :
        train = DataLoader(self.set_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self) :
        valid = DataLoader(self.set_valid, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return valid
    
    def test_dataloader(self) :
        test = DataLoader(self.set_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return test