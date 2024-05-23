import os
import re
import math
import numpy as np
from itertools import chain
import datasets
from datasets import Dataset
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, IterableDataset

class PretrainDataset(IterableDataset) :
    def __init__(self, path : str, tokenizer, args) :
        # self.arrow_list = self.get_arrow_files(path)
        
        # 밑에 거 안 씀
        # self.data = self.get_ds(self.arrow_list)
        # self.data = iter(datasets.load_from_disk(path)['train']['text'])
        
        self.max_length = args.max_length
        self.mask_ratio = args.mask_ratio
        self.poisson_lambda = args.poisson_lambda
        
        self.tokenizer = tokenizer
        self.bos = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        self.eos = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        self.pad = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        self.rest = []
        self.truncated = []
    
        self.data = datasets.load_dataset(path)['train']
        self.idx = 0
        
    def __iter__(self) :
        for idx in range(len(self.data)) :
            text = self.data[idx]['text']
            self.truncate(text)
            for trunc in self.truncated :
                items = self.get_inputs(trunc)
                yield items
            self.truncated = []
        
        items = self.get_inputs(self.rest)
        yield items
        
    
    # def __iter__(self) :

    #     for arrow in self.arrow_list :
    #         data = iter(Dataset.from_file(arrow)['text'])
            
    #         while True :
    #             try :
    #                 text = next(data)
    #                 self.truncate(text)
                    
    #                 for trunc in self.truncated :
    #                     items = self.get_inputs(trunc)
    #                     # decoder_input = trunc # token id list
    #                     # encoder_input = self.masking(decoder_input)
                        
    #                     # # padding
    #                     # input_ids = torch.tensor(encoder_input + [self.pad]*(self.max_length - len(encoder_input)))
    #                     # attention_mask = torch.tensor([1]*len(encoder_input) + [0]*(self.max_length - len(encoder_input)))
    #                     # decoder_input_ids = torch.tensor(decoder_input + [self.pad]*(self.max_length - len(decoder_input)))
    #                     # decoder_attention_mask = torch.tensor([1]*len(decoder_input) + [0]*(self.max_length - len(decoder_input)))
    #                     # labels = torch.tensor([1]*len(decoder_input) + [-100]*(self.max_length - len(decoder_input)))
                        
    #                     # items = {
    #                     #     'input_ids' : input_ids,
    #                     #     'attention_mask' : attention_mask,
    #                     #     'decoder_input_ids' : decoder_input_ids,
    #                     #     'decoder_attention_mask' : decoder_attention_mask,
    #                     #     'labels' : labels
    #                     # }
    #                     yield items
                        
    #             except StopIteration :
    #                 break
                
    #         if arrow == self.arrow_list[-1] :
    #             items = self.get_inputs(self.rest)
    #             yield items
                    
    def get_inputs(self, trunc : list) -> dict :
        decoder_input = trunc # token id list
        encoder_input = self.masking(decoder_input)
        
        # padding
        input_ids = torch.tensor(encoder_input[:512] + [self.pad]*(self.max_length - len(encoder_input)))
        attention_mask = torch.tensor([1]*len(encoder_input[:512]) + [0]*(self.max_length - len(encoder_input)))
        decoder_input_ids = torch.tensor(decoder_input + [self.pad]*(self.max_length - len(decoder_input)))
        decoder_attention_mask = torch.tensor([1]*len(decoder_input) + [0]*(self.max_length - len(decoder_input)))
        # labels = torch.tensor(decoder_input + [-100]*(self.max_length - len(decoder_input))) # <s> text </s>
        labels = torch.tensor(decoder_input[1:] + [self.eos] + [-100]*(self.max_length - len(decoder_input))) # <s> text </s>
        
        try :
            assert len(input_ids) == self.max_length and len(attention_mask) == self.max_length \
                and len(decoder_input_ids) == self.max_length and len(decoder_attention_mask) == self.max_length \
                and len(labels) == self.max_length, \
                f'input_ids : {len(input_ids)}, attention_mask : {len(attention_mask)}, decoder_input_ids : {len(decoder_input_ids)} \
                    decoder_attention_mask : {len(decoder_attention_mask)}, labels : {len(labels)}'
        except AssertionError :
            print(f'input_ids : {input_ids}\nattention_mask : {attention_mask}\ndecoder_input_ids : {decoder_input_ids}\ndecoder_attention_mask : {decoder_attention_mask}\nlabels : {labels}')
        
        items = {
            'input_ids' : input_ids,
            'attention_mask' : attention_mask,
            'decoder_input_ids' : decoder_input_ids,
            'decoder_attention_mask' : decoder_attention_mask,
            'labels' : labels
        }
        return items              
        
    # def __iter__(self) :
    #     if not self.truncated :
    #         text = next(self.data)
    #         self.truncate(text)
            
            
    #     if self.truncated :
    #         decoder_input = self.truncated[0] # [0, ~, 1]
    #         encoder_input = self.masking(decoder_input) # [0, ~, 1] with <mask>
            
    #     self.truncated = self.truncated[1:]
        
    #     # padding
    #     input_ids = torch.tensor(encoder_input + [self.pad]*(self.max_length - len(encoder_input)))
    #     attention_mask = torch.tensor([1]*len(encoder_input) + [0]*(self.max_length - len(encoder_input)))
    #     decoder_input_ids = torch.tensor(decoder_input + [self.pad]*(self.max_length - len(decoder_input)))
    #     decoder_attention_mask = torch.tensor([1]*len(decoder_input) + [0]*(self.max_length - len(decoder_input)))
    #     labels = torch.tensor([1]*len(decoder_input) + [-100]*(self.max_length - len(decoder_input)))
        
    #     items = {
    #         'input_ids' : input_ids,
    #         'attention_mask' : attention_mask,
    #         'decoder_input_ids' : decoder_input_ids,
    #         'decoder_attention_mask' : decoder_attention_mask,
    #         'labels' : labels
    #     }
    #     yield items
            
            
    def masking(self, decoder_input : list) -> list:
        # text = self.tokenizer.decode(decoder_input[1:-1]) # <s> text </s>
        text = self.tokenizer.decode(decoder_input[2:])
        
        tokens = text.split(' ')
        len_doc = len(tokens)
        to_mask = math.ceil(len(tokens) * self.mask_ratio)
        num_masked = 0
        
        while num_masked < to_mask :
            len_span = np.minimum(np.random.poisson(lam=self.poisson_lambda), len_doc)
            mask_start_idx = int(np.random.uniform(0, len_doc - len_span))
            
            tokens = np.concatenate(
                [
                    tokens[:mask_start_idx],
                    ['<mask>'],
                    tokens[mask_start_idx + len_span:]
                ],
                axis=None
            )
            
            len_doc -= len_span - 1
            num_masked += len_span
        noised_doc = self.tokenizer.bos_token + ' '.join(tokens) #+ self.tokenizer.eos_token
        noised_doc = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(noised_doc))
        return noised_doc
            
    
    def truncate(self, text : str) -> list :
        '''
        text -> [tokenized list]
        '''
        # text = re.sub(r'\([\d]+\)', '', text)
        text = re.sub('\([^)]+\)', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'\n{2,}', '\n', text)
        text = text.replace('(', ' ( ')
        text = text.replace(')', ' ) ')
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        
        while len(tokens) >= self.max_length :
            trunc_size = self.max_length - len(self.rest) - 2 
            if trunc_size < 0 : trunc_size = 0
            #if not self.rest else self.max_length - len(self.rest) - 2#- 1
            trunc = tokens[:trunc_size]
            
            if len(self.rest) <= self.max_length - 2 :
                # concat
                trunc = self.rest + tokens[:trunc_size]
                self.rest = []
            elif len(self.rest) > self.max_length - 2 :
                trunc = self.rest[:self.max_length - 2]
                self.rest = self.rest[self.max_length - 2 :]
            tokens = tokens[trunc_size:]
            
            trunc = [self.eos, self.bos] + trunc # </s><s> text
            assert len(trunc) == self.max_length, f"length of truncation is {len(trunc)}, not {self.max_length}.\n"
            
            self.truncated.append(trunc)
        self.rest += tokens
        while len(self.rest) >= self.max_length :
            trunc = [self.eos, self.bos] + self.rest[:self.max_length - 2] # </s><s> text
            self.rest = self.rest[self.max_length - 2:]
            self.truncated.append(trunc)
        
    def get_arrow_files(self, path : str) :
        arrow_files = sorted(list(filter(lambda x : x.endswith('.arrow'), [os.path.join(path, arrow) for arrow in os.listdir(path)])))
        return arrow_files
    
    def get_ds(self, arrow_files) :
        iter_data = []
        for arrow_file in arrow_files :
            ds = Dataset.from_file(arrow_file)
            iter_data.append(iter(ds['text']))
        
        final_iter = iter_data[0]
        iter_data = iter_data[1:]
        for iter_d in iter_data :
            final_iter = chain(final_iter, iter_d)
            
        return final_iter    
    
class PretrainDataModule(pl.LightningDataModule) :
    def __init__(self, path : str, tokenizer, args) :
        '''
        path : path
        tokenizer
        args
        '''
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        
        self.setup()

    def setup(self, stage=None) :
        self.set_train = PretrainDataset(
                                            "/home/ailab/Desktop/NY/2023_ipactory/data/03_orig_train", 
                                            self.tokenizer, 
                                            self.args
                                        )
        self.set_valid = PretrainDataset(
                                            "/home/ailab/Desktop/NY/2023_ipactory/data/04_orig_valid",
                                            self.tokenizer,
                                            self.args
                                        )

    def train_dataloader(self) :
        train = DataLoader(
            self.set_train, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=False,
            multiprocessing_context="fork",
            # shuffle=True
        )
        return train
    
    def val_dataloader(self) :
        valid = DataLoader(
            self.set_valid, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            multiprocessing_context="fork",
            pin_memory=False
        )
        return valid