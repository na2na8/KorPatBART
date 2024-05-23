import argparse
import math
import os
import random
import numpy as np

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer

from BARTDataModule import BARTDataModule
from BARTTranslate_2 import Translate

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)

    pl.seed_everything(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Set random seed : {random_seed}")
    
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    g1 = parser.add_argument_group("CommonArgument")
    g1.add_argument("--path", type=str, default="/home/nykim/23_ipactory/01_downstream/translate/data")
    
    g1.add_argument("--tokenizer", type=str, default="gogamza/kobart-base-v1")
    g1.add_argument("--model", type=str, default="gogamza/kobart-base-v1")
    # g1.add_argument("--tokenizer", type=str, default="/home/ailab/Desktop/NY/2023_ipactory/pretrain/10_models/2500000_steps")
    # g1.add_argument("--model", type=str, default="/home/ailab/Desktop/NY/2023_ipactory/pretrain/10_models/2500000_steps")
    
    g1.add_argument("--ckpt_path", type=str, default="/home/nykim/23_ipactory/01_downstream/translate/01_bart_ckpt/bart_abst/epoch=04-valid_rouge1=0.510-valid_rouge2=0.365-valid_rougel=0.502.ckpt")
    g1.add_argument("--gpu_lists", type=str, default="0", help="string; make list by splitting by ','") # gpu list to be used

    g2 = parser.add_argument_group("TrainingArgument")
    g2.add_argument("--save_ckpt_dir", type=str, default='/home/nykim/23_ipactory/01_downstream/translate/01_bart_ckpt')
    g2.add_argument("--epochs", type=int, default=5)
    g2.add_argument("--max_length", type=int, default=512)
    g2.add_argument("--batch_size", type=int, default=16)
    g2.add_argument("--learning_rate", type=float, default=1e-5)
    g2.add_argument("--num_workers", type=int, default=8)
    g2.add_argument("--logging_dir", type=str, default='/home/nykim/23_ipactory/01_downstream/translate/01_bart_ckpt')
    
    g2.add_argument("--encoder_column", type=str, default="claim_kor")
    g2.add_argument("--decoder_column", type=str, default="astrt_cont_kor")
    # g2.add_argument("--encoder_column", type=str, default="astrt_cont_eng")
    # g2.add_argument("--decoder_column", type=str, default="astrt_cont_kor")
    # g2.add_argument("--encoder_column", type=str, default="claim_eng")
    # g2.add_argument("--decoder_column", type=str, default="claim_kor")
    g2.add_argument("--score_mode", type=str, default="rouge")

    args = parser.parse_args()

    set_random_seed(random_seed=42)

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    print("DM set up...")
    dm = BARTDataModule(args.path, args, tokenizer)
    dm = dm.test_dataloader()
    gpu_list = [int(gpu) for gpu in args.gpu_lists.split(',')]

    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #                             monitor='valid_rouge2',
    #                             dirpath=args.save_ckpt_dir,
    #                             filename='{epoch:02d}-{valid_rouge1:.3f}-{valid_rouge2:.3f}-{valid_rougel:.3f}',
    #                             verbose=False,
    #                             save_last=True,
    #                             mode='max',
    #                             save_top_k=1
    #                         )
    
    tb_logger = pl_loggers.TensorBoardLogger(args.logging_dir)
    # lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        logger=tb_logger,
        # callbacks=[checkpoint_callback, lr_logger],
        default_root_dir=args.save_ckpt_dir,
        # max_epochs=args.epochs,
        accelerator='gpu',
        devices=[0],
    )
    
    model = Translate.load_from_checkpoint(checkpoint_path=args.ckpt_path, args=args, tokenizer=tokenizer)
    trainer.test(model, dm)