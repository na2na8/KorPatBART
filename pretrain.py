import argparse
import math
import os
import random

from pytorch_lightning import loggers as pl_loggers
import torch
from transformers import AutoTokenizer

from utils.callback import CheckpointEveryNSteps
from PretrainDataModule import *
from PatentPretrain import *

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
    # g1.add_argument("--path", type=str, default='/home/ailab/Desktop/NY/2023_ipactory/data/csv/data.csv')
    # g1.add_argument("--path", type=str, default="/home/ailab/Desktop/NY/2023_ipactory/data/dedup")
    g1.add_argument("--path", type=str, default="/home/ailab/Desktop/NY/2023_ipactory/data/dedup/train")
    g1.add_argument("--tokenizer", type=str, default="gogamza/kobart-base-v1")
    g1.add_argument("--model", type=str, default="gogamza/kobart-base-v1")
    g1.add_argument("--ckpt_path", type=str, default="/home/ailab/Desktop/NY/2023_ipactory/pretrain/08_ckpt/step=720000-LAST-VALID_EPOCH_BLEU=56.489-TRAIN_STEP_BLEU=59.845-TRAIN_STEP_LOSS=0.692.ckpt")
    # g1.add_argument("--ckpt_path", type=str, default='/home/ailab/Desktop/NY/2023_ipactory/ckpt/test/step=000100-TRAIN_STEP_LOSS=4.630-TRAIN_STEP_BLEU=0.706.ckpt')
    # g1.add_argument("--ckpt_last_step", type=int, default=0)
    g1.add_argument("--gpu_lists", type=str, default="0", help="string; make list by splitting by ','") # gpu list to be used

    g2 = parser.add_argument_group("TrainingArgument")
    g2.add_argument("--ckpt_dir", type=str, default='/home/ailab/Desktop/NY/2023_ipactory/pretrain/08_ckpt')
    g2.add_argument("--epochs", type=int, default=1)
    # g2.add_argument("--num_warmup_steps", type=int, default=10000)
    # g2.add_argument("--skiprows", type=int, default=3500000) # num of validation
    g2.add_argument("--max_length", type=int, default=512)
    
    g2.add_argument("--batch_size", type=int, default=16)
    # g2.add_argument("--batch_size", type=int, default=4)
    
    g2.add_argument("--mask_ratio", type=float, default=0.3)
    g2.add_argument("--poisson_lambda", type=float, default=3.0)
    
    # g2.add_argument("--chunksize", type=int, default=1000000)
    # g2.add_argument("--chunksize", type=int, default=1000)
    
    # g2.add_argument("--num_data", type=int, default=35091902) # total num of data
    g2.add_argument("--min_learning_rate", type=float, default=1e-5)
    # g2.add_argument("--max_learning_rate", type=float, default=1e-4)
    g2.add_argument("--num_workers", type=int, default=8)
    g2.add_argument("--logging_dir", type=str, default='/home/ailab/Desktop/NY/2023_ipactory/pretrain/08_ckpt')

    args = parser.parse_args()

    set_random_seed(random_seed=42)

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # trained_rows = args.ckpt_last_step * args.batch_size
    
    print("DM set up...")
    dm = PretrainDataModule(args.path, tokenizer, args)
    # dm = PretrainDataModule(args.path, args.skiprows, trained_rows , args.chunksize, args.num_data, tokenizer, args)
    # dm = PretrainDataModule(args.path, args.skiprows, 0 , 10000, args.num_data, tokenizer, args)
    # dm.setup()
    # t = next(iter(train))

    gpu_list = [int(gpu) for gpu in args.gpu_lists.split(',')]

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                                monitor='VALID_EPOCH_BLEU',
                                dirpath=args.ckpt_dir,
                                filename='{step:06d}-{VALID_EPOCH_BLEU:.3f}-{TRAIN_STEP_BLEU:.3f}-{TRAIN_STEP_LOSS:.3f}',
                                verbose=False,
                                save_last=True,
                                every_n_train_steps=10000,
                                mode='max',
                                save_top_k=10
                            )
    checkpoint_callback.CHECKPOINT_NAME_LAST = '{step:06d}-LAST-{VALID_EPOCH_BLEU:.3f}-{TRAIN_STEP_BLEU:.3f}-{TRAIN_STEP_LOSS:.3f}'
    # checkpoint_callback = CheckpointEveryNSteps(
    #                             save_step_frequency=500,
    #                             ckpt_path=args.ckpt_path,
    #                             prefix="500-Step-Checkpoint",
    #                         )
    
    tb_logger = pl_loggers.TensorBoardLogger(args.logging_dir)
    lr_logger = pl.callbacks.LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_logger],
        default_root_dir=args.ckpt_dir,
        log_every_n_steps=100,
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=[0],
        val_check_interval=100000
    )
    
    model = PatentPretrain(args, device, tokenizer)
    trainer.fit(model, dm, ckpt_path=args.ckpt_path)