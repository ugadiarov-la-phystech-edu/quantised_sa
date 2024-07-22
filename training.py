import os
import random
import sys

from datasets.stub import Stub

sys.path.append("..")

import numpy as np
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from argparse import ArgumentParser

from datasets import CLEVR
from models import QuantizedClassifier, SlotAttentionAE

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

DEFAULT_SEED = 42
# ------------------------------------------------------------
# Parse args
# ------------------------------------------------------------
parser = ArgumentParser()

parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--rtd_loss_coef', type=float, default=6.)
parser.add_argument('--rtd_lp', type=float, default=2.)
parser.add_argument('--use_weightnorm_sampler', action='store_true')
parser.add_argument('--no-rtd_q_normalize', dest='rtd_q_normalize', action='store_false')
parser.add_argument('--num_slots', type=int, default=10)

# add PROGRAM level args
program_parser = parser.add_argument_group('program')

# logger parameters
program_parser.add_argument("--log_model", default=True)

# dataset parameters
program_parser.add_argument("--train_path", type=str)

# Experiment parameters
program_parser.add_argument("--batch_size", type=int, default=2)
program_parser.add_argument("--from_checkpoint", type=str, default='')
program_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
program_parser.add_argument("--nums", type=int, nargs='+')
program_parser.add_argument("--project", type=str, required=True)
program_parser.add_argument("--group", type=str, required=True)
program_parser.add_argument("--run_name", type=str, required=True)
program_parser.add_argument("--num_workers", type=int, default=4)

# Add model specific args
# parser = SlotAttentionAE.add_model_specific_args(parent_parser=parser)

# Add all the available trainer options to argparse#
# parser = pl.Trainer.add_argparse_args(parser)

# Parse input
args = parser.parse_args()

# ------------------------------------------------------------
# Random
# ------------------------------------------------------------

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.set_float32_matmul_precision('medium')

# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------


train_dataset = CLEVR(images_path=os.path.join(args.train_path, 'images', 'train'),
                    scenes_path= os.path.join(args.train_path, 'scenes', 'CLEVR_train_scenes.json'),
                    max_objs=10)

val_dataset = CLEVR(images_path=os.path.join(args.train_path, 'images', 'val'),
                    scenes_path= os.path.join(args.train_path, 'scenes', 'CLEVR_val_scenes.json'),
                    max_objs=10)

# train_dataset = Stub()
# val_dataset = Stub()


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

# model
dict_args = vars(args)

# autoencoder = QuantizedClassifier(**dict_args)
autoencoder = SlotAttentionAE(**dict_args)
state_dict = torch.load("./clevr10_sp")
autoencoder.load_state_dict(state_dict=state_dict, strict=False)

wandb_logger = WandbLogger(project=args.project, group=args.group, name=args.run_name)

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

monitor = 'Validation MSE'

# checkpoints
save_top_k = 1
checkpoint_callback = ModelCheckpoint(every_n_epochs=10)

# Learning rate monitor
lr_monitor = LearningRateMonitor(logging_interval='step')


callbacks = [
    checkpoint_callback,
    lr_monitor,
]

# ------------------------------------------------------------
# Trainer
# ------------------------------------------------------------
# trainer parameters
profiler = 'simple'  # 'simple'/'advanced'/None

# trainer
trainer = pl.Trainer(max_epochs=args.max_epochs,
                     profiler=profiler,
                     callbacks=callbacks,
                     logger=wandb_logger)

if not len(args.from_checkpoint):
    args.from_checkpoint = None

# Train
trainer.fit(autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.from_checkpoint)