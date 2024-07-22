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


train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=16, shuffle=True, drop_last=True)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

# model
dict_args = vars(args)

# autoencoder = QuantizedClassifier(**dict_args)
autoencoder = SlotAttentionAE(**dict_args)
state_dict = torch.load("./clevr10_sp")
autoencoder.load_state_dict(state_dict=state_dict, strict=False)
project_name = 'set_prediction_CLEVR'

wandb_logger = WandbLogger(project='set_prediction_CLEVR', name=f'nums {args.nums!r} s {args.seed}')

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