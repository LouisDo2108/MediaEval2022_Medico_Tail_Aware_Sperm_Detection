#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time : 2022-08-22
@Author : Nguyen Huu Hung
@File : train.py
"""
import gc
from itertools import cycle
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import FileStorageObserver
from torch.utils.tensorboard import SummaryWriter

from linear_model import SimpleDataset, SimpleModel

ex = Experiment('simple_softmax')


@ex.config
def config():
    # Train/Validation Params
    n_epoch = 1000
    train_interval = 10
    resume_iteration = None
    validation_interval = 100
    checkpoint_interval = 200
    seed = 1810

    # Cache params
    is_cache_train = False
    n_file_train = 1000
    is_cache_val = False
    n_file_val = 10

    logdir = f'runs/train_simple_softmax-' + datetime.now().strftime(
        '%m%d%m%y-%H%M%S')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Optimizer setup
    learning_rate = 0.001
    learning_rate_decay_steps = 10000
    learning_rate_decay_rate = 0.98
    clip_gradient_norm = 3

    # Model Params
    is_softmax = True

    ex.observers.append(FileStorageObserver.create(logdir))
    pass


@ex.automain
def train(logdir, learning_rate, n_epoch, train_interval, validation_interval,
          checkpoint_interval, is_softmax):
    print_config(ex.current_run)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    train_dataset = SimpleDataset(group="train", save_path="train_feats.csv")
    val_dataset = SimpleDataset(group="validation", save_path="val_feats.csv")
    train_loader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              drop_last=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=len(val_dataset),
                            shuffle=True,
                            drop_last=False)

    model = SimpleModel(is_softmax=is_softmax)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    criterion = nn.L1Loss()
    print(
        f"Train sample: {len(train_dataset)} / Val sample: {len(val_dataset)} "
    )
    for epoch in range(1, n_epoch + 1):
        for batch in train_loader:
            pred = model(batch['data'])
            loss = criterion(pred, batch['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pass
        print(f"[{epoch:>5d}/{n_epoch:>5d}]\n" +
              f"├── train_loss    : {loss:>7f}")
        if epoch % train_interval == 0:
            writer.add_scalar("train/loss", loss, global_step=epoch)

        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['data'])
                val_loss = criterion(pred, batch['label'])
                print(f"[{epoch:>5d}/{n_epoch:>5d}]\n" +
                      f"├── val_loss    : {val_loss:>7f}")
            if epoch % validation_interval == 0:
                writer.add_scalar("validation/loss",
                                  val_loss,
                                  global_step=epoch)
            if epoch % checkpoint_interval == 0:
                model_path = os.path.join(logdir, f"model-{epoch}.pt")
                torch.save(model, model_path)
        pass
    pass
