'''MIT License
Copyright (C) 2020 Prokofiev Kirill, Intel Corporation
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.'''

import json
import logging
import os
import os.path as osp
import sys
from importlib import import_module

import numpy as np
import torch
from attrdict import AttrDict as adict
from torch.utils.data import DataLoader

def load_checkpoint(checkpoint_path, net, map_location, optimizer=None, load_optimizer=False, strict=True):
    ''' load a checkpoint of the given model. If model is using for training with imagenet weights provided by
        this project, then delete some wights due to mismatching architectures'''
    print("\n==> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        unloaded = net.load_state_dict(checkpoint['state_dict'], strict=strict)
        missing_keys, unexpected_keys = (', '.join(i) for i in unloaded)
    else:
        unloaded = net.load_state_dict(checkpoint, strict=strict)
        missing_keys, unexpected_keys = (', '.join(i) for i in unloaded)
    if missing_keys or unexpected_keys:
        logging.warning(f'THE FOLLOWING KEYS HAVE NOT BEEN LOADED:\n\nmissing keys: {missing_keys}\
            \n\nunexpected keys: {unexpected_keys}\n')
        print('proceed traning ...')
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'epoch' in checkpoint:
        return checkpoint['epoch']
