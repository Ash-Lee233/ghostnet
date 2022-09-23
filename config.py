
"""
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed

config = ed({
    "num_classes": 1000,
    "batch_size": 128,
    "epoch_size": 500,
    "warmup_epochs": 20,
    "lr_init": 0.1,
    "lr_max": 0.4,
    'lr_end': 1e-6,
    'lr_decay_mode': 'cosine',
    "momentum": 0.9,
    "weight_decay": 4e-5,
    "label_smooth": 0.1,
    "loss_scale": 128,
    "use_label_smooth": True,
    "label_smooth_factor": 0.1,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 20,
    "keep_checkpoint_max": 10,
    "save_checkpoint_path": "./",
})
