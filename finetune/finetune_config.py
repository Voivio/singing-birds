import numpy as np
import csv
import os

"""A detailed description of all variables is available in /finetune/finetune.py train() function.
"""

workspace = "/content/drive/MyDrive/ECE228/"
pretrained_checkpoint_path = "/content/drive/MyDrive/ECE228/cnn14_birdcall_mAP=5876.pth"
resume_iteration = 0
early_stop = 1000000

sample_rate = 32000

window_size = 1024
hop_size = 320
mel_bins = 64

fmin = 2000
fmax = 10000

batch_size = 16
learning_rate = 1e-3


model_type = "ConvAttention" # Transfer_Cnn14, Transfer_ResNet, ConvAttention. Please notice the combination with pretrained_checkpoint_path.
freeze_base = True #  choose the freeze the base model or not.
cuda = True

loss_type = 'clip_bce'
balanced = "balanced"
augmentation = "mixup"

clip_samples = sample_rate * 10     # Audio clips are 10-second

classes_num = 50
samples_per_class = 100
hdf5_path = os.path.join("/content/drive/MyDrive/ECE228/", "train_new.hdf5")
