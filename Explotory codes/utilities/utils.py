import numpy as np
import argparse
import csv
import os
import glob
import datetime
import time
import logging
import h5py
import librosa

import pandas as pd

def create_folder(fd):
    """Creates the folders with given path.
    
    Arguments:
        fd - Directory path.
    """
    if not os.path.exists(fd):
        os.makedirs(fd)

def create_logging(log_dir, filemode):
    """Creates the gile logger.

    Arguments:
        log_dir - The directory of the log files.
        filemode - Specifies the open mode of the file.

    Returns:
        Configured logging object.
    """
    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, '{:04d}.log'.format(i1))):
        i1 += 1
        
    log_path = os.path.join(log_dir, '{:04d}.log'.format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging

def read_metadata(csv_path):
    """Read the metadata of the dataset.

    Arguments:
        csv_path - Path of the csv metadata.

    Returns:
        meta_dict: {'audio_name': (audios_num,), 'target': (audios_num, classes_num)}
    """
    df = pd.read_csv(csv_path)

    audios_num = len(df)

    all_classes = df["ebird_code"].unique()
    classes_num = len(all_classes)

    targets = np.zeros((audios_num,), dtype=np.int32)
    audio_names = df["title"].apply(lambda x: f"{x.split(' ')[0]}.wav").values

    for i in range(classes_num):
        targets[df["ebird_code"] == all_classes[i]] = i
    
    meta_dict = {'audio_name': audio_names, "bird_codes": all_classes, 'target': targets}
    return meta_dict

def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0 : audio_length]

def float32_to_int16(x):
    assert np.max(np.abs(x)) <= 1.2
    x = np.clip(x, -1, 1)
    return (x * 32767.).astype(np.int16)