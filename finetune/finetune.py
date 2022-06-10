import os
import sys

# sys.path.insert(1, os.path.join(sys.path[0], './utils'))

import numpy as np
import argparse
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from utils.utilities import (create_folder, create_logging, Mixup,
                       StatisticsContainer)

from utils.pytorch_utils import (move_data_to_device, count_parameters, do_mixup)

from utils.evaluate import Evaluator
from utils.losses import get_loss_func

import finetune_config
from bird_data_generator import (BirdDataSet, BirdSampler, BirdEvaluateSampler, bird_collate_fn)
from finetune_models import (Transfer_Cnn14, Transfer_ResNet, ConvAttention)


def train():
    """Train AudioSet tagging model. 

    Hyper-parameters (stored in `fintune_config.py`):
      workspace: str

      sample_rate: int, specifies the sample rate of audio samples
      window_size: int, the window_size used in STFT
      hop_size: int, the hop_size used in STFT
      mel_bins: int, the mel_bins in Log-Mel transformation
      fmin: int, the minimal frequency of the samples
      fmax: int, the maximal frequency of the samples
      model_type: str, spacifies the backend (architecture)

      loss_type: 'clip_bce'
      augmentation: 'none' | 'mixup', use augmentation or not
      batch_size: int
      learning_rate: float

      resume_iteration: int, whether to resume previous training
      early_stop: int, whether to use early stop
      cuda: bool, whether to use GPU or not

      pretrained_checkpoint_path: str, specifies the path to pretrained weight
      freeze_base: bool, whether to freeze the parameters in the base model

      classes_num: int, the number of classes to be classified
      samples_per_class: int, the number of samples in each class
      hdf5_path: str, the path to the hdf5 file of all audio data
    """

    # Arugments & parameters
    workspace = finetune_config.workspace

    # Model hyperparameters
    sample_rate = finetune_config.sample_rate
    window_size = finetune_config.window_size
    hop_size = finetune_config.hop_size
    mel_bins = finetune_config.mel_bins
    fmin = finetune_config.fmin
    fmax = finetune_config.fmax
    model_type = finetune_config.model_type

    # Optimizer parameters
    loss_type = finetune_config.loss_type
    augmentation = finetune_config.augmentation
    batch_size = finetune_config.batch_size
    learning_rate = finetune_config.learning_rate

    # Train procedure settings
    resume_iteration = finetune_config.resume_iteration
    early_stop = finetune_config.early_stop
    device = torch.device('cuda') if finetune_config.cuda and torch.cuda.is_available() else torch.device('cpu')
    filename = "finetune"

    # Arguments relatecd with pretrained data.
    pretrained_checkpoint_path = finetune_config.pretrained_checkpoint_path
    freeze_base = finetune_config.freeze_base
    pretrain = True if pretrained_checkpoint_path else False

    num_workers = 1
    clip_samples = finetune_config.clip_samples
    classes_num = finetune_config.classes_num
    samples_per_class = finetune_config.samples_per_class
    hdf5_path = finetune_config.hdf5_path
    loss_func = get_loss_func(loss_type)

    # Paths
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename,
                                   'sr={},ws={},hs={},mel={},fmin={},fmax={},model={},batch={}'.format(
                                       sample_rate, window_size, hop_size, mel_bins, fmin, fmax, model_type,
                                       batch_size))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename,
                                   'sr={},ws={},hs={},mel={},fmin={},fmax={},model={},batch={}'.format(
                                       sample_rate, window_size, hop_size, mel_bins, fmin, fmax, model_type,
                                       batch_size),
                                   'statistics.pkl')
    create_folder(os.path.dirname(statistics_path))

    logs_dir = os.path.join(workspace, 'logs', filename,
                            'sr={},ws={},hs={},mel={},fmin={},fmax={},model={},batch={}'.format(
                                sample_rate, window_size, hop_size, mel_bins, fmin, fmax, model_type, batch_size))
    create_logging(logs_dir, filemode='w')

    if 'cuda' in str(device):
        logging.info('Using GPU.')
        device = 'cuda'
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
        device = 'cpu'

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size,
                  hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax,
                  classes_num=classes_num, freeze_base=freeze_base)
    # Load pretrained model
    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    params_num = count_parameters(model)
    # flops_num = count_flops(model, clip_samples)
    logging.info('Parameters num: {}'.format(params_num))
    # logging.info('Flops num: {:.3f} G'.format(flops_num / 1e9))

    # Dataset will be used by DataLoader later. Dataset takes a meta as input 
    # and return a waveform and a target.
    dataset = BirdDataSet(hdf5_path=hdf5_path, sample_rate=sample_rate)

    train_sampler = BirdSampler(
        hdf5_path=hdf5_path,
        indices_range=(0, 0.8),
        batch_size=batch_size,
        samples_per_class=samples_per_class
    )
    val_sampler = BirdEvaluateSampler(
        hdf5_path=hdf5_path,
        indices_range=(0.8, 0.9),
        batch_size=batch_size,
        samples_per_class=samples_per_class
    )
    test_sampler = BirdEvaluateSampler(
        hdf5_path=hdf5_path,
        indices_range=(0.9, 1),
        batch_size=batch_size,
        samples_per_class=samples_per_class
    )

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                               batch_sampler=train_sampler, collate_fn=bird_collate_fn,
                                               num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_sampler=val_sampler, collate_fn=bird_collate_fn,
                                             num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_sampler=test_sampler, collate_fn=bird_collate_fn,
                                              num_workers=num_workers, pin_memory=True)

    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)

    # Evaluator
    evaluator = Evaluator(model=model)

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    train_bgn_time = time.time()

    # Resume training
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(workspace, 'checkpoints', filename,
                                              'sr={},ws={},hs={},mel={},fmin={},fmax={},model={},batch={}'.format(
                                                  sample_rate, window_size, hop_size, mel_bins, fmin, fmax, model_type,
                                                  batch_size),
                                              '{}_iterations.pth'.format(resume_iteration))

        logging.info('Loading checkpoint {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        train_sampler.load_state_dict(checkpoint['sampler'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = checkpoint['iteration']

    else:
        iteration = 0

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    time1 = time.time()

    all_losses = []
    for batch_data_dict in train_loader:
        """batch_data_dict: {
            'audio_name': (batch_size [*2 if mixup],), 
            'waveform': (batch_size [*2 if mixup], clip_samples), 
            'target': (batch_size [*2 if mixup], classes_num), 
            (ifexist) 'mixup_lambda': (batch_size * 2,)}
        """
        # Evaluate
        if (iteration % 100 == 0 and iteration > resume_iteration) or (iteration == 0):
            train_fin_time = time.time()

            bal_statistics = evaluator.evaluate(val_loader)
            test_statistics = evaluator.evaluate(test_loader)

            logging.info('Validate bal mAP: {:.3f}'.format(
                np.mean(bal_statistics['average_precision'])))
            logging.info('Validate test mAP: {:.3f}'.format(
                np.mean(test_statistics['average_precision'])))

            statistics_container.append(iteration, bal_statistics, data_type='bal')
            statistics_container.append(iteration, test_statistics, data_type='test')
            statistics_container.dump()

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'iteration: {}, train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(iteration, train_time, validate_time))

            logging.info('------------------------------------')

            train_bgn_time = time.time()

        # Save model
        if iteration % 100 == 0:
            checkpoint = {
                'iteration': iteration,
                'model': model.module.state_dict(),
                'sampler': train_sampler.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))

            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))

        # Mixup lambda
        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(
                batch_size=len(batch_data_dict['waveform']))

        # Move data to device
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

        # Forward
        model.train()

        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'],
                                      batch_data_dict['mixup_lambda'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': do_mixup(batch_data_dict['target'],
                                                    batch_data_dict['mixup_lambda'])}
            """{'target': (batch_size, classes_num)}"""
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': batch_data_dict['target']}
            """{'target': (batch_size, classes_num)}"""

        # Loss
        loss = loss_func(batch_output_dict, batch_target_dict)

        # Backward
        loss.backward()
        print("Loss : {}".format(loss))
        all_losses.append(loss.detach().clone().item())
        print("All losses:")
        print(all_losses)

        optimizer.step()
        optimizer.zero_grad()

        if iteration % 10 == 0:
            print('--- Iteration: {}, train time: {:.3f} s / 10 iterations ---' \
                  .format(iteration, time.time() - time1))
            time1 = time.time()

        # Stop learning
        if iteration == early_stop:
            break

        iteration += 1


if __name__ == '__main__':
    train()
