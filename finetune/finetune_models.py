from utils.pytorch_utils import do_mixup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

import math
from collections import OrderedDict

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
sys.path.insert(1, os.path.join(sys.path[0], '../pytorch'))


"""Utility functions for initializing layers.
"""


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_layer_identity(layer):
    """Initialize a Linear or Convolutional layer with identity mapping"""
    # print("Identity init {}".format(layer))
    nn.init.eye_(layer.weight[:, :, 0])

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


"""Basic building blocks of models
"""


def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)


def conv1x1(in_planes, out_planes):
    """1x1 convolution
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


"""Attention block
"""

# Initialize the attention module as a nn.Module subclass


class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # TODO: Implement the Key, Query and Value linear transforms as 1x1 convolutional layers
        # Hint: channel size remains constant throughout
        self.conv_query = nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_key = nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_value = nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=1)

    def forward(self, x):
        N, C, H, W = x.shape

        # Pass the input through conv_query, reshape the output volume to (N, C, H*W)
        q = self.conv_query(x.view(N, C, -1))
        # Pass the input through conv_key, reshape the output volume to (N, C, H*W)
        k = self.conv_key(x.view(N, C, -1))
        # Pass the input through conv_value, reshape the output volume to (N, C, H*W)
        v = self.conv_value(x.view(N, C, -1))
        # Implement the above formula for attention using q, k, v, C
        attention = torch.bmm(
            # softmax(qk^T / sqrt(c))
            torch.softmax(
                torch.bmm(
                    # qk^T / sqrt(c)
                    q,
                    torch.transpose(k, 1, 2)
                ) / math.sqrt(C), dim=-1),
            # * v
            v
        )
        # Reshape the output to (N, C, H, W) before adding to the input volume
        attention = attention.reshape(N, C, H, W)
        return x + attention


"""Encapsulation of the basic blocks
"""


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_Attention(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock_Attention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.attention = Attention(in_channels=out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer_identity(self.attention.conv_query)
        init_layer_identity(self.attention.conv_key)
        init_layer_identity(self.attention.conv_value)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.relu_(self.attention(x))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock5x5(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(ConvBlock5x5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


"""ResNet building blocks
"""
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample = None,  batch_norm: bool = True):
        """
        Define a residual block used in ResNet.
        
        Inputs:
        - in_channels: Number of channels in the input image
        - out_channels: Number of channels produced by the convolution
        - stride: Strides of the convolution layers. 
        A tuple specifies the stride for each layer. Default: (1, 1)
        - downsample: A nn.Module to complete downsample, in order to match dimensions.
        - batch_norm: If `true`, batch normalization is added before activation functions
        
        Returns: Tensor after computation.
        """
        super(ResidualBlock, self).__init__()
        self.batch_norm = batch_norm

        self.stride = stride
        self.downsample = downsample

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels, kernel_size=3, stride=1,
            padding=1, groups=1, bias=False, dilation=1)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, groups=1, bias=False, dilation=1)

        if batch_norm:
            # enable batch norm before activation
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

    def init_weights(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        # padding dimensions are described starting from the last dimension and moving forward
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)

        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=(2, 2))
        else:
            x = x

        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(shortcut + x)

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, batch_norm=True):
        super(ResNet, self).__init__()

        self.batch_norm = batch_norm

        self.in_channels = 64

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels:
            models = []
            if stride == 1:
                models.append(nn.Conv2d(
                    self.in_channels, out_channels, kernel_size=1, stride=1, bias=False)
                )

                if self.batch_norm:
                    models.append(nn.BatchNorm2d(out_channels))
                
                downsample = nn.Sequential(*models)

                init_layer(downsample[0])
                init_bn(downsample[1])
            elif stride == 2:
                models.append(nn.AvgPool2d(kernel_size=2))
                models.append(nn.Conv2d(
                    self.in_channels, out_channels, kernel_size=1, stride=1, bias=False
                ))

                if self.batch_norm:
                    models.append(nn.BatchNorm2d(out_channels))
                
                downsample = nn.Sequential(*models)

                init_layer(downsample[1])
                init_bn(downsample[2])

        layers = []
        layers.append(block(self.in_channels, out_channels,
                      stride, downsample, self.batch_norm))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_channels, out_channels, batch_norm=self.batch_norm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class ResNetSpectro(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num):

        super(ResNetSpectro, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center, pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)

        self.resnet = ResNet(block=ResidualBlock, layers=[
                             2, 2, 2, 2], zero_init_residual=True)

        self.conv_block_after1 = ConvBlock(in_channels=512, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x)
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clipwise_output': clipwise_output,
                       'embedding': embedding}

        return output_dict


"""Base CNN14 model. From PANN codes.
"""
class Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num):

        super(Cnn14, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center, pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(mel_bins)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))

        output_dict = {'clipwise_output': clipwise_output,
                       'embedding': embedding}

        return output_dict


"""Define the models used in finetuning
"""
class Transfer_Cnn14(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, freeze_base):
        """Classifier for a new task using pretrained Cnn14 as a sub module.
        """
        super(Transfer_Cnn14, self).__init__()
        audioset_classes_num = 527

        pretrian_window_size = 1024
        pretrain_hop_size = 320
        pretrain_mel_bins = 64

        self.skip_spectrogram_extractor = window_size != pretrian_window_size or hop_size != pretrain_hop_size
        self.skip_logmel_extractor = window_size != pretrian_window_size or mel_bins != pretrain_mel_bins
        self.skip_bn0 = mel_bins != pretrain_mel_bins

        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin,
                          fmax, audioset_classes_num)

        # Transfer to another task layer
        self.fc_transfer = nn.Linear(2048, classes_num, bias=True)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.parameters():
                param.requires_grad = False

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_transfer)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)

        param_names = []
        for n, _ in self.named_parameters():
            param_names.append(n)
        checkpoint_param_names = checkpoint['model'].keys()

        # resuming previous training if the lengths are equal
        resume_flag = len(param_names) == len(checkpoint_param_names)

        skip_names = []
        if not resume_flag and self.skip_spectrogram_extractor:
            skip_names.extend([
                "spectrogram_extractor.stft.conv_real.weight",
                "spectrogram_extractor.stft.conv_imag.weight"
            ])
        if not resume_flag and self.skip_logmel_extractor:
            skip_names.extend([
                "logmel_extractor.melW",
            ])
        if not resume_flag and self.skip_bn0:
            skip_names.extend([
                "bn0.weight",
                "bn0.bias",
                "bn0.running_mean",
                "bn0.running_var",
                "bn0.num_batches_tracked"
            ])

        if len(skip_names) == 0:
            # no weight should be skipped
            self.base.load_state_dict(checkpoint['model'])
        else:
            # certain weights in skip_names should be skipped
            for name in skip_names:
                del checkpoint['model'][name]
            (missing_keys, unexpected_keys) = self.base.load_state_dict(
                checkpoint['model'], strict=False)
            print("Discarded keys: {}".format(missing_keys))
            print("Unexpected keys: {}".format(unexpected_keys))

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        embedding = output_dict['embedding']

        clipwise_output = F.softmax(self.fc_transfer(embedding), dim=-1)
        output_dict['clipwise_output'] = clipwise_output

        return output_dict


class Transfer_ResNet(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, freeze_base=False):
        super(Transfer_ResNet, self).__init__()

        self.base = ResNetSpectro(sample_rate, window_size, hop_size, mel_bins, fmin,
                                  fmax, classes_num)

        if freeze_base:
            # Freeze AudioSet pretrained layers
            for param in self.base.get_submodule("resnet").parameters():
                param.requires_grad = False

        self.init_weights()
        # self.init_from_resnet18(torchvision.models.resnet18(pretrained=True))

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)
        self.base.load_state_dict(checkpoint['model'])

    def init_weights(self):
        pass

    def init_from_resnet18(self, resnet18):
        print("initializing using resnet18")
        resnet_submodule = self.base.get_submodule("resnet")

        layer_params = []
        for n, v in resnet18.named_parameters():
            if "layer" in n:
                layer_params.append(v)

        for param1, param2 in zip(resnet_submodule.parameters(), layer_params):
            param1.data.copy_(param2.data)

    def forward(self, input, mixup_lambda=None):
        """Input: (batch_size, data_length)
        """
        output_dict = self.base(input, mixup_lambda)
        return output_dict


class ConvAttention(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, freeze_base=False):
        """Attention enhanced CNN"""

        super(ConvAttention, self).__init__()

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window, center=center, pad_mode=pad_mode,
                                                 freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
                                                 freeze_parameters=True)

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock_Attention(
            in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock_Attention(
            in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock_Attention(
            in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc2 = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc2)

    def load_from_pretrain(self, pretrained_checkpoint_path):
        checkpoint = torch.load(pretrained_checkpoint_path)

        param_names = []
        for n, _ in self.named_parameters():
            param_names.append(n)
        checkpoint_param_names = checkpoint['model'].keys()

        # resuming previous training if the lengths are equal
        resume_flag = len(param_names) == len(checkpoint_param_names)

        if resume_flag:
            print("Loading conv_attention_weights")
            self.load_state_dict(checkpoint['model'])
        else:
            if "base" in next(iter(checkpoint_param_names)):
                # loading from the tranfer_cnn_14
                print("Loading transfer_cnn_14 weights")
                new_checkpoint = OrderedDict()

                for k, v in checkpoint['model'].items():
                    if "base" in k:
                        new_checkpoint[k.replace("base.", "")] = v
                    elif "fc_transfer" in k:
                        new_checkpoint[k.replace("fc_transfer", "fc2")] = v

                (missing_keys, unexpected_keys) = self.load_state_dict(
                    new_checkpoint, strict=False)
            else:
                # loading from the cnn14
                print("loading cnn14 weights")
                (missing_keys, unexpected_keys) = self.load_state_dict(
                    checkpoint['model'], strict=False)
            print("Missing keys: {}".format(missing_keys))
            print("Unexpected keys: {}".format(unexpected_keys))

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        # (batch_size, 1, time_steps, freq_bins)
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)    # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc2(x))

        output_dict = {'clipwise_output': clipwise_output,
                       'embedding': embedding}

        return output_dict
