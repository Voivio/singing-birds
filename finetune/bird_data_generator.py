import numpy as np
import csv
import time
import logging
import h5py

import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '../utils'))


class BirdDataSet(object):
    def __init__(self, hdf5_path, sample_rate=32000):
        """This class takes the meta of an audio clip as input, and return 
        the waveform and target of the audio clip. This class is used by DataLoader. 
        """
        self.hdf5_path = hdf5_path
        self.sample_rate = sample_rate

    def __getitem__(self, index):
        """Load waveform and target of an audio clip.
        
        Args:
          index - int.

        Returns: 
          data_dict: {
            'audio_name': str, 
            'waveform': (clip_samples,), 
            'target': int}
        """
        hdf5_path = self.hdf5_path

        with h5py.File(hdf5_path, 'r') as hf:
            audio_name = hf['audio_name'][index]
            waveform = hf['waveform'][index]
            waveform = self.resample(waveform)
            target = hf['target'][index].astype(np.int32)

            one_hot = np.zeros((50,))
            one_hot[target] = 1

        data_dict = {
            'audio_name': audio_name, 'waveform': waveform, 'target': one_hot}

        return data_dict

    def resample(self, waveform):
        """Resample.

        Args:
          waveform: (clip_samples,)

        Returns:
          (resampled_clip_samples,)
        """
        if self.sample_rate == 32000:
            return waveform
        elif self.sample_rate == 16000:
            return waveform[0:: 2]
        elif self.sample_rate == 8000:
            return waveform[0:: 4]
        else:
            raise Exception('Incorrect sample rate!')


class Base(object):
    def __init__(self, hdf5_path, batch_size, samples_per_class, random_seed):
        """Base class of train sampler.
        
        Args:
          hdf5_path: string
          batch_size: int
          samples_per_class: int
          random_seed: int
        """
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class
        self.random_state = np.random.RandomState(random_seed)

        # Black list
        self.black_list_names = []
        logging.info('Black list samples: {}'.format(len(self.black_list_names)))

        # Load target
        load_time = time.time()

        with h5py.File(hdf5_path, 'r') as hf:
            # change :100 to : if done testing
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:5000]]
            targets = hf['target'][:5000].astype(np.float32)

        self.audios_num = targets.shape[0]
        self.classes_num = np.amax(targets).astype(int) + 1

        logging.info('Training number: {}'.format(self.audios_num))
        logging.info('Classes number: {}'.format(self.classes_num))
        logging.info('Samples per class: {}'.format(self.samples_per_class))
        logging.info('Load target time: {:.3f} s'.format(time.time() - load_time))


class BirdSampler(Base):
    def __init__(self, hdf5_path, indices_range, batch_size, samples_per_class, random_seed=1234):
        """Balanced sampler. Generate batch meta for training.
        
        Args:
          hdf5_path: string
          indices_range: (min, max). Specifies the range of indices to be used.
            This is for splitting the train/eval dataset. min/max in (0, 1).
          batch_size: int
          samples_per_class: int
          random_seed: int
        """
        super(BirdSampler, self).__init__(hdf5_path, batch_size, samples_per_class, random_seed)

        start = np.floor(indices_range[0] * self.samples_per_class).astype(int)
        end = np.floor(indices_range[1] * self.samples_per_class).astype(int)
        self.indexes = np.tile(np.arange(start, end), (self.classes_num, 1)) \
                       + np.arange(self.classes_num).reshape(-1, 1) * self.samples_per_class
        self.indexes = self.indexes.astype(np.int32).ravel()

        # Shuffle indexes
        self.random_state.shuffle(self.indexes)
        self.pointer = 0

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'hdf5_path': string, 'index_in_hdf5': int}, 
            ...]
        """
        batch_size = self.batch_size

        while True:
            batch_meta = []
            i = 0
            while i < batch_size:
                index = self.indexes[self.pointer]
                
                self.pointer += 1

                # Shuffle indexes and reset pointer
                if self.pointer >= len(self.indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.indexes)

                # If audio in black list then continue
                if self.audio_names[index] in self.black_list_names:
                    continue
                else:
                    batch_meta.append(index)
                    i += 1

            yield batch_meta

    def state_dict(self):
        state = {
            'indexes': self.indexes,
            'pointer': self.pointer}
        return state

    def load_state_dict(self, state):
        self.indexes = state['indexes']
        self.pointer = state['pointer']


class BirdEvaluateSampler(object):
    def __init__(self, hdf5_path, indices_range, batch_size, samples_per_class):
        """Evaluate sampler. Generate batch meta for evaluation.
        
        Args:
          indexes_hdf5_path: string
          batch_size: int
        """
        self.batch_size = batch_size

        with h5py.File(hdf5_path, 'r') as hf:
            # change :100 to : if done testing
            targets = hf['target'][:5000].astype(np.float32)
        self.audios_num = len(targets)
        self.classes_num = np.amax(targets).astype(int) + 1

        start = np.floor(indices_range[0] * samples_per_class).astype(int)
        end = np.floor(indices_range[1] * samples_per_class).astype(int)

        self.indexes = np.tile(np.arange(start, end), (self.classes_num, 1)) \
                       + np.arange(self.classes_num).reshape(-1, 1) * samples_per_class
        self.indexes = self.indexes.astype(np.int32).ravel() 

    def __iter__(self):
        """Generate batch meta for training. 
        
        Returns:
          batch_meta: e.g.: [
            {'hdf5_path': string, 
             'index_in_hdf5': int}
            ...]
        """
        batch_size = self.batch_size
        pointer = 0

        while pointer < len(self.indexes):
            batch_meta = self.indexes[pointer:pointer+batch_size].tolist()
            pointer += batch_size
            yield batch_meta


def bird_collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), 'target': int}, 
                             {'audio_name': str, 'waveform': (clip_samples,), 'target': int},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), 'target': (batch_size,)}
    """
    np_data_dict = {}

    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])

    return np_data_dict
