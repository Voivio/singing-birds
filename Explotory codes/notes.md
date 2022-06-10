# The loading procedure of dataset

Refer to [DATASETS & DATALOADERS](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) and [TORCH.UTILS.DATA](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler) for how to use `DataSet` and `DataLoader` classes. Simply put, `DataSet` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around the `Dataset` to enable easy access to the samples.

In file `/utils/dataset.py` defines a function `pack_waveforms_to_hdf5` which converts the audio data into a `hdf5` filw with fields `audio_name`, `waveform` (loaded audio as a array), `target` (label), `sample_rate`.

Then, in file `/utils/data_generator.py`, the `AudioSetDataset` class is defined as a map-style dataset class that implements the `__getitem__()` and `__len__() `protocols. All other classes endswith `Sampler` are samplers used by `DataLoader` from PyTorch to yield the indices to access from dataset. Finally, the `collate_fn` is also used to be passed to `DataLoader` to collate the single samples into a mini-batch of data. The whole procedure is similar to:

```Python
for indices in batch_sampler:
    yield collate_fn([dataset[i] for i in indices])
```

To use the existing codes, we should implement our own version of `DataSet`, sampler, `collate_fn` and add them into `/pytorch/main.py`. It would be better to make a duplicate version.