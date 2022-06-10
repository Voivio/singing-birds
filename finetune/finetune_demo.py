
import torch
import numpy as np
import argparse
import librosa
import pandas as pd

from finetune_models import Transfer_Cnn14

def demo(args):

    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    use_cuda = args.cuda
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # defualt configurations 
    sample_rate = 32000
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 2000
    fmax = 10000
    classes_num = 50

    # initialize model
    model = Transfer_Cnn14(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num, freeze_base=True)
    
    # load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if use_cuda:
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')

    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = torch.Tensor(waveform).to(device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)
    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    # size : (classes_num,)


    # Print audio tagging top probabilities
    labels = np.unique(pd.read_csv("./resources/train_new.csv").loc[:4999, "ebird_code"])
    sorted_indexes = np.argsort(clipwise_output)[::-1]
    for k in range(5):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run demo of ')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()
    demo(args)