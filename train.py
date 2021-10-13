import os
import torch
import matplotlib.pyplot as plt
import torchaudio
from torch.utils.data import Dataset, DataLoader


class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, index):
        sample_path = self.root_dir + os.listdir(self.root_dir)[index]
        waveform, sample_rate = torchaudio.load(sample_path)
        return waveform, sample_rate


if __name__ == "__main__":
    num_threads = 2
    device = 'cpu'
    dataroot = "datasets/piano/train/"

    audio_dataset = AudioDataset(dataroot)

    dataloader = DataLoader(audio_dataset, batch_size=1, shuffle=True, num_workers=num_threads)

    waveform, sample_rate = audio_dataset[0]
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
        axes[c].set_ylabel(f'Channel {c + 1}')

    axes[c].set_xlim((-.1, 1.2))
    figure.suptitle("Waveform")
    plt.show()
