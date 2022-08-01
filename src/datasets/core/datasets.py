import random

import json
import torch
import torchaudio
from torch.utils.data import Dataset

from typing import Tuple

class EnhHFLibriDataset(Dataset):
    def __init__(
        self, manifest_path: str, sample_rate: int = 44100, 
        segment_length_sec: int = 10, random_seed: int = 42
        ):

        super(EnhHFLibriDataset, self).__init__()

        # load dataset
        with open(manifest_path, mode='r', encoding='utf-8') as fr:
            lines = fr.readlines()
        self.data = [json.loads(line.strip('\r\n')) for line in lines]

        # set random seed
        random.seed(random_seed)

        # initialise required variables
        self.segment_length = int(sample_rate * segment_length_sec)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx.tolist()

        item = self.data[idx]

        ### PROCESS INPUTS ###         
        input_signal, input_sr = torchaudio.load(item['input_filepath'], normalize=True)
        target_signal, target_sr = torchaudio.load(item['target_filepath'], normalize=True)

        input_signal_length = input_signal.size(1)
        target_signal_length = target_signal.size(1)
        assert input_signal_length == target_signal_length, 'Mismatch of input and target signal length'
        assert input_sr == target_sr, 'Mismatch of input and target sample rates'

        # pad audio if insufficient length
        if input_signal_length < self.segment_length:
            input_signal = torch.nn.functional.pad(input_signal, (0, self.segment_length - input_signal.size(1)), 'constant')
            target_signal = torch.nn.functional.pad(target_signal, (0, self.segment_length - target_signal.size(1)), 'constant')

        # truncate audio if more than req. length
        elif input_signal_length > self.segment_length:
            input_signal = input_signal[:, :self.segment_length]
            target_signal = target_signal[:, :self.segment_length]

        return input_signal, target_signal

    def __len__(self) -> int:
        return len(self.data)
