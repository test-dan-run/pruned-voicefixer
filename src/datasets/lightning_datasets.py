import os
import json
from typing import Optional
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .core.datasets import EnhHFLibriDataset

class LightningEnhHFLibriDataset(pl.LightningDataModule):
    def __init__(self, dataset_cfg: DictConfig, batch_size: int = 32):

        super(LightningEnhHFLibriDataset, self).__init__()
        self.cfg = dataset_cfg
        self.batch_size = batch_size

    @staticmethod
    def update_manifest_audiopaths(path: str) -> str:

        manifest_dir = os.path.dirname(path)
        with open(path, mode='r', encoding='utf-8') as fr:
            lines = fr.readlines()
        items = [json.loads(line.strip('\r\n')) for line in lines]
        # append manifest_dir to audiopath
        for item in items:
            item['input_filepath'] = os.path.join(manifest_dir, item['input_filepath'])
            item['target_filepath'] = os.path.join(manifest_dir, item['target_filepath'])
        
        output_filename = f'updated_{os.path.basename(path)}'
        output_path = os.path.join(manifest_dir, output_filename)
        with open(output_path, mode='w', encoding='utf-8') as fw:
            for item in items:
                fw.write(json.dumps(item)+'\n')

        return output_path

    def prepare_data(self):
        
        # change audio filepaths from relative to absolute paths
        self.cfg.train_manifest_path = LightningEnhHFLibriDataset.update_manifest_audiopaths(self.cfg.train_manifest_path)
        self.cfg.valid_manifest_path = LightningEnhHFLibriDataset.update_manifest_audiopaths(self.cfg.valid_manifest_path)
        if self.cfg.test_manifest_path:
            self.cfg.test_manifest_path = LightningEnhHFLibriDataset.update_manifest_audiopaths(self.cfg.test_manifest_path)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self.train_data = EnhHFLibriDataset(
                manifest_path = self.cfg.train_manifest_path, 
                sample_rate = self.cfg.sample_rate, 
                segment_length_sec = self.cfg.segment_length_sec, 
                random_seed = self.cfg.random_seed,
            )

            self.valid_data = EnhHFLibriDataset(
                manifest_path = self.cfg.valid_manifest_path, 
                sample_rate = self.cfg.sample_rate, 
                segment_length_sec = self.cfg.segment_length_sec, 
                random_seed = self.cfg.random_seed,
            )

        if stage == 'test':
            self.test_data = EnhHFLibriDataset(
                manifest_path = self.cfg.test_manifest_path, 
                sample_rate = self.cfg.sample_rate, 
                segment_length_sec = self.cfg.segment_length_sec, 
                random_seed = self.cfg.random_seed,
            )

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.valid_data, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, batch_size=self.batch_size)
