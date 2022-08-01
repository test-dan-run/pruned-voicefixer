import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchaudio.transforms import MelSpectrogram

import pytorch_lightning as pl
from omegaconf import DictConfig

from .models.modules import LogMelTransform
from .models.unet import UNetResComplex_100Mb

class Generator(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-8):
        super(Generator, self).__init__()
        self.eps = eps
        self.logmel_transform = LogMelTransform()
        self.analysis_module = UNetResComplex_100Mb(channels = num_channels)

    def forward(self, mel: torch.Tensor):
        '''
        send logmel spectrogram into analysis module
        '''
        logmel = self.logmel_transform(mel)
        out = self.analysis_module(logmel)
        return out + logmel

class LightningVoiceFixer(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig, optim_cfg: DictConfig, run_cfg: DictConfig, *args, **kwargs):
        super(LightningVoiceFixer, self).__init__()

        self.generator = Generator(model_cfg.num_channels)
        self.melspec_transform = MelSpectrogram(
            sample_rate = model_cfg.mel.sample_rate,
            n_fft = model_cfg.mel.n_fft,
            win_length = model_cfg.mel.win_length,
            hop_length = model_cfg.mel.hop_length,
            n_mels = model_cfg.mel.n_mels,
            center = model_cfg.mel.center,
            pad_mode = model_cfg.mel.pad_mode,
            window = model_cfg.mel.window
        )
        self.logmel_transform = LogMelTransform()

        self.lr = optim_cfg.lr
        self.betas = optim_cfg.betas
        self.lr_lambda = lambda step: self.get_lr_lambda(step,
                                                         gamma = optim_cfg.lr_decay,
                                                         warmup_steps=optim_cfg.warmup_steps,
                                                         reduce_lr_every_n_steps=optim_cfg.reduce_lr_every_n_steps)
        self.l1loss = torch.nn.L1Loss()


    def get_lr_lambda(self, step, gamma, warmup_steps, reduce_lr_every_n_steps):
        r"""Get lr_lambda for LambdaLR. E.g.,

        .. code-block: python
            lr_lambda = lambda step: get_lr_lambda(step, warm_up_steps=1000, reduce_lr_steps=10000)

            from torch.optim.lr_scheduler import LambdaLR
            LambdaLR(optimizer, lr_lambda)
        """
        if step <= warmup_steps:
            return step / warmup_steps
        else:
            return gamma ** (step // reduce_lr_every_n_steps)

    def init_weights(self, module: nn.Module):
        for m in module.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def pre(self, wav: torch.Tensor) -> torch.Tensor:
        """Waveform to spectrogram.
        Args:
          input: (batch_size, channels, segment_samples)
        Outputs:
          output: (batch_size, channels, time_steps, freq_bins)
        """
        # (batch_size, channels, mel_bins, time_steps)
        mel = self.melspec_transform(wav)

        return mel.permute(0,1,3,2)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.generator(mel)

    def training_step(self, batch, batch_idx):

        input_signals, target_signals = batch
        input_mels = self.pre(input_signals)
        target_mels = self.pre(target_signals)

        generated_mels = self(input_mels)
        train_loss = self.l1loss(generated_mels, self.logmel_transform(target_mels))

        self.log("train_loss", train_loss, on_step=True, on_epoch=False, sync_dist=True)
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):

        input_signals, target_signals = batch
        input_mels = self.pre(input_signals)
        target_mels = self.pre(target_signals)

        generated_mels = self(input_mels)
        val_loss = self.l1loss(generated_mels, self.logmel_transform(target_mels))
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, sync_dist=True)

        return {"loss": val_loss}

    def configure_optimizers(self):
        optimizer_g = Adam(self.generator.parameters(),
                                       lr=self.lr, amsgrad=True, betas=(self.betas[0], self.betas[1]))

        scheduler_g = {
            'scheduler': LambdaLR(optimizer_g, self.lr_lambda),
            'interval': 'step',
            'frequency': 1
        }
        return ([optimizer_g], [scheduler_g])