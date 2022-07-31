from torchlibrosa.stft import STFT, ISTFT, magphase
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from math import ceil
# from scipy.io import loadmat
# import os.path as op

# def load_mat2numpy(fname=""):
#     '''
#     Args:
#         fname: pth to mat
#         type:
#     Returns: dic object
#     '''
#     if (len(fname) == 0):
#         return None
#     else:
#         return loadmat(fname)

# class PQMF(nn.Module):
#     def __init__(self, N, M, project_root):
#         super().__init__()
#         self.N = N  # nsubband
#         self.M = M  # nfilter
#         try:
#             assert (N, M) in [(8, 64), (4, 64), (2, 64)]
#         except:
#             print("Warning:", N, "subband and ", M, " filter is not supported")
#         self.pad_samples = 64
#         self.name = str(N) + "_" + str(M) + ".mat"
#         self.ana_conv_filter = nn.Conv1d(1, out_channels=N, kernel_size=M, stride=N, bias=False)
#         data = load_mat2numpy(op.join(project_root,"arnold_workspace/src/tools/pytorch/modules/filters/f_" + self.name))
#         data = data['f'].astype(np.float32) / N
#         data = np.flipud(data.T).T
#         data = np.reshape(data, (N, 1, M)).copy()
#         dict_new = self.ana_conv_filter.state_dict().copy()
#         dict_new['weight'] = torch.from_numpy(data)
#         self.ana_pad = nn.ConstantPad1d((M - N, 0), 0)
#         self.ana_conv_filter.load_state_dict(dict_new)

#         self.syn_pad = nn.ConstantPad1d((0, M // N - 1), 0)
#         self.syn_conv_filter = nn.Conv1d(N, out_channels=N, kernel_size=M // N, stride=1, bias=False)
#         gk = load_mat2numpy(op.join(project_root,"arnold_workspace/src/tools/pytorch/modules/filters/h_" + self.name))
#         gk = gk['h'].astype(np.float32)
#         gk = np.transpose(np.reshape(gk, (N, M // N, N)), (1, 0, 2)) * N
#         gk = np.transpose(gk[::-1, :, :], (2, 1, 0)).copy()
#         dict_new = self.syn_conv_filter.state_dict().copy()
#         dict_new['weight'] = torch.from_numpy(gk)
#         self.syn_conv_filter.load_state_dict(dict_new)

#         for param in self.parameters():
#             param.requires_grad = False

#     def __analysis_channel(self, inputs):
#         return self.ana_conv_filter(self.ana_pad(inputs))

#     def __systhesis_channel(self, inputs):
#         ret = self.syn_conv_filter(self.syn_pad(inputs)).permute(0, 2, 1)
#         return torch.reshape(ret, (ret.shape[0], 1, -1))

#     def analysis(self, inputs):
#         '''
#         :param inputs: [batchsize,channel,raw_wav],value:[0,1]
#         :return:
#         '''
#         inputs = F.pad(inputs,((0,self.pad_samples)))
#         ret = None
#         for i in range(inputs.size()[1]):  # channels
#             if (ret is None):
#                 ret = self.__analysis_channel(inputs[:, i:i + 1, :])
#             else:
#                 ret = torch.cat((ret,
#                                  self.__analysis_channel(inputs[:, i:i + 1, :]))
#                                 , dim=1)
#         return ret

#     def synthesis(self, data):
#         '''
#         :param data: [batchsize,self.N*K,raw_wav_sub],value:[0,1]
#         :return:
#         '''
#         ret = None
#         # data = F.pad(data,((0,self.pad_samples//self.N)))
#         for i in range(data.size()[1]):  # channels
#             if (i % self.N == 0):
#                 if (ret is None):
#                     ret = self.__systhesis_channel(data[:, i:i + self.N, :])
#                 else:
#                     new = self.__systhesis_channel(data[:, i:i + self.N, :])
#                     ret = torch.cat((ret, new), dim=1)
#         ret = ret[...,:-self.pad_samples]
#         return ret

#     def forward(self, inputs):
#         return self.ana_conv_filter(self.ana_pad(inputs))

class FrequencyDomainParser(nn.Module):
    def __init__(self,
                 window_size=2048,
                 hop_size=441,
                 center=True,
                 pad_mode='reflect',
                 window='hann',
                 freeze_parameters = True,
                 subband = None,
                 ):
        super(FrequencyDomainParser, self).__init__()
        self.subband = subband
        if(self.subband is None):
            self.stft = STFT(n_fft=window_size, hop_length=hop_size,
                win_length=window_size, window=window, center=center,
                pad_mode=pad_mode, freeze_parameters=freeze_parameters)

            self.istft = ISTFT(n_fft=window_size, hop_length=hop_size,
                win_length=window_size, window=window, center=center,
                pad_mode=pad_mode, freeze_parameters=freeze_parameters)
        else:
            self.stft = STFT(n_fft=window_size // self.subband, hop_length=hop_size // self.subband,
                win_length=window_size // self.subband, window=window, center=center,
                pad_mode=pad_mode, freeze_parameters=freeze_parameters)

            self.istft = ISTFT(n_fft=window_size // self.subband, hop_length=hop_size // self.subband,
                win_length=window_size // self.subband, window=window, center=center,
                pad_mode=pad_mode, freeze_parameters=freeze_parameters)

        # if(subband is not None and root is not None):
        #     self.qmf = PQMF(subband, 64, root)

    def complex_spectrogram(self, input, eps=0.):
        # [batchsize, samples]
        # return [batchsize, 2, t-steps, f-bins]
        real, imag = self.stft(input)
        return torch.cat([real, imag], dim=1)

    def reverse_complex_spectrogram(self, input, eps=0., length=None):
        # [batchsize, 2[real,imag], t-steps, f-bins]
        wav = self.istft(input[:, 0:1, ...], input[:, 1:2, ...], length=length)
        return wav

    def spectrogram(self, input, eps=0.):
        (real, imag) = self.stft(input.float())
        return torch.clamp(real ** 2 + imag ** 2, eps, np.inf) ** 0.5

    def spectrogram_phase(self, input, eps=0.):
        (real, imag) = self.stft(input.float())
        mag = torch.clamp(real ** 2 + imag ** 2, eps, np.inf) ** 0.5
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def wav_to_spectrogram_phase(self, input, eps=1e-8):
        """Waveform to spectrogram.

        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        cos_list = []
        sin_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            mag, cos, sin = self.spectrogram_phase(input[:,channel, :], eps=eps)
            sp_list.append(mag)
            cos_list.append(cos)
            sin_list.append(sin)

        sps = torch.cat(sp_list, dim=1)
        coss = torch.cat(cos_list, dim=1)
        sins = torch.cat(sin_list, dim=1)
        return sps, coss, sins

    def spectrogram_phase_to_wav(self, sps, coss, sins, length):
        channels_num = sps.size()[1]
        res = []
        for i in range(channels_num):
            res.append(self.istft(sps[:,i:i+1,...] * coss[:,i:i+1,...], sps[:,i:i+1,...] * sins[:,i:i+1,...], length))
            res[-1] = res[-1].unsqueeze(1)
        return torch.cat(res,dim=1)

    def wav_to_spectrogram(self, input, eps=1e-8):
        """Waveform to spectrogram.

        Args:
          input: (batch_size,channels_num, segment_samples)

        Outputs:
          output: (batch_size, channels_num, time_steps, freq_bins)
        """
        sp_list = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            sp_list.append(self.spectrogram(input[:, channel, :], eps=eps))
        output = torch.cat(sp_list, dim=1)
        return output

    def spectrogram_to_wav(self, input, spectrogram, length=None):
        """Spectrogram to waveform.
        Args:
          input: (batch_size, segment_samples, channels_num)
          spectrogram: (batch_size, channels_num, time_steps, freq_bins)

        Outputs:
          output: (batch_size, segment_samples, channels_num)
        """
        channels_num = input.shape[1]
        wav_list = []
        for channel in range(channels_num):
            (real, imag) = self.stft(input[:, channel, :])
            (_, cos, sin) = magphase(real, imag)
            wav_list.append(self.istft(spectrogram[:, channel: channel + 1, :, :] * cos,
                                       spectrogram[:, channel: channel + 1, :, :] * sin, length))
        output = torch.stack(wav_list, dim=1)
        return output

    # todo the following code is not bug free!
    def wav_to_complex_spectrogram(self, input, eps = 0.):
        # [batchsize , channels, samples]
        # [batchsize, 2[real,imag]*channels, t-steps, f-bins]
        res = []
        channels_num = input.shape[1]
        for channel in range(channels_num):
            res.append(self.complex_spectrogram(input[:, channel, :], eps=eps))
        return torch.cat(res,dim=1)

    def complex_spectrogram_to_wav(self, input, eps=0., length=None):
        # [batchsize, 2[real,imag]*channels, t-steps, f-bins]
        # return  [batchsize, channels, samples]
        channels = input.size()[1] // 2
        wavs = []
        for i in range(channels):
            wavs.append(self.reverse_complex_spectrogram(input[:,2*i:2*i+2,...],eps = eps, length = length))
            wavs[-1] = wavs[-1].unsqueeze(1)
        return torch.cat(wavs,dim=1)

    # def wav_to_complex_subband_spectrogram(self, input, eps=0.):
    #     # [batchsize, channels, samples]
    #     # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
    #     subwav = self.qmf.analysis(input) # [batchsize, subband*channels, samples]
    #     subspec = self.wav_to_complex_spectrogram(subwav)
    #     return subspec

    # def complex_subband_spectrogram_to_wav(self, input, eps=0.):
    #     # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
    #     # [batchsize, channels, samples]
    #     subwav = self.complex_spectrogram_to_wav(input)
    #     data = self.qmf.synthesis(subwav)
    #     return data

    # def wav_to_mag_phase_subband_spectrogram(self, input, eps=1e-8):
    #     """
    #     :param input:
    #     :param eps:
    #     :return:
    #         loss = torch.nn.L1Loss()
    #         model = FDomainHelper(subband=4)
    #         data = torch.randn((3,1, 44100*3))

    #         sps, coss, sins = model.wav_to_mag_phase_subband_spectrogram(data)
    #         wav = model.mag_phase_subband_spectrogram_to_wav(sps,coss,sins,44100*3//4)

    #         print(loss(data,wav))
    #         print(torch.max(torch.abs(data-wav)))

    #     """
    #     length, pad = input.size()[-1], 0
    #     while ((length + pad) % self.qmf.N != 0):  pad += 1
    #     F.pad(data, (0, pad))
    #     subwav = self.qmf.analysis(input) # [batchsize, subband*channels, samples]
    #     sps, coss, sins = self.wav_to_spectrogram_phase(subwav, eps=eps)
    #     return sps,coss,sins

    # def mag_phase_subband_spectrogram_to_wav(self, sps,coss,sins, length,eps=0.):
    #     # [batchsize, 2[real,imag]*subband*channels, t-steps, f-bins]
    #     # [batchsize, channels, samples]
    #     subwav = self.spectrogram_phase_to_wav(sps,coss,sins, ceil(length / self.qmf.N) + self.qmf.pad_samples // self.qmf.N)
    #     data = self.qmf.synthesis(subwav)
    #     return data[...,:length]

if __name__ == "__main__":
    pass
    # import matplotlib.pyplot as plt
    # loss = torch.nn.L1Loss()

    # SUB = 8
    # for length in range(1,9):
    #     model = FrequencyDomainParser(subband=SUB)
    #     data = torch.randn((3,3, 44100*length))

    #     sps,coss,sins = model.wav_to_mag_phase_subband_spectrogram(data)
    #     print(sps.size(), coss.size(), sins.size())
    #     wav = model.mag_phase_subband_spectrogram_to_wav(sps,coss,sins,44100*length)

    #     plt.plot(wav[0,0,...].numpy())
    #     plt.plot(data[0,0,...].numpy())
    #     plt.plot(data[0,0,...].numpy()-wav[0,0,...].numpy())
    #     plt.show()

    #     print(loss(data,wav))
    #     print(torch.sum(torch.abs(data-wav)))
