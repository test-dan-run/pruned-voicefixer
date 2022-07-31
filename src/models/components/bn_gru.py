import torch.nn as nn
from torch import Tensor

class BN_GRU(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int = 1, 
        bidirectional: bool = False, batchnorm: bool = True, 
        dropout: float = 0.0) -> None:

        super(BN_GRU, self).__init__()
        self.bn = nn.BatchNorm2d(1) if batchnorm else None

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True)
                
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, inputs: Tensor) -> Tensor:
        # (batch, 1, seq, feature)
        if self.bn is not None:
            inputs = self.bn(inputs)
        out, _ = self.gru(inputs.squeeze(1))
        return out.unsqueeze(1)