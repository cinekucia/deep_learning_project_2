from torch import nn, Tensor

from settings import NUM_CLASSES


class LSTMModel(nn.Module):
    def __init__(
        self,
        num_mel_bins: int,
        hidden_layer_size: int,
        layers_num: int,
        bidirectional: bool,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            num_mel_bins, hidden_layer_size, layers_num, bidirectional=bidirectional, batch_first=True, dropout=dropout
        )
        self.linear = nn.Linear(hidden_layer_size * (2 if bidirectional else 1), NUM_CLASSES)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        output, (hidden, _) = self.lstm(src)
        output = output.mean(dim=1)
        output = self.linear(output)
        return output


class RNNModel(nn.Module):
    def __init__(
        self,
        num_mel_bins: int,
        hidden_layer_size: int,
        layers_num: int,
        bidirectional: bool,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn = nn.RNN(
            num_mel_bins, hidden_layer_size, layers_num, bidirectional=bidirectional, batch_first=True, dropout=dropout
        )
        self.linear = nn.Linear(hidden_layer_size * (2 if bidirectional else 1), NUM_CLASSES)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        output, _ = self.rnn(src)
        output = output.mean(dim=1)
        output = self.linear(output)
        return output
