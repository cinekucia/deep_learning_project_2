from torch import nn, Tensor


class FCEmbedding(nn.Module):
    """
    Embedding by two fully connected layers.
    """
    def __init__(self, input_size: int, embedding_size: int, hidden_layer_size: int):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.linear1 = nn.Linear(input_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        x = x.view(-1, self.input_size)
        x = self.relu(self.linear1(x))
        return self.linear2(x).view(batch_size, -1, self.embedding_size)
