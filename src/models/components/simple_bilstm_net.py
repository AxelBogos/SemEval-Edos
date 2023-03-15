import torch
from torch import nn


class SimpleBiLstmNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 1)

    def forward(self, x):
        # embedded = self.embedding(x)
        embedded = x
        output, _ = self.lstm(embedded)
        pooled = torch.mean(output, dim=1)
        out = self.dropout(pooled)
        out = self.fc(out)
        return out


if __name__ == "__main__":
    _ = SimpleBiLstmNet()
