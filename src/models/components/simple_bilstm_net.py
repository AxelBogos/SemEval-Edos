import torch
from torch import nn


class SimpleBiLstmNet(nn.Module):
    def __init__(self, args):
        super().__init__()

        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=args.embedding_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.num_layers,
            bidirectional=args.bidirectional,
            batch_first=True,
            dropout=args.dropout,
        )
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc = nn.Linear(args.hidden_dim * 2 if args.bidirectional else args.hidden_dim, 1)

    def forward(self, x):
        # embedded = self.embedding(x)
        embedded = x
        output, _ = self.lstm(embedded)
        pooled = torch.mean(output, dim=1)
        out = self.dropout(pooled)
        out = self.fc(out)
        return out
