import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embeddings=None, trainable=False, dropout=.0):
        super(Embeddings, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

        if embeddings is not None:
            self.init_embeddings(embeddings, trainable)

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights), requires_grad=trainable)

    def forward(self, x):
        embeddings = self.embedding(x)

        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)

        return embeddings