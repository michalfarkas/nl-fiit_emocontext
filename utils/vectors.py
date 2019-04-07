import numpy as np

from vocabulary import Vocabulary
from neural.embedding_layer import Embeddings

class Vectors(object):
    """
    Loading different types of pre-trained word embeddings
    Args:
        name (): type of embeddings to load
        file_name (): file name of pre-trained embeddings
        embedding_dim (): dimmension of embeddings
    """

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'glove')
        self.file_name = kwargs.get('file_name')
        self.embedding_dim = kwargs.get('embedding_dim', 300)

        self.vocab, self.length, self.embeddings = getattr(self, self.name)()

    def glove(self):
        embeddings = []
        vocab = Vocabulary()

        with open(self.file_name, encoding='UTF-8') as f:

            for line in f:
                values = line.split()
                vocab.add_word(values[0])
                embeddings.append(np.asarray(values[1:], dtype='float32'))

            if "<UNK>" not in vocab.word2idx:
                vocab.add_word("<UNK>")
                embeddings.append(np.random.uniform(low=-0.05, high=0.05, size=self.embedding_dim))

            return vocab, len(embeddings), Embeddings(vocab_size=len(embeddings),
                                     embedding_dim=self.embedding_dim,
                                     embeddings=np.array(embeddings, dtype='float32'),
                                     trainable=False)

    def elmo(self):
        raise NotImplementedError

    def word2vec(self):
        raise NotImplementedError
