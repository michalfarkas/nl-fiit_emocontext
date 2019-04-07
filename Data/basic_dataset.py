from torch.utils.data.dataset import Dataset
from vocabulary import Vocabulary

from tqdm import tqdm
import numpy as np

class BasicDataset(Dataset):
    def __init__(self):
        super(BasicDataset, self).__init__()
        self.preprocessing = self.tokenize

    def vectorize(self, text, word2idx, max_length=None):
        """
        Coverting array of tokens to array of ids, with a fixed max length and zero padding
        Args:
            text (): list of words
            word2idx (): dictionary of word to ids
            max_length (): the maximum length of the input sequence
        Returns: zero-padded list of ids
        """

        if max_length is None:
            words = np.zeros(len(text), dtype=int)
        else:
            words = np.zeros(max_length, dtype=int)
            text = text[:max_length]

        for i, token in enumerate(text):
            words[i] = word2idx[token] if token in word2idx else word2idx["<UNK>"]

        return words

    def tokenize(self, text):
        return text.lower().split()

    def process_data(self, data):
        return [self.preprocessing(x) for x in tqdm(data)]


    def process_data_with_vocabulary(self, data):
        _data = []
        vocab = Vocabulary()

        for x in tqdm(data):
            tokens = self.preprocessing(x)
            vocab.process_tokens(tokens)
            _data.append(tokens)
        return _data, vocab