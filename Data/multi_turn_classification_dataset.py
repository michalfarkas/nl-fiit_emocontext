from Data.basic_dataset import BasicDataset
from vocabulary import Vocabulary
import torch
import random
import numpy

class MultiTurnClassificationDataset(BasicDataset):
    def __init__(self, x, y, separator="\t", vocab=None, max_length=None, preprocessing=None, device='cuda', split_turns=True):
        """
        Args:
            x (): pandas.Series, turns are separated by separator param
            y (): list of training labels
            max_length (int): the max length for each sample. if 0 the maximum length in the dataset is used
            vocab (Vocabulary): vocabulary class instance
        """
        super(BasicDataset, self).__init__()

        self.device = device
        self.labels = y
        self.split_turns = split_turns
        self.ids =  list(x.index)

        self.data = list()
        self.vocab = vocab if vocab is not None else Vocabulary()
        self.preprocessing = self.tokenize if preprocessing is None else preprocessing

        if split_turns:
            x = x.str.split(separator)
            for i in range(len(x.iloc[0])):
                processed_turn = self.process_data(x.str[i])
                self.data.append(processed_turn)
                if vocab is None:
                        for sample in processed_turn:
                            self.vocab.process_tokens(sample)
        else:
            x = x.str.replace("\t", " ; ")
            processed_turn = self.process_data(x)
            self.data.append(processed_turn)
            if vocab is None:
                    for sample in processed_turn:
                        self.vocab.process_tokens(sample)

        self.max_length = max_length if max_length is not None else max([max([len(sample) for sample in turn]) for turn in self.data])

        data_tensor_list = list()
        label_tensor_list = list()
        lengths_tensor_list = list()
        for turn in self.data:
            data_tensor_list.append(torch.stack([torch.Tensor(self.vectorize(sample, self.vocab.word2idx, self.max_length)).long() for sample in turn], 1).t())
            lengths_tensor_list.append(torch.tensor([min(len(sample), self.max_length) for sample in turn], dtype=torch.int16))
        self.data_tensor = torch.stack(data_tensor_list)
        self.labels_tensor = torch.tensor(self.labels, dtype=torch.int64) if self.labels is not None else None
        self.lengths_tensor = torch.stack(lengths_tensor_list)
        self.ids_tensor = torch.tensor(self.ids, dtype=torch.int64)

        self.shuffle_list = numpy.arange(len(self.ids))


    def __getitem__(self, index):
        # Tuple: (ids, vectorized, lengths, raw_strings, labels)
        # ids: tensor of shape (batch_size,)
        # vectorized: list of turns, each turn is tensor with shape(batch_size, max_seq_len)
        # labels: tensor of shape (batch_size,)
        # lengths: list of turns, each turn is tensor of shape (batch_size,)
        # raw_strings: list of turns, each turn is list of samples, each sample is a list of strings(words); just how elmo lieks it

        ids = self.ids_tensor[self.shuffle_list[index]].to(self.device)
        vectorized = list(torch.chunk(self.data_tensor[:, self.shuffle_list[index]].to(self.device), chunks=self.data_tensor.size()[0]))
        lengths = list(torch.chunk(self.lengths_tensor[:, self.shuffle_list[index]].to(self.device), chunks=self.lengths_tensor.size()[0]))
        for i in range(len(vectorized)):
            # we get rid of the first dimension
            vectorized[i] = vectorized[i].squeeze(0)
            lengths[i] = lengths[i].squeeze(0)
        labels = self.labels_tensor[self.shuffle_list[index]].to(self.device) if self.labels is not None else None
        raw_strings = list()
        for turn in self.data:
            raw_strings.append([turn[i] for i in self.shuffle_list[index]])

        return ids, vectorized, lengths, raw_strings, labels

    def reset(self, state=None):
        if state is not None:
            random.setstate(state)
        random.shuffle(self.shuffle_list)

    def __len__(self):
        return len(self.data[0])
