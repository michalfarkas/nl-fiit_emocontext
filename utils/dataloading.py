import pandas as pd

from datasets.classification_dataset import ClassificationDataset
from torch.utils.data import DataLoader


class ClassificationData(object):
    def __init__(self, preprocessing=None, vocab=None, max_length=None, batch_size=32, **kwargs):

        self.preprocessing = preprocessing
        self.max_length = max_length
        self.vocab = vocab
        self.batch_size = batch_size

        self.vectorization = kwargs.get('vectorization', True)

        self.train_file = kwargs.get('train_file')
        self.test_file = kwargs.get('test_file')
        self.validation_file = kwargs.get('validation_file', None)

        self.split_ratio = kwargs.get('split_ratio', 0.8)
        self.x_column = kwargs.get('x_column', 0)
        self.y_column = kwargs.get('y_column', 1)

        self.train_data = self.valid_data = self.test_data = None
        self.train_set = self.valid_set = self.test_set = None
        self.train_loader = self.valid_loader = self.test_loader = None

    def load_data(self, sep=',', header=None):

        train = pd.read_csv(self.train_file, sep=sep, header=0, quoting=1).sample(frac=1).values
        test = pd.read_csv(self.test_file, sep=sep, header=header, quoting=1).values

        if self.validation_file:
            valid = pd.read_csv(self.valid_file, sep=sep, header=0, quoting=1).sample(frac=1).values
        else:
            split_part = int(len(train) * self.split_ratio)
            valid = train[split_part:]
            train = train[:split_part]

        self.train_data, self.valid_data, self.test_data = train, valid, test

    def create_datasets(self):
        train_set = ClassificationDataset(self.train_data[:, self.x_column], self.train_data[:, self.y_column], self.vocab,
                                          max_length=self.max_length, preprocessing=self.preprocessing, vectorization=self.vectorization)
        valid_set = ClassificationDataset(self.valid_data[:, self.x_column], self.valid_data[:, self.y_column], self.vocab,
                                          max_length=self.max_length, preprocessing=self.preprocessing, vectorization=self.vectorization)
        test_set = ClassificationDataset(self.test_data[:, self.x_column], self.test_data[:, self.y_column], self.vocab,
                                         max_length=self.max_length, preprocessing=self.preprocessing, vectorization=self.vectorization)

        self.train_set, self.valid_set, self.test_set = train_set, valid_set, test_set

    def create_loaders(self):
        train_loader = DataLoader(self.train_set, self.batch_size, shuffle=True)
        valid_loader = DataLoader(self.valid_set, self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_set, self.batch_size)

        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader

    def pipeline(self):
        self.load_data()
        self.create_datasets()
        self.create_loaders()
