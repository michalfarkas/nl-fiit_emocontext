import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from neural.embedding_layer import Embeddings
from neural.ELMo_layer import ELMo_layer
from neural.gaussian_noise import GaussianNoise

class LSTMClassifier(nn.Module):
    def __init__(self, embeddings, vocab_size, **kwargs):
        super(LSTMClassifier, self).__init__()

        self.num_layers = kwargs.get('num_layers', 1)
        self.hidden_dim = kwargs.get('hidden_dim', 256)
        self.output_dim = kwargs.get('output_dim', 1)
        self.batch_size = kwargs.get('batch_size', 32)
        self.bidirectional = kwargs.get('bidirectional', True)
        self.num_directions = 1 if not self.bidirectional else 2

        self.dropout = 0 if self.num_layers == 1 else kwargs.get('dropout')
        self.glove_dropout = nn.Dropout(p=kwargs.get('glove_dropout', 0.))
        self.glove_noise = kwargs.get('glove_noise', 0.)
        self.elmo_dropout = nn.Dropout(p=kwargs.get('elmo_dropout', 0.))
        self.elmo_noise = kwargs.get('elmo_noise', 0)

        self.lstm_recurrent_init = kwargs.get('lstm_recurrent_initializer', None)
        self.lstm_input_init = kwargs.get('lstm_input_initializer', None)
        self.bias_init = kwargs.get('bias_initializer', None)
        self.dense_init = kwargs.get('dense_initializer', None)

        self.lstm_input_dim = 0
        self.using_glove = kwargs.get('use_glove', True)
        if self.using_glove:
            self.embedding = embeddings
            self.embedding_dim = embeddings.embedding.embedding_dim
            self.lstm_input_dim += self.embedding_dim
        self.using_elmo = kwargs.get('use_elmo', True)
        if self.using_elmo:
            self.embedding_dim_elmo = 1024
            self.embedding_elmo = ELMo_layer()
            self.lstm_input_dim += self.embedding_dim_elmo

        self.encoder = nn.LSTM(input_size=self.lstm_input_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.num_layers,
                               bidirectional=self.bidirectional,
                               dropout=self.dropout,
                               batch_first=True)

        self.hidden2out = nn.Linear(self.hidden_dim * self.num_directions, self.output_dim)
        self.glove_gaussian_layer = GaussianNoise(self.elmo_noise)
        self.elmo_gaussian_layer = GaussianNoise(self.glove_noise)
        self.init_lstm()
        self.init_linear()
        #self.softmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)

    def init_lstm(self):
        for name, param in zip(self.encoder._parameters, self.encoder._parameters.values()):
            if name.startswith("weight_ih") and self.lstm_input_init is not None:
                init, value = self.lstm_input_init
                init(param, value)
            elif name.startswith("weight_hh") and self.lstm_recurrent_init is not None:
                init, value = self.lstm_recurrent_init
                init(param, value)
            elif name.startswith("bias_") and self.bias_init:
                init, value = self.bias_init
                init(param, value)
    def init_linear(self):
        if self.bias_init is not None:
            init, value = self.bias_init
            init(self.hidden2out.bias, value)
        if self.dense_init is not None:
            init, value = self.bias_init
            init(self.hidden2out.weight, value)

    def forward(self, x):
        _, x, lengths, x_str = x
        if type(x) is list:
            x, lengths, x_str = (x[2], lengths[2], x_str[2])
        batch_size = x.size()[0]

        sorted_lengths, indices = torch.sort(lengths, descending=True)
        sorted_batch = x[indices]

        embedded = list()
        if self.using_elmo:
            embedded_elmo = self.elmo_dropout(self.elmo_gaussian_layer(self.embedding_elmo(x_str)['elmo_representations'][0]))
            embedded_elmo_padded = torch.nn.functional.pad(embedded_elmo, (0, 0, 0, x.size()[1] - embedded_elmo.size()[1]))
            embedded.append(embedded_elmo_padded[indices])
        if self.using_glove:
            embedded.append(self.glove_dropout(self.glove_gaussian_layer(self.embedding(sorted_batch))))

        embedded_concat = torch.cat(embedded, dim=2)

        packed_embedded = pack_padded_sequence(embedded_concat, sorted_lengths, batch_first=True)
        lstm_output, (ht, ct) = self.encoder(packed_embedded)
        padded_output, _ = pad_packed_sequence(lstm_output, batch_first=True)

        # output = padded_output[:,-1]
        output = ht[-1] if not self.bidirectional else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

        output = self.hidden2out(output)

        unsorted_output = torch.empty_like(output)
        unsorted_output[indices] = output


        return unsorted_output
