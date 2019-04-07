import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from neural.embedding_layer import Embeddings
from neural.ELMo_layer import ELMo_layer
from neural.gaussian_noise import GaussianNoise

class SimpleEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_layers=2, activation=torch.nn.LeakyReLU):
        super(SimpleEncoder, self).__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers-1):
            self.layers.append(torch.nn.Linear(input_size, input_size))
        self.layers.append(torch.nn.Linear(input_size, output_size))
        self.activation = activation()
    def forward(self, x):
        x = torch.cat(x, dim=-1)
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
class HierarchicalLSTMClassifier(nn.Module):
    def __init__(self, embeddings, vocab_size, **kwargs):
        super(HierarchicalLSTMClassifier, self).__init__()

        self.num_layers = kwargs.get('num_layers', 1)
        self.hidden_dim_utt = kwargs.get('hidden_dim_utt', 256)
        self.hidden_dim_top = kwargs.get('hidden_dim_top', 256)
        self.output_dim = kwargs.get('output_dim', 1)
        self.batch_size = kwargs.get('batch_size', 32)
        self.bidirectional = kwargs.get('bidirectional', True)
        self.separated_utterance_lstm = kwargs.get('separated_encoders', False)
        self.turns = kwargs.get('turns', 1)
        self.num_directions = 1 if not self.bidirectional else 2

        self.dropout = 0 if self.num_layers == 1 else kwargs.get('dropout')
        self.dropout_layer = nn.Dropout(p=kwargs.get('dropout', 0.))
        self.glove_dropout = nn.Dropout(p=kwargs.get('glove_dropout', 0.))
        self.glove_noise = kwargs.get('glove_noise', 0.)
        self.elmo_dropout = kwargs.get('elmo_dropout', 0.)
        self.elmo_noise = kwargs.get('elmo_noise', 0)

        self.utterance_lstm_recurrent_init = kwargs.get('utterance_lstm_recurrent_initializer', None)
        self.utterance_lstm_input_init = kwargs.get('utterance_lstm_input_initializer', None)
        self.top_lstm_recurrent_init = kwargs.get('top_lstm_recurrent_initializer', None)
        self.top_lstm_input_init = kwargs.get('top_lstm_input_initializer', None)
        self.bias_init = kwargs.get('bias_initializer', None)
        self.dense_init = kwargs.get('dense_initializer', None)
        self.top_type = kwargs.get('top_encoder_type', 'lstm')

        self.lstm_input_dim = 0
        self.using_glove = kwargs.get('use_glove', True)
        if self.using_glove:
            self.embedding = embeddings
            self.embedding_dim = embeddings.embedding.embedding_dim  # lol
            self.lstm_input_dim += self.embedding_dim
        self.using_elmo = kwargs.get('use_elmo', True)
        if self.using_elmo:
            self.embedding_dim_elmo = 1024
            self.embedding_elmo = ELMo_layer(dropout=self.elmo_dropout)
            self.lstm_input_dim += self.embedding_dim_elmo

        if self.separated_utterance_lstm:
            self.utterance_encoder = torch.nn.ModuleList()
            for i in range(0, self.turns):
                self.utterance_encoder.append(nn.LSTM(input_size=self.lstm_input_dim,
                            hidden_size=self.hidden_dim_utt,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            dropout=self.dropout,
                            batch_first=True).cuda())
        else:
            self.utterance_encoder = torch.nn.ModuleList()
            self.utterance_encoder.append(nn.LSTM(input_size=self.lstm_input_dim,
                            hidden_size=self.hidden_dim_utt,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            dropout=self.dropout,
                            batch_first=True).cuda())
        if self.top_type == "lstm":
            self.top_level_encoder = nn.LSTM(input_size=self.hidden_dim_utt*self.num_directions*self.num_layers,
                                hidden_size=self.hidden_dim_top,
                                num_layers=1,
                                bidirectional=False,
                                dropout=self.dropout,
                                batch_first=False)
        elif self.top_type == "simple":
            self.top_level_encoder = SimpleEncoder(self.hidden_dim_utt*self.num_directions*self.turns*self.num_layers, self.hidden_dim_top * self.num_layers)

        # self.attention_2 = RecurrentAttention(1024)
        # self.attention = SimpleAttention(1024)
        # self.attention = SelfAttention(self.lstm_input_dim, True, 2, 0., mode="original")
        self.hidden2out = nn.Linear(self.hidden_dim_top, self.output_dim)
        self.glove_gaussian_layer = GaussianNoise(self.elmo_noise)
        self.elmo_gaussian_layer = GaussianNoise(self.glove_noise)
        # self.init_linear()
        # self.init_lstm()
        self.softmax = torch.nn.Softmax(dim=1) #torch.nn.LogSoftmax(dim=1)
        self.activation_func = torch.nn.LeakyReLU()

    def init_lstm(self):
        for index in range(len(self.utterance_encoder)):
            for name, param in zip(self.utterance_encoder[index]._parameters, self.utterance_encoder[index]._parameters.values()):
                if name.startswith("weight_ih") and self.utterance_lstm_input_init is not None:
                    init, value = self.utterance_lstm_input_init
                    init(param, value)
                elif name.startswith("weight_hh") and self.utterance_lstm_recurrent_init is not None:
                    init, value = self.utterance_lstm_recurrent_init
                    init(param, value)
                elif name.startswith("bias_") and self.bias_init:
                    init, value = self.bias_init
                    init(param, value)
        if self.top_type == 'lstm':
            for name, param in zip(self.top_level_encoder._parameters, self.top_level_encoder._parameters.values()):
                if name.startswith("weight_ih") and self.top_lstm_input_init is not None:
                    init, value = self.top_lstm_input_init
                    init(param, value)
                elif name.startswith("weight_hh") and self.top_lstm_recurrent_init is not None:
                    init, value = self.top_lstm_recurrent_init
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
        batch_size = x[0].size()[0]

        utterance_encoders_output = list()
        for index, (turn, turn_lengths, turn_str) in enumerate(zip(x, lengths, x_str)):
            sorted_lengths, indices = torch.sort(turn_lengths, descending=True)
            sorted_batch = turn[indices]
            #del turn_lengths
            embedded = list()
            if self.using_elmo:
                embedded_elmo = self.elmo_gaussian_layer(self.embedding_elmo(turn_str)['elmo_representations'][0])
                embedded_elmo = torch.nn.functional.pad(embedded_elmo, (0, 0, 0, turn.size()[1] - embedded_elmo.size()[1]))
                embedded.append(embedded_elmo[indices])
                del embedded_elmo, turn_str, turn
            if self.using_glove:
                embedded.append(self.glove_dropout(self.embedding(sorted_batch)))

            embedded = torch.cat(embedded, dim=2)

            # attended = self.attention_2(embedded_concat)
            # attended = self.attention(embedded_concat)
            # embedded, _, _ = self.attention(embedded, sorted_lengths)
            embedded = pack_padded_sequence(embedded, sorted_lengths, batch_first=True)
            _, (ht, ct) = self.utterance_encoder[index if self.separated_utterance_lstm else 0](embedded)
            # del _, embedded, sorted_lengths, sorted_batch, ct

            utterance_output = torch.reshape(ht.transpose(1, 0), (batch_size, -1))
            unsorted_output = torch.empty_like(utterance_output)
            unsorted_output[indices] = utterance_output
            utterance_encoders_output.append(self.dropout_layer(unsorted_output))
            # del indices, utterance_output, unsorted_output

        if self.top_type == 'lstm':
            utterance_encoders_output_tensor = torch.stack(utterance_encoders_output)
            _, (ht, ct) = self.top_level_encoder(utterance_encoders_output_tensor)
            # del _, ct
            output = torch.reshape(ht.transpose(1, 0), (batch_size, -1))
            # del ht
        elif self.top_type == 'simple':
            output = self.top_level_encoder(utterance_encoders_output)
        output = self.hidden2out(output)
        output = self.activation_func(output) if self.activation_func is not None else output



        return output#self.softmax(output)
