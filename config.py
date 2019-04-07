from Models import lstm_classifier, hierarchical_lstm_classifier
import torch

#train_file = "./Data/train.txt"
train_file = "/home/mehre/[DEV]/nl-fiit_emocontext/Data/train.txt"
validation_file = "/home/mehre/[DEV]/nl-fiit_emocontext/Data/dev.txt"
test_file_wo_labels = "/home/mehre/[DEV]/nl-fiit_emocontext/Data/testwithoutlabels.txt"
test_file_w_labels = "/home/mehre/[DEV]/nl-fiit_emocontext/Data/test.txt"
#submission_file = None
submission_file = "/home/mehre/[DEV]/nl-fiit_emocontext/Data/dev-submission.txt"
#embedding_file = "./Data/glove.6B.300d.txt"
embedding_file = "/home/mehre/[DEV]/nl-fiit_emocontext/Data/glove.6B.300d.txt"
preprocessor = "Ekphrasis"
#preprocessor = "NL-FIIT"

batch_size = 32
max_epochs = 40
optimizer = 'adam'
# model = lstm_classifier.LSTMClassifier
model = hierarchical_lstm_classifier.HierarchicalLSTMClassifier
hyper_parameter_search_times = 4
ensemble = 5
# ensemble = None
# ensemble = 3
ensemble_type = 'sum'
ensemble_max_weight = 5
ensemble_rebalance = True

early_stop_monitor = 'val_microF1'
early_stop_min_delta = 0
early_stop_patience = 3
early_stop_mode = 'max'

restore_monitor = 'val_microF1'
restore_mode = 'max'

max_grad_norm = 0.25

classes = 4
split_turns = True
turns = 0 if not split_turns else slice(0, None, 1)
# turns = 2  # only last
include_strings = True
train_val_test_split = None  # [0.9, 0.1]
test = True
# label_weights = [1.0, 1.0, 1.0, 1.0]  # none
label_weights = [0.25, 0.25, 0.25, 1.6923]  # train -> test_w_others
# label_weights = [1.5625, 1.5625, 1.5625, 0.5]  # train -> balanced
# label_weights = [1.5625, 1.5625, 1.5625, 0.3]  # train -> balanced-ish
# label_weights = [6.25, 6.25, 6.25, 0.2841]  # test -> balanced
# label_weights = [1.33, 1.33, 1.33, 0.]  # train -> w/o others

labels_to_index = {
    "happy": 0,
    "sad": 1,
    "angry": 2,
    "others": 3,
}

index_to_labels = {
    0: "happy",
    1: "sad",
    2: "angry",
    3: "others"
}

lstm_classifier_params = {
    'output_dim': 4,
    'hidden_dim': 2048,
    'use_glove': True,
    'use_elmo': False,
    'lstm_recurrent_initializer': (torch.nn.init.orthogonal_, torch.nn.init.calculate_gain('tanh')),
    'lstm_input_initializer': (torch.nn.init.xavier_uniform_, torch.nn.init.calculate_gain('tanh')),
    'bias_initializer': (torch.nn.init.constant_, 0),
    'dense_initializer': (torch.nn.init.xavier_uniform_, 1),
    'glove_noise': 0.,
    'glove_dropout': 0.,
    'elmo_noise': 0.5,
    'elmo_dropout': 0.5
}

hierarchical_lstm_classifier_params = {
    'output_dim': 4,
    'hidden_dim_utt': 3072,
    'hidden_dim_top': 1024,
    'turns': 3,
    'separated_encoders': False,
    'use_glove': False,
    'use_elmo': True,
    'num_layers': 1,
    'dropout': 0.5,
    'top_encoder_type': 'lstm',
    'utterance_lstm_recurrent_initializer': (torch.nn.init.orthogonal_, torch.nn.init.calculate_gain('tanh')),
    'utterance_lstm_input_initializer': (torch.nn.init.xavier_uniform_, torch.nn.init.calculate_gain('tanh')),
    'top_lstm_recurrent_initializer': (torch.nn.init.orthogonal_, torch.nn.init.calculate_gain('tanh')),
    'top_lstm_input_initializer': (torch.nn.init.xavier_uniform_, torch.nn.init.calculate_gain('tanh')),
    'bias_initializer': (torch.nn.init.constant_, 0),
    'dense_initializer': (torch.nn.init.xavier_uniform_, 1),
    'glove_noise': 0.,
    'glove_dropout': 0.,
    'elmo_noise': 3.,
    'elmo_dropout': 0.6
}
