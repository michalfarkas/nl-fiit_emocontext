import torch.nn as nn
from allennlp.modules.elmo import Elmo, batch_to_ids

class ELMo_layer(nn.Module):
    """
    Proxy for ELMo embeddings.
    https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
    """
    def __init__(self, dropout=0, num_layers=2, options=None, weights=None, other_params={}):
        super(ELMo_layer, self).__init__()

        options_file = options if options is not None else "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = weights if weights is not None else"https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        self.elmo = Elmo(options_file, weight_file, num_layers, dropout=dropout, **other_params).cuda()

    def forward(self, sentences):
        character_ids = batch_to_ids(sentences).cuda()

        return self.elmo(character_ids)
