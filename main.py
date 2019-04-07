import torch
from Models import lstm_classifier
from utils.vectors import Vectors
from Data.multi_turn_classification_dataset import MultiTurnClassificationDataset
from Preprocessing.ekphrasis_proxy import EkphrasisProxy
from Preprocessing.preprocessing import Preprocessing
import config
import pandas as pd
from pytoune.framework import Model, BestModelRestore, EarlyStopping, ClipNorm, MultiStepLR
import numpy
from Baseline.baseline import getMetrics
import time
import utils.print_utils as p_utils
import utils.numeric_utils as n_utils
from Models.ensembles import Ensemble
from metrics import microF1


class pytoune_generator():
    def __init__(self, dataset, turns, allow_reset=True, return_labels=True):
        self.dataset = dataset
        self.turns = turns
        self.allow_reset = allow_reset
        self.return_labels = return_labels
        self.state = None

    def __getitem__(self, index):
        index = index % int(numpy.ceil(len(self.dataset) / config.batch_size))
        if index == 0 and self.allow_reset:
            self.dataset.reset(self.state)
        result = self.dataset[index*config.batch_size:(index+1)*config.batch_size]
        return (result[:-1], result[-1]) if self.return_labels else result[:-1]

    def __len__(self):
        return len(self.dataset) // config.batch_size + 1

    def set_reset(self, value, state):
        self.allow_reset = value
        self.state = state


def load_data():
    print("..Loading the data")
    print("....Glove")
    vectors = Vectors(**{'file_name': config.embedding_file, 'embedding_dim': 300}).glove()

    print("....Setting up preprocessor")
    with p_utils.add_indent(3):
        if config.preprocessor == "Ekphrasis":
            preprocessor = EkphrasisProxy().preprocess_text
        elif config.preprocessor == "NL-FIIT":
            preprocessor = Preprocessing().process_test
        else:
            raise NotImplementedError

    print("....Data")
    raw_data = pd.read_csv(config.train_file, sep="\t", header=0, quoting=3).sample(frac=1)
    rawish_data = raw_data['turn1'] + "\t" + raw_data['turn2'] + "\t" + raw_data['turn3']
    labels_as_indices = [config.labels_to_index[label] for label in raw_data['label']]
    if config.train_val_test_split is not None:
        rawish_train = rawish_data.iloc[0:int(raw_data.shape[0]*config.train_val_test_split[0])]
        rawish_val = rawish_data.iloc[int(raw_data.shape[0]*config.train_val_test_split[0]):int(raw_data.shape[0]*(config.train_val_test_split[0]+config.train_val_test_split[1]))]
        labels_val = labels_as_indices[int(raw_data.shape[0] * config.train_val_test_split[0]):int(raw_data.shape[0] * (config.train_val_test_split[0] + config.train_val_test_split[1]))]
        labels_train = labels_as_indices[0:int(raw_data.shape[0] * config.train_val_test_split[0])]
    else:
        rawish_train = rawish_data
        labels_train = labels_as_indices
        raw_val = pd.read_csv(config.validation_file, sep="\t", header=0, quoting=3)
        rawish_val = raw_val['turn1'] + "\t" + raw_val['turn2'] + "\t" + raw_val['turn3']
        labels_val = [config.labels_to_index[label] for label in raw_val['label']]


    if config.test:
        if config.train_val_test_split is not None and len(config.train_val_test_split) > 2:
            rawish_test = rawish_data.iloc[int(raw_data.shape[0]*(config.train_val_test_split[0]+config.train_val_test_split[1])):]
            labels_test = labels_as_indices[int(raw_data.shape[0] * (config.train_val_test_split[0] + config.train_val_test_split[1])):]
        else:
            raw_test = pd.read_csv(config.test_file_w_labels, sep="\t", header=0, quoting=3)
            rawish_test = raw_test['turn1'] + "\t" + raw_test['turn2'] + "\t" + raw_test['turn3']
            labels_test = [config.labels_to_index[label] for label in raw_test['label']]

    with p_utils.add_indent(3, mode='err'):
        train_data = MultiTurnClassificationDataset(rawish_train,
                                                    labels_train,
                                                    vocab=vectors[0],
                                                    preprocessing=preprocessor,
                                                    split_turns=config.split_turns)
        val_data = MultiTurnClassificationDataset(rawish_val,
                                                  labels_val,
                                                  vocab=vectors[0],
                                                  preprocessing=preprocessor,
                                                  split_turns=config.split_turns)

    train_generator = pytoune_generator(train_data, config.turns)
    val_generator = pytoune_generator(val_data, config.turns)

    result = (train_generator, val_generator)
    if config.test:
        with p_utils.add_indent(3, mode='err'):
            test_data = MultiTurnClassificationDataset(rawish_test,
                                                       labels_test,
                                                       vocab=vectors[0],
                                                       preprocessing=preprocessor,
                                                       split_turns=config.split_turns)
        test_generator = pytoune_generator(test_data, config.turns, allow_reset=False)
        result += (test_generator, labels_test)

    if config.submission_file is not None:
        raw_submission = pd.read_csv(config.test_file_wo_labels, sep="\t", header=0, quoting=3)
        rawish_submission = raw_submission['turn1'] + "\t" + raw_submission['turn2'] + "\t" + raw_submission['turn3']

        with p_utils.add_indent(3, mode='err'):
            submission_data = MultiTurnClassificationDataset(rawish_submission,
                                                             y=None,
                                                             vocab=vectors[0],
                                                             preprocessing=preprocessor,
                                                             split_turns=config.split_turns)
        submission_generator = pytoune_generator(submission_data, config.turns, allow_reset=False, return_labels=False)
        return result + (vectors,) + (raw_submission, submission_generator)


    return result + (vectors, )


def set_up_model(vectors, **kwargs):
    print("..Setting up the model")
    classifier_params = config.lstm_classifier_params if config.model is lstm_classifier.LSTMClassifier else config.hierarchical_lstm_classifier_params
    model = config.model(vectors[2], vectors[1], **classifier_params).cuda()
    model_restore_callback = BestModelRestore(monitor=config.restore_monitor, mode=config.restore_mode, verbose=True)
    early_stopping_callback = EarlyStopping(monitor=config.early_stop_monitor,
                                            min_delta=config.early_stop_min_delta,
                                            patience=config.early_stop_patience,
                                            mode=config.early_stop_mode,
                                            verbose=True)
    grad_clip_callback = ClipNorm(model.parameters(), config.max_grad_norm)
    # lr_callback_1 = ReduceLROnPlateau(mode="max", patience=1, monitor="val_microF1", threshold=0.002, min_lr=1e-6, verbose=True, threshold_mode='abs')
    lr_callback_2 = MultiStepLR(milestones=[2])
    loss_fn = torch.nn.CrossEntropyLoss(torch.Tensor(config.label_weights).cuda())
    pt_fw = Model(model, config.optimizer, loss_fn, metrics=['acc', microF1])
    return pt_fw, [model_restore_callback, early_stopping_callback, grad_clip_callback], model


def train(pt_fw, train_generator, val_generator, callbacks):
    print("..Training the model")
    with p_utils.add_indent(2):
        pt_fw.fit_generator(
            train_generator,
            val_generator,
            epochs=config.max_epochs,
            callbacks=callbacks
          )


def evaluate(pt_fw, test_generator, labels_test):
    print("..Evaluating through Pytoune")
    loss_pt, accuracy_pt, predictions = pt_fw.evaluate_generator(
        test_generator,
        return_pred=True
    )
    print("....loss: {}".format(loss_pt))
    print("....accuracy: {}".format(accuracy_pt))
    #print("....F1: {}".format(f1_pt))

    print("..Evaluating through baseline script")
    with p_utils.add_indent(2):
        metrics = getMetrics(numpy.concatenate(predictions), n_utils.get_one_hot(numpy.array(labels_test), 4))
    return metrics
def get_submission_file(pt_fw, raw_submission, submission_generator):
    labels_raw = pt_fw.predict_generator(submission_generator)
    labels = numpy.concatenate([numpy.stack([config.index_to_labels[numpy.argmax(label)] for label in labels_batch]) for labels_batch in labels_raw])
    raw_submission['label'] = pd.Series(labels, index=raw_submission.index)
    raw_submission.to_csv("test.txt", sep="\t", index=False)

if __name__ == '__main__':
    if config.test and not config.ensemble:
        train_generator, val_generator, test_generator, labels_test, vectors, raw_submission, submission_generator = load_data()
        time.sleep(1)
        pt_fw, callbacks, _ = set_up_model(vectors)
        time.sleep(1)
        train(pt_fw, train_generator, val_generator, callbacks)
        time.sleep(1)
        evaluate(pt_fw, test_generator, labels_test)
        get_submission_file(pt_fw, raw_submission, submission_generator)
    elif config.test and config.ensemble:
        train_generator, val_generator, test_generator, labels_test, vectors, raw_submission, submission_generator = load_data()
        time.sleep(1)
        ens = Ensemble(config.ensemble, torch.nn.CrossEntropyLoss, train_generator, val_generator, rebalance=config.ensemble_rebalance, type=config.ensemble_type)
        time.sleep(1)
        ens.set_up_models(vectors)
        time.sleep(1)
        ens.train()
        time.sleep(1)
        ens.evaluate(test_generator, labels_test)
        ens.get_submission_file(submission_generator, raw_submission)
    print("DONE")
