import torch
from pytoune.framework.model import Model
import config
from utils import print_utils as p_utils
import random
from pytoune.framework import Callback
from tqdm import tqdm
import numpy
from utils.numeric_utils import get_one_hot
from Baseline.baseline import getMetrics
from Models.lstm_classifier import LSTMClassifier
from pytoune.framework import Model, BestModelRestore, EarlyStopping, ClipNorm, MultiStepLR
import time
from metrics import microF1
import pandas as pd


class SampleWeightsSetter(Callback):
    def __init__(self, generator):
        super(SampleWeightsSetter, self).__init__()
        self.weights = None
        self.generator = generator

    def set_new_weights(self, weights):
        self.weights = weights

    def on_batch_begin(self, batch, logs):
        if self.weights:
            batch_data = self.generator[batch-1]
            batch_weights = list()
            for sample in batch_data[0][0]:
                batch_weights.append(self.weights[sample.item()])
            batch_weights_t = torch.Tensor(numpy.stack(batch_weights)).cuda()
            self.model.loss_function.set_sample_weight(batch_weights_t)

class StackedClassifier(torch.nn.Module):
    def __init__(self, ensemble_size, n_classes, **kwargs):
        super(StackedClassifier, self).__init__()
        self.ensemble_size =ensemble_size
        self.n_classes = n_classes
        self.layers = torch.nn.ModuleList()
        self.n_layers = kwargs.get('layers', 1)
        self.activation = kwargs.get('activation', torch.nn.LeakyReLU)()
        for i in range(0, self.n_layers):
            if i== self.n_layers-1:
                new_layer = torch.nn.Linear(self.n_classes*self.ensemble_size, 4)
            else:
                new_layer = torch.nn.Linear(self.n_classes*self.ensemble_size, self.n_classes*self.ensemble_size)
            self.layers.append(new_layer)

    def forward(self, x):
        x = torch.nn.functional.softmax(x, dim=1)
        for layer in self.layers:
            x = self.activation(layer(x))
        return torch.nn.functional.log_softmax(x, dim=1)


class EnsembleLoss(torch.nn.Module):
    def __init__(self, base_loss):
        super(EnsembleLoss, self).__init__()
        self.base_loss_fn = base_loss(torch.Tensor(config.label_weights).cuda(), reduction='none')
        self.sample_weights = torch.Tensor([1]).cuda().detach()

    def forward(self, x, y):
        loss = self.base_loss_fn(x, y)
        if not y.size() == self.sample_weights.size():
            sample_weights = self.sample_weights.expand(y.size())
        else:
            sample_weights = self.sample_weights
        weighted_loss = sample_weights.mul(loss)
        self.sample_weights = torch.Tensor([1]).cuda().detach()
        return weighted_loss.mean()

    def set_sample_weight(self, value):
        self.sample_weights = value


class StackGenerator():
    def __init__(self, models_pt, base_generator, reset=True):
        self.data_list = list()
        self.labels = list()
        base_generator.set_reset(False, None)
        for model_pt in models_pt:
            model_pt.cuda()
            predictions_list = list()
            labels_list = list()
            for batch_index in tqdm(range(0, len(base_generator))):
                sample_x, sample_y = base_generator[batch_index]
                predictions = model_pt.predict_on_batch(sample_x)
                predictions_list.append(predictions)
                labels_list.append(sample_y)
            self.labels = numpy.concatenate(labels_list)
            self.data_list.append(numpy.concatenate(predictions_list))
            model_pt.cpu()
        self.allow_reset = reset
        self.data_stacked = numpy.stack(self.data_list, axis=0).transpose((1, 0, 2))
        self.shuffle_list = numpy.arange(self.labels.size)
    def __getitem__(self, index):
        index = index % int(numpy.ceil(self.data_stacked.shape[1] / config.batch_size))
        if index == 0 and self.allow_reset:
            self.reset()
        sample = self.data_stacked[self.shuffle_list[index * config.batch_size:(index + 1) * config.batch_size]]
        label = self.labels[self.shuffle_list[index * config.batch_size:(index + 1) * config.batch_size]]
        return torch.tensor(sample.reshape((sample.shape[0], sample.shape[1]*sample.shape[2])), dtype=torch.float32).cuda(), torch.tensor(label).cuda()
    def reset(self, state=None):
        if state is not None:
            random.setstate(state)
        random.shuffle(self.shuffle_list)
    def __len__(self):
        return len(self.labels)

class Ensemble():
    def __init__(self, size, loss, train_gen, val_gen, type= "stack",rebalance=None):
        self.constituents = list()
        self.constituents_pt = list()
        self.constituents_callbacks = list()
        self.size = size
        self.train_generator = train_gen
        self.validation_generator = val_gen
        self.weight_setter = SampleWeightsSetter(train_gen)
        self.rebalance = rebalance
        self.loss_fn = EnsembleLoss(loss)
        self.type = type
    def set_up_models(self, embeddings):
        print("..Seting up the models")
        classifier_params = config.lstm_classifier_params if config.model is LSTMClassifier else config.hierarchical_lstm_classifier_params
        vocabulary, n_embeddings, embedding_layer = embeddings
        for i in range(0, self.size):
            model = config.model(embedding_layer, n_embeddings, **classifier_params).cpu()
            model_restore_callback = BestModelRestore(monitor=config.restore_monitor, mode=config.restore_mode,
                                                      verbose=True)
            early_stopping_callback = EarlyStopping(monitor=config.early_stop_monitor,
                                                    min_delta=config.early_stop_min_delta,
                                                    patience=config.early_stop_patience,
                                                    mode=config.early_stop_mode,
                                                    verbose=True)
            grad_clip_callback = ClipNorm(model.parameters(), config.max_grad_norm)
            lr_callback_2 = MultiStepLR(milestones=[3])
            model_pt = Model(model, config.optimizer, self.loss_fn, metrics=['acc', microF1]).cpu()

            self.constituents.append(model)
            self.constituents_pt.append(model_pt)
            self.constituents_callbacks.append((model_restore_callback, early_stopping_callback, grad_clip_callback, lr_callback_2))
        if self.type == "stack":
            self.stack_model = StackedClassifier(self.size, config.classes)
            model_restore_callback = BestModelRestore(monitor=config.restore_monitor, mode=config.restore_mode,
                                                      verbose=True)
            early_stopping_callback = EarlyStopping(monitor=config.early_stop_monitor,
                                                    min_delta=config.early_stop_min_delta,
                                                    patience=config.early_stop_patience,
                                                    mode=config.early_stop_mode,
                                                    verbose=True)
            grad_clip_callback = ClipNorm(self.stack_model.parameters(), config.max_grad_norm)
            self.stack_model_pt = Model(self.stack_model, config.optimizer, torch.nn.NLLLoss(torch.Tensor(config.label_weights).cuda()), metrics=['acc', microF1]).cpu()
            self.stack_callbacks = [model_restore_callback, early_stopping_callback, grad_clip_callback]

    def train(self):
        print("..Training the models")
        random_state = random.getstate()
        running_prediction_dict = dict()
        weights_dict = dict()
        for index, (model, model_pt, callbacks) in enumerate(zip(self.constituents, self.constituents_pt, self.constituents_callbacks)):
            print("....Model {}".format(index))
            random.setstate(random_state)
            self.train_generator.set_reset(True, random_state)
            model_pt.cuda()
            with p_utils.add_indent(3):
                model_pt.fit_generator(
                    self.train_generator,
                    self.validation_generator,
                    epochs=config.max_epochs,
                    callbacks=list(callbacks + (self.weight_setter,))
                )
            if self.rebalance:
                self.calculate_new_sample_weights(model_pt, random_state, running_prediction_dict, index, weights_dict)
            model_pt.cpu()
            model_pt.optimizer = None
            self.constituents_callbacks[index] = None
        self.weight_setter.set_new_weights(None)
        if self.type == "stack":
            self.train_stack_model()

    def train_stack_model(self):
        self.stack_model_pt.cuda()
        train_generator = StackGenerator(self.constituents_pt, self.train_generator)
        validation_generator = StackGenerator(self.constituents_pt, self.validation_generator)
        with p_utils.add_indent(3):
            self.stack_model_pt.fit_generator(
                train_generator,
                validation_generator,
                epochs=30,
                callbacks=self.stack_callbacks,
            )

    def calculate_new_sample_weights(self, model_pt, random_state, running_prediction_dict, index, weights_dict):
        print("....Calculating new sample weights")
        time.sleep(0.5)
        self.train_generator.set_reset(True, random_state)
        random.setstate(random_state)
        for batch_index in tqdm(range(0, len(self.train_generator))):
            sample_x, sample_y = self.train_generator[batch_index]
            loss, acc, predictions = model_pt.evaluate_on_batch(sample_x, sample_y, return_pred=True)
            for sample_id, label, prediction in zip(sample_x[0], sample_y, predictions):
                prediction = get_one_hot(prediction.argmax(), 4)
                label = get_one_hot(numpy.array(label.item()), 4)
                if not sample_id.item() in running_prediction_dict:
                    running_prediction_dict[sample_id.item()] = prediction
                else:
                    running_prediction_dict[sample_id.item()] += prediction
                weights_dict[sample_id.item()] = (index + 2) / (numpy.sum(running_prediction_dict[sample_id.item()] * label) + 1)
                # weights_dict[sample_id.item()] = (1-(numpy.sum(prediction * label)/index))*config.ensemble_max_weight
        self.weight_setter.set_new_weights(weights_dict)
        time.sleep(0.5)

    def evaluate(self, test_generator, labels_test):
        print("..Evaluation")
        predictions_all = list()
        for index, (model_pt, model) in enumerate(zip(self.constituents_pt, self.constituents)):
            model_pt.cuda()
            model.cuda()
            print("....Model {}".format(index))
            print("......evaluating through Pytoune")
            model_pt.loss_function.set_sample_weight(torch.Tensor([1]).cuda().detach())
            loss_pt, accuracy_pt, predictions = model_pt.evaluate_generator(
                test_generator,
                return_pred=True
            )
            print("........loss: {}".format(loss_pt))
            print("........accuracy: {}".format(accuracy_pt))
            predictions_all.append(numpy.concatenate(predictions))
            print("......evaluating through baseline script")
            with p_utils.add_indent(4):
                metrics = getMetrics(numpy.concatenate(predictions), get_one_hot(numpy.array(labels_test), 4))
            model_pt.cpu()
            model.cpu()

        print("....Ensemble evaluation (through baseline script)")
        with p_utils.add_indent(3):
            predictions_all_stacked = numpy.stack(predictions_all).transpose(1,  0, 2)
            if self.type == "stack":
                combined_predictions = self.stack_model_pt.predict(predictions_all_stacked.reshape((predictions_all_stacked.shape[0], predictions_all_stacked.shape[1]*predictions_all_stacked.shape[2])))
            elif self.type == "sum":
                combined_predictions = self.sum_predictions(predictions_all)
            elif self.type == "vote":
                combined_predictions = self.vote_predictions(predictions_all)
            metrics = getMetrics(combined_predictions, get_one_hot(numpy.array(labels_test), 4))
        return metrics

    def get_submission_file(self, submission_generator, raw_data):
        print("..Creating submission file")
        predictions_all = list()
        for index, (model_pt, model) in enumerate(zip(self.constituents_pt, self.constituents)):
            model_pt.cuda()
            model.cuda()
            model_pt.loss_function.set_sample_weight(torch.Tensor([1]).cuda().detach())
            predictions = model_pt.predict_generator(
                submission_generator,
            )
            predictions_all.append(numpy.concatenate(predictions))
            model_pt.cpu()
            model.cpu()

        with p_utils.add_indent(3):
            predictions_all_stacked = numpy.stack(predictions_all).transpose(1, 0, 2)
            if self.type == "stack":
                combined_predictions = self.stack_model_pt.predict(predictions_all_stacked.reshape((predictions_all_stacked.shape[0], predictions_all_stacked.shape[1] * predictions_all_stacked.shape[2])))
            elif self.type == "sum":
                combined_predictions = self.sum_predictions(predictions_all)
            elif self.type == "vote":
                combined_predictions = self.vote_predictions(predictions_all)

        labels = numpy.stack([config.index_to_labels[numpy.argmax(label)] for label in combined_predictions])
        raw_data['label'] = pd.Series(labels, index=raw_data.index)
        raw_data.to_csv("test.txt", sep="\t", index=False)
        return None

    def normalize_predictions(self, prediction):
        return numpy.expand_dims(1 / (prediction.sum(axis=0) / prediction.shape[0]),  axis=0) * prediction[0]

    def sum_predictions(self, predictions):
        sum = numpy.stack(predictions).sum(axis=0)
        return get_one_hot(sum.argmax(axis=1), 4)
    def vote_predictions(self, predictions):
        leaderboard = numpy.zeros(predictions[0].shape)
        for index, prediction in enumerate(predictions):
            for j in range(prediction.shape[0]):
                leaderboard[j] += get_one_hot(prediction[j].argmax(), 4)
        return get_one_hot(leaderboard.argmax(axis=1), 4)
    def rank_predictions(self):
        raise NotImplemented

