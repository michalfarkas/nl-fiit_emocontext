import numpy


def get_one_hot(targets, nb_classes):
    res = numpy.eye(nb_classes)[numpy.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])


def get_random_number():
    return 4  # chosen by a fair dice roll; guaranteed to be random
