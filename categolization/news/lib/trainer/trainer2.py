import os
import numpy as np

from config import conf
from lib.utility import utils
from lib.model.dlnn_karal import Dlnn as dlnn


def main():
    meta = utils.pickle_load(conf.METADATA_FILE)
    label_and_data = utils.pickle_load(conf.DATA_FILE)
    np.random.shuffle(label_and_data)
    length = len(label_and_data)
    boundary = int(length * conf.TRAINING_DATA_RATIO)
    test = label_and_data[boundary:]
    train = label_and_data[0:boundary - 1]
    print(len(test))
    print(len(train))
    print(meta.categories)
    #
    np.random.shuffle(train)
    labels = train[:, 0]
    data = train[:, 1]
    n_batch = len(labels)
    n_categories = len(meta.categories)
    predict_label = test[:, 0].tolist()
    predict_data = test[:, 1].tolist()

    listed_data, listed_label = [],  []
    data_append, label_append = listed_data.append, listed_label.append
    for d in data:
        data_append(list(map(float, d)))
    for label in labels:
        label_append(list(map(float, label)))

    model = dlnn(conf.NUM_UNITS1, n_categories,
                 np.array(listed_data), np.array(listed_label))
    model.set_model()
    model.fit()

if __name__ == '__main__':
    main()

