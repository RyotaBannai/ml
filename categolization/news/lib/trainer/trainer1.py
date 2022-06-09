
import os
import numpy as np

from config import conf
from lib.utility import utils
from lib.model import dlnn_tf as dlnn


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
    num_categories = len(meta.categories)
    predict_label = test[:, 0].tolist()
    predict_data = test[:, 1].tolist()

    model = dlnn.DoubleLayerNetwork(conf.LEARNING_RATE,
                                    conf.NUM_UNITS1, conf.NUM_UNITS2,
                                    conf.PCA_DIMENSION, num_categories,
                                    conf.LOG_FILE)

    for step in range(conf.TOTAL_STEP):

        np.random.shuffle(train)
        batch_label = train[:conf.BATCH_SIZE, 0].tolist()
        batch_data = train[:conf.BATCH_SIZE, 1].tolist()
        model.sess.run(model.train_step, feed_dict={
            model.x: batch_data, model.t: batch_label,
            model.keep_prob: conf.KEEP_PROB})

        if step % 100 == 0:
            stdout_log(
                step=step,
                cur_model=model,
                data=predict_data,
                label=predict_label
            )

            dirname = str(conf.CORE_DIR)
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            model.saver.save(model.sess, dirname + "/model.ckpt")



def stdout_log(step, cur_model, data, label):
    summary, loss_val, acc_val = cur_model.sess.run(
        [cur_model.summary, cur_model.loss, cur_model.accuracy],
        feed_dict={cur_model.x: data, cur_model.t: label, cur_model.keep_prob: 1.0}
    )
    print('Step: %d, Loss: %f, Accuracy: %f' % (step, loss_val, acc_val))
    cur_model.writer.add_summary(summary, step)


if __name__ == '__main__':
    main()

