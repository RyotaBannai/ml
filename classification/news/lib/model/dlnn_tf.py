#import tensorflow as tf
# Solution to fix prob: AttributeError: module 'tensorflow' has no attribute 'placeholder'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# EOS
import os
from config import conf


class DoubleLayerNetwork:
    """
    b: bias
    w: weight in each node

    """
    def __init__(self, learning_rate,
                 num_units1, num_units2,
                 vec_dim, num_categories, logfile
                 ):
        with tf.Graph().as_default():
            self.prepare_model(learning_rate,
                               num_units1, num_units2,
                               vec_dim, num_categories
                               )
            self.prepare_session(logfile)

    def prepare_model(self, learning_rate,
                      num_units1, num_units2,
                      vec_dim, num_categories
                      ):
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, vec_dim])

        with tf.name_scope('hidden1'):
            w1 = tf.Variable(tf.truncated_normal([vec_dim, num_units1]))  # Outputs random values from a truncated normal distribution.
            b1 = tf.Variable(tf.zeros(num_units1))
            hidden1 = tf.nn.relu(tf.matmul(x, w1) + b1)

        # with tf.name_scope('hidden2'):
        #     w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
        #     b2 = tf.Variable(tf.zeros(num_units2))
        #     hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            hidden1_drop = tf.nn.dropout(hidden1, keep_prob) # Computes dropout: randomly sets elements to zero to prevent overfitting. 1.0 = no dropout

        with tf.name_scope('output'):
            w0 = tf.Variable(tf.zeros([num_units1, num_categories]))
            b0 = tf.Variable(tf.zeros([num_categories]))
            p = tf.nn.softmax(tf.matmul(hidden1_drop, w0) + b0)  # 出力層 simoidとか使える

        with tf.name_scope('optimizer'):
            t = tf.placeholder(tf.float32, [None, num_categories])
            loss = -1 * tf.reduce_sum(t * tf.log(p))
            train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss) # minimize :  combines calls compute_gradients() and apply_gradients()

        with tf.name_scope('evaluator'):
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))  # 精度計算: 第2パラメーターに1をセットすると、行ごとに最大となる列を返す.
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        # tf.summary.histogram('weights_hidden2', w2)
        # tf.summary.histogram('biases_hidden2', b2)
        # tf.summary.histogram('weights_hidden1', w1)
        # tf.summary.histogram('biases_hidden1', b1)
        # tf.summary.histogram('weights_output', w0)
        # tf.summary.histogram('biases_output', b0)

        self.x = x
        self.t = t
        self.p = p
        self.keep_prob = keep_prob
        self.train_step = train_step
        self.loss = loss
        self.accuracy = accuracy

    def prepare_session(self, logfile):
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saved_prams = str(conf.CORE_DIR / "model.ckpt")
        if os.path.exists(saved_prams+'.index'):
            saver.restore(sess, saved_prams)
        sess.run(tf.global_variables_initializer())
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(logfile, sess.graph)

        self.sess = sess
        self.saver = saver
        self.summary = summary
        self.writer = writer