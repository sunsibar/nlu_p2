'''
Combine RNN and static features and build a classifier
on top of them.
'''
import numpy as np
import tensorflow as tf

class SimpleEndingClassifier:
    ''' Uses binary logistic regression'''

    def __init__(self, config, use_rnn=True, num_features=None):
        self.config = config
        self.use_rnn = use_rnn
        self.data_dir = config['data_dir']
        self.output_dir = config['output_dir']
        self.train_data_file = config['train_data_file']
        self.story_cloze_file = config['story_cloze_file']
        self.mode = config['mode']
        assert self.mode in ['training', 'validation', 'inference']
        self.static_features = config['static_features']
        self.vocab_size = config['vocab_size']
        self.limit_num_samples = config['limit_num_samples'] = 20  # Unused ?

        self._num_features = num_features

        self.reuse = True if self.mode == 'validation' else False # have two models in parallel, on training, one validation; same weights

        self.binlog_classifier = BinaryLogisticClassifier(self.mode, self.num_features)


    def build_model(self):
        self.binlog_classifier.build_model()



    def train(self, batch, sess):
        pass
        #


    # access points: self.binolog_classifier.predictions :
    #   batch_sz -sized vector giving 1's or 2's

    def infer(self, batch, sess):
        pass


    @property
    def inputs(self):
        return self.binlog_classifier.inputs

    @inputs.setter
    def inputs(self, value):
        self.binlog_classifier.inputs = value

    @property
    def predictions(self):
        return self.binlog_classifier.predictions

    @property
    def num_features(self, num_sentences=6):
        if self._num_features is not None:
            return self._num_features
        else:
            num_f = 0
            if self.use_rnn:
                num_f += num_sentences
            # Check that you didn't forget to increase the number of features for new features
            for key, value in zip(self.static_features.keys(), self.static_features.values()):
                assert key in ['sentence_lengths', 'sentiment']

            if self.static_features['sentence_lengths']:
                num_f += num_sentences
            if self.static_features['sentiment']:
                raise NotImplementedError
            self._num_features = num_f
            return self._num_features


def get_features(batch, feature_dict, use_rnn=False, sess=None, rnn=None):
    '''
    :param feature_dict: a dictionary with 'True' or 'False' values for various feature types
    if use_rnn, calculate rnn features as well. This needs a session and an rnn to be passed.
    :return: a numpy array of shape batch_size x num_features
    '''
    num_sentences = batch.num_sentences()
    features = np.zeros(shape=(batch.batch_size(), 0))
    if use_rnn:
        assert sess is not None and rnn is not None
        rnn_feats = get_RNN_features(sess, rnn, batch)
        np.stack((features, rnn_feats), axis=1)

    sent_lens = []
    if feature_dict['sentence_lengths']:
        for i in range(num_sentences):
            sent_lens.append(batch.sent_len(i))
        sent_lens = np.array(sent_lens)  # .transpose()
    features.append(sent_lens)
    if feature_dict['sentiment']:
        raise NotImplementedError

    # make batch dimension be the outer dimension
    features = np.array(features).reshape(shape=(-1, features.shape[-1])).transpose()
    return features


def get_RNN_features(self, sess, rnn, batch):
        assert self.use_rnn
        ending_1 = [5]
        ending_2 = [6]
        sent_without_ending = [0,1,2,3,4]
        sent_n_ending_1 = [0,1,2,3,4,5]
        sent_n_ending_2 = [0,1,2,3,4,6]

        # first, get full story probabilities
        ending_idx         = batch.sents_len(sent_without_ending)
        feed_dict          = rnn.get_feed_dict_train(batch, which_sentences=sent_n_ending_1)
        p_end1_I_story = sess.run([rnn.word_probabs],   feed_dict=feed_dict)
        p_end1_I_story = np.product(p_end1_I_story[ : , ending_idx : ], axis=1)
        feed_dict          = rnn.get_feed_dict_infer(batch, which_sentences=sent_n_ending_2)
        p_end2_I_story = sess.run([rnn.word_probabs],   feed_dict=feed_dict)
        p_end2_I_story = np.product(p_end2_I_story[ : , ending_idx : ], axis=1)

        # then, get both endings' probability
        feed_dict   = rnn.get_feed_dict_train(batch, which_sentences=ending_1)
        p_end1      = sess.run([rnn.sequence_probab], feed_dict=feed_dict)
        feed_dict   = rnn.get_feed_dict_train(batch, which_sentences=ending_2)
        p_end2      = sess.run([rnn.sequence_probab], feed_dict=feed_dict)

        # get additional features p_endi_I_sent / p_endi
        p1_I_by_p1 =  p_end1_I_story / p_end1
        p2_I_by_p2 =  p_end2_I_story / p_end2

        probabs = np.array([p_end1, p_end2, p_end1_I_story, p_end2_I_story, p1_I_by_p1, p2_I_by_p2])
        probabs = probabs.transpose()
        assert probabs.shape[1] == 6 and probabs.shape[0] == batch.batch_size()
        #    probabs = {'p_end1': p_end1,                        'p_end2': p_end2,
        #               'p_end1_given_story': p_end1_I_story,    'p_end2_given_story': p_end2_I_story,
        #               'p1_I_by_p1':,   'p2_I_by_p2':}
        return probabs


class BinaryLogisticClassifier:
    ''' Tries to predict a 1 or a 2 from the input.
        Very simple; storage and such are left to the caller.'''

    def __init__(self, mode, num_features):
        self.mode = mode
        assert self.mode in ['training', 'validation', 'inference']
        self.inputs = tf.placeholder(tf.float32, shape=[None, num_features], name='input_pl')
        self.targets = tf.placeholder(tf.int32, shape=[None,], name='target_pl')

        self.reuse = True if self.mode == 'validation' else False  # have two models in parallel, on training, one validation; same weights

    def build_model(self):

        with tf.variable_scope('binary_logistic_classifier', reuse=self.reuse):
            self.logit_1 = tf.contrib.layers.fully_connected(self.inputs, 1, activation_fn=None)
            if self.mode is not 'inference':
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logit_1)
            self.logit_2 = tf.fill(tf.shape(self.logit_1), 1.)
            self.logits = tf.concat([self.logit_1, self.logit_2], axis= 1)
            self.predictions = tf.argmax(self.logits, axis=1)
            self.predictions += tf.ones_like(self.predictions)
            #self.predictions = tf.less(self.logit_1, 1.)
            #tf.argmax(self.logits, axis=1) + tf.constant(1, shape=(self.logits.shape[0]))