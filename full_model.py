'''
Combine RNN and static features and build a classifier
on top of them.
'''
import tensorflow as tf


class SimpleEndingClassifier:
    ''' Uses binary logistic regression'''

    def __init__(self, config, placeholders, use_rnn=True):
        self.config = config
        self.use_rnn = use_rnn
        self.data_dir = config['data_dir']
        self.output_dir = config['output_dir']
        self.train_data_file = config['train_data_file']
        self.story_cloze_file = config['StoryCloze_file']
        self.mode = config['mode']
        assert self.mode in ['training', 'validation', 'inference']
        self.static_features = config['static_features']
        self.vocab_size = config['vocab_size']
        self.limit_num_samples = config['limit_num_samples'] = 20  # Unused ?

        self.inputs = placeholders['input_pl']
        self.inputs_rnn = placeholders['input_rnn_pl']
        self.targets = placeholders['target_pl']
        self.reuse = True if self.mode == 'validation' else False # have two models in parallel, on training, one validation; same weights


    def build_model(self):

        with tf.variable_scope('simple_classifier', reuse=self.reuse):
            self.logit_1 = tf.contrib.layers.fully_connected(self.inputs, 1, activation_fn=None)
            self.logit_2 = tf.constant(1.)
            self.logits = tf.tuple([self.logit_1, self.logit_2])
            if self.mode is not 'inference':
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logit_1)
            self.predictions = tf.argmax(self.logits, axis=1)


        pass
        # For each batch of data, do:
            # Extract two last-sentence static features
            # Extract two full-sentence static features
            # Extract two last-sentence RNN-probabs p(ending)
            # Extract two full-sentence RNN-probabs p(ending, beginning)
            # Calculate p(ending | beginning) for both endings
            # Set a classifier on top

            # access points: self.classification :
            #   [p1, p2] giving the two probabilities that ending 1 or ending 2 is correct