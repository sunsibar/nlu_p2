
import tensorflow as tf
from RNN_model import RNNModel
from full_model import SimpleEndingClassifier

# Do this outside of the network, to guarantee that every rnn model uses the same placeholders.
# Then, train.py can generically train all rnns (hopefully).
def get_rnn_model_and_placeholders(rnn_config):
    # create placeholders that we need to feed the required data into the model
    # input, output: 2d array, batch_sz x max_num_words
    #   transforming word-id-lists to (huge) vectors is handled in tf's embedding functionality (for input)
    #   and in its sequence2sequence_loss (output)
    input_pl = tf.placeholder(tf.float32, shape=[None, None], name='rnn_input_pl')
    target_pl = tf.placeholder(tf.float32, shape=[None, None], name='rnn_output_pl')
    seq_lengths_pl = tf.placeholder(tf.int32, shape=[None], name='seq_lengths_pl')
    #mask_pl = tf.placeholder(tf.float32, shape=[None, None], name='rnn_mask_pl')

    placeholders = {'rnn_input_pl': input_pl,
                    'rnn_target_pl': target_pl,
                    'seq_lengths_pl': seq_lengths_pl}

    if rnn_config['model'] =="simple":
        rnn_model_class = RNNModel
    else:
        raise KeyError('Unknown model: '+rnn_config['model'])
    return rnn_model_class, placeholders


def get_model_and_placeholders(config):

    input_pl = tf.placeholder(tf.float32, shape=[None, None], name='input_pl')
    #input_rnn_probab_pl = tf.placeholder(tf.float32, shape=[None, None], name='input_rnn_probab_pl')
    target_pl = tf.placeholder(tf.float32, shape=[None], name='output_pl')
    #seq_lengths_pl = tf.placeholder(dtype=tf.int32, shape=[None,],name='sequence_length_list')
    _, rnn_placeholders = get_rnn_model_and_placeholders(config.rnn_config)

    placeholders = {'input_pl': input_pl,
                    #'input_rnn_pl': input_rnn_probab_pl,
                    'target_pl': target_pl}
    placeholders_combined = {**placeholders, **rnn_placeholders}

    if config['model'] =="simple":
        model_class = SimpleEndingClassifier
    else:
        raise KeyError('Unknown model: '+config['model'])
    return model_class, placeholders_combined