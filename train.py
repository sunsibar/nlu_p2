'''
Trains a full model
'''
import os
import numpy as np
import tensorflow as tf
from RNN_model import RNNModel
from dataset import storydata_from_csv, Preprocessor
from train_utils import get_model_and_placeholders
from full_model import SimpleEndingClassifier



# ---  set up datasets --- #
def set_up_datasets(config):

    train_data_file = os.path.join(config['data_dir'], config['train_data_file'])
    story_dataset_train, story_dataset_val = \
        storydata_from_csv(train_data_file, config['rnn_config']['batch_size'],
                           has_titles=True, has_ending_labels=False)
    prep = Preprocessor(config, dataset=story_dataset_train)
    #   story_dataset_train.preprocess(prep)
    #   story_dataset_val.preprocess(prep)

    quiz_file = os.path.join(config['data_dir'], config['StoryCloze_file'])
    story_dataset_quiz_train, story_dataset_quiz_val = \
        storydata_from_csv(quiz_file, config['rnn_config']['batch_size'],
                           has_titles=False, has_ending_labels=True)
    story_dataset_quiz_train.preprocess(prep)
    story_dataset_quiz_val.preprocess(prep)

    return story_dataset_quiz_train, story_dataset_quiz_val



def main(config):

    dataset_train, dataset_val = set_up_datasets(config)

    rnn_model_cls, placeholders = get_model_and_placeholders(config)
    classifier = SimpleEndingClassifier(config, placeholders, use_rnn=config['use_rnn'])

    # Load trained RNN model
    rnn = sess = saver = None
    if config['use_rnn']:
        rnn = rnn_model_cls(config['rnn_config'])
        sess = tf.Session()
        saver = tf.train.Saver()

        ckpt_id = config['rnn_checkpoint_id']
        if ckpt_id is None:
            ckpt_path = tf.train.latest_checkpoint(config['rnn_model_dir'])
        else:
            ckpt_path = os.path.join(os.path.abspath(config['rnn_model_dir']), ckpt_id)
        print('Evaluating ' + ckpt_path)
        saver.restore(sess, ckpt_path)

    try:
        # Build full_model "on top" of it


        # For each batch of data, do:

        # Get the classifications from the model
        # results.append((index where probab is higher) + 1) --> "1" or "2"



    finally:
        if config['use_rnn']:
            saver.save(sess, config['output_dir'], global_step=None) # TODO
            sess.close()


if __name__ == "__main__":
    from config import config
    main(config)