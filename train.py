'''
Trains a full model
'''
import os
import shutil
import time
import numpy as np
import tensorflow as tf
from RNN_model import RNNModel
from dataset import storydata_from_csv, Preprocessor
from train_utils import get_rnn_model
from full_model import BinaryLogisticClassifier, get_features
import datetime

## TODO: add summaries


# ---  set up datasets --- #
def set_up_datasets(config):

    train_data_file = os.path.join(config['data_dir'], config['train_data_file'])
    story_dataset_train, story_dataset_val = \
        storydata_from_csv(train_data_file, config['rnn_config']['batch_size'],
                           has_titles=True, has_ending_labels=False)
    prep = Preprocessor(config, dataset=story_dataset_train)
    #   story_dataset_train.preprocess(prep)
    #   story_dataset_val.preprocess(prep)

    quiz_file = os.path.join(config['data_dir'], config['story_cloze_file'])
    story_dataset_quiz_train, story_dataset_quiz_val = \
        storydata_from_csv(quiz_file, config['rnn_config']['batch_size'],
                           has_titles=False, has_ending_labels=True)
    story_dataset_quiz_train.preprocess(prep)
    story_dataset_quiz_val.preprocess(prep)

    return story_dataset_quiz_train, story_dataset_quiz_val



def main(config, valid_config):

        # set the output dir
    timestamp = datetime.datetime.now().strftime("%y-%b-%d_%Hh%M-%S")  # str(int(time.time()))
    config['model_dir'] = os.path.abspath(
        os.path.join(config['output_dir'], config['name'] + '/' + timestamp))

    shutil.copy2("config_full.py", os.path.join(config['output_dir'], 'theconfig.txt'))
    print("write to {}\n".format(config['model_dir']), flush=True)


    dataset_train, dataset_val = set_up_datasets(config)
    #sample_batch = dataset_train.get_batch() #does not advance batch pointer


    # Load trained RNN model weights
    rnn = saver_rnn = rnn_sess = None
    rnn_graph = tf.Graph()
    if config['use_rnn']:
        ckpt_id = config['rnn_model_id']
        if ckpt_id is None:
            ckpt_path = tf.train.latest_checkpoint(config['rnn_model_dir'])
        else:
            ckpt_path = os.path.join(os.path.abspath(config['rnn_model_dir']), ckpt_id)

        with rnn_graph.as_default():
            rnn_model_cls = get_rnn_model(config['rnn_config'])
            rnn = rnn_model_cls(config['rnn_config'])
            saver_rnn = tf.train.Saver()#tf.train.import_meta_graph(ckpt_path + ".meta")
            rnn_sess = tf.Session()
            saver_rnn.restore(rnn_sess, ckpt_path)
            print('Loaded RNN weights from ' + ckpt_path)


    classifier_graph = tf.Graph()

    try:
        # set up
        with classifier_graph.as_default():

            classifier = BinaryLogisticClassifier(config, use_rnn=config['use_rnn'])
            classifier.build_graph()
            valid_classifier = BinaryLogisticClassifier(valid_config, use_rnn=config['use_rnn'])
            valid_classifier.build_graph()

            # set up training
            with tf.variable_scope('training'):
                learning_rate = tf.Variable(config['learning_rate'], name="learning_rate", trainable=False)

                tvars = tf.trainable_variables(scope='binary_logistic_classifier')
                grads, _ = tf.clip_by_global_norm(tf.gradients(classifier.loss, tvars), config['max_grad_norm'])
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                global_step = tf.Variable(0, name="global_step", trainable=False)
                train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                     global_step=global_step, name='train_classifier_op')

                #global_epoch = tf.Variable(0, name='global_epoch', trainable=False)
                #epoch_counter_op = tf.assign(global_epoch, global_epoch + 1, name='epoch_counter_op')


            saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=config['n_checkpoints_to_keep'])
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())  # let's hope this initializes only one graph
            sess.run(tf.local_variables_initializer())

        # start training
        start_time = time.time()
        for e in range(config['num_epochs']):
            step = tf.train.global_step(sess, global_step)

            train_loss = 0.
            for i, batch in enumerate(dataset_train.all_batches(shuffle=True)):
                # For each batch of data, do:

                # get features, static features need no tensorflow so doesn't matter that it's the rnn graph
                # need rnn graph in case we use an rnn
                with rnn_graph.as_default():
                    features = get_features(batch, config['static_features'], config['use_rnn'], rnn_sess, rnn)

                with classifier_graph.as_default():
                    # get the target labels
                    feed_dict = {classifier.inputs: features, classifier.targets: batch.ending_labels}
                    fetches = {'train_op': train_op, 'loss': classifier.loss, 'lr': learning_rate}
                    # train the model
                    train_output = sess.run(fetches, feed_dict)

                    train_loss += train_output['loss'] * batch.batch_size
                    time_delta = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                    print('\rEpoch: {:3d} [{:4d}/{:4d}] time: {:s} lr: {:f} loss: {:f}'.format(
                        e + 1, i + 1, dataset_train.n_batches, time_delta, train_output['lr'],
                        train_output['loss']), end='')



            valid_loss = 0.
            valid_accur = 0.
            for i, batch in enumerate(dataset_val.all_batches()):
                with rnn_graph.as_default():
                    features = get_features(batch, config['static_features'], config['use_rnn'], rnn_sess, rnn)

                with classifier_graph.as_default():
                    feed_dict = {valid_classifier.inputs: features, valid_classifier.targets: batch.ending_labels }
                    fetches = {'loss': valid_classifier.loss, 'accuracy': valid_classifier.accuracy, 'accuracy_op': valid_classifier.accuracy_op}
                    val_output = sess.run(fetches, feed_dict)
                    accur = sess.run([valid_classifier.accuracy], feed_dict)[0]
                    valid_loss += val_output['loss'] * batch.batch_size
                    valid_accur += accur * batch.batch_size

            time_delta = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            print('\rEpoch: {:3d}, time: {:s}; avg training loss: {:f}'.format(
                e + 1, time_delta, train_loss / dataset_train.data_size), end='\n')
            print('\rEpoch: {:3d}, time: {:s} \t --- validation loss: {:f} --- '.format(
                e + 1, time_delta, valid_loss / dataset_val.data_size), end='\n')
            print('\rEpoch: {:3d}, time: {:s} \t --- validation accuracy: {:f} --- '.format(
                e + 1, time_delta, accur / dataset_val.data_size), end='\n')

            if (e + 1) % config['save_checkpoints_every_epoch'] == 0:
                with classifier_graph.as_default():
                    if not os.path.exists(config['model_dir'] + "-ep"+str(e+1)):
                        os.makedirs(config['model_dir'] + "-ep"+str(e+1))
                    ckpt_path = saver.save(sess, config['model_dir'] + "-ep"+str(e+1), global_step=e+1)
                    print("Model saved to: "+ ckpt_path)

                    #with classifier_graph.as_default():
                    #    sess.run(epoch_counter_op)



    finally:
        with classifier_graph.as_default():
            print("\nFinishing...")
            ckpt_path = saver.save(sess, config['model_dir'], global_step=e+1)
            print("... model saved to: "+ ckpt_path )
        sess.close()
        if config['use_rnn']:
            rnn_sess.close()


if __name__ == "__main__":
    from config_full import config, valid_config

    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])
    main(config, valid_config)
