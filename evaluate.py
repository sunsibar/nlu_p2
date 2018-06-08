import tensorflow as tf
import numpy as np
import os
from train import set_up_datasets, get_rnn_model








def main(config):


    # load data
    dataset_train, dataset_val = set_up_datasets(config)

    # Load rnn model

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

    ckpt_id = config['model_id']
    if ckpt_id is None:
        ckpt_path = tf.train.latest_checkpoint(config['model_dir'])
    else:
        ckpt_path = os.path.join(os.path.abspath(config['model_dir']), ckpt_id)

    with classifier_graph.as_default():
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(ckpt_path)

# No. This model works .. not at all, so no reason to evaluate on the story cloze test set.



    sess.close()
    print("done")
















if __name__ == '__main__':

    from config_full import infer_config
    main(infer_config)