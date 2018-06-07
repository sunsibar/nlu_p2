
import os
import sys
import shutil
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat


from train import set_up_datasets, get_rnn_model
from full_model import get_RNN_features




def get_perplexities_right_wrong(config, rnn_config, output_dir):

    dataset_train, dataset_val = set_up_datasets(config)

    # Load trained RNN model weights

    with tf.Session() as sess:
        ckpt_id = config['rnn_model_id']
        if ckpt_id is None:
            ckpt_path = tf.train.latest_checkpoint(config['rnn_model_dir'])
        else:
            ckpt_path = os.path.join(os.path.abspath(config['rnn_model_dir']), ckpt_id)

        rnn_model_cls = get_rnn_model(config['rnn_config'])
        rnn = rnn_model_cls(config['rnn_config'])
        saver_rnn = tf.train.Saver()#tf.train.import_meta_graph(ckpt_path + ".meta")
        saver_rnn.restore(sess, ckpt_path)
        print('Loaded RNN weights from ' + ckpt_path)


        sess.run(tf.global_variables_initializer())  # let's hope this initializes only one graph
        sess.run(tf.local_variables_initializer())


        right_ending_probabs = []
        wrong_ending_probabs = []
        cond_right_ending_probabs = []
        cond_wrong_ending_probabs = []
        right_ending_lengths = []
        wrong_ending_lengths = []
        story_lengths = []
        for batch in dataset_train.all_batches():       # TODO: log_features=True - better values
            features = get_RNN_features(sess, rnn, batch, log_rnn_features=True)
            r_e_p = [f[0] if  batch.ending_labels[i] == 0 else f[1] for i, f in enumerate(features) ]
            w_e_p = [f[1] if  batch.ending_labels[i] == 0 else f[0] for i, f in enumerate(features) ]
            cr_e_p = [f[2] if  batch.ending_labels[i] == 0 else f[3] for i, f in enumerate(features) ]
            cw_e_p = [f[3] if  batch.ending_labels[i] == 0 else f[2] for i, f in enumerate(features) ]
            r_e_l = [len(story[ 4 + batch.ending_labels[i] ])   for i, story in enumerate(batch.story_ids) ]
            w_e_l = [len(story[ 5 - batch.ending_labels[i] ])   for i, story in enumerate(batch.story_ids) ]
            s_l = [len(story[ : 4])   for i, story in enumerate(batch.story_ids) ]
            right_ending_probabs.append(r_e_p)
            wrong_ending_probabs.append(w_e_p)
            cond_right_ending_probabs.append(cr_e_p)
            cond_wrong_ending_probabs.append(cw_e_p)
            right_ending_lengths.append(r_e_l)
            wrong_ending_lengths.append(w_e_l)
            story_lengths.append(s_l)


        for batch in dataset_val.all_batches():
            features = get_RNN_features(sess, rnn, batch, log_rnn_features=True)
            r_e_p = [f[0] if  batch.ending_labels[i] == 0 else f[1] for i, f in enumerate(features) ]
            w_e_p = [f[1] if  batch.ending_labels[i] == 0 else f[0] for i, f in enumerate(features) ]
            cr_e_p = [f[2] if  batch.ending_labels[i] == 0 else f[3] for i, f in enumerate(features) ]
            cw_e_p = [f[3] if  batch.ending_labels[i] == 0 else f[2] for i, f in enumerate(features) ]
            r_e_l = [len(story[ 4 + batch.ending_labels[i] ])   for i, story in enumerate(batch.story_ids) ]
            w_e_l = [len(story[ 5 - batch.ending_labels[i] ])   for i, story in enumerate(batch.story_ids) ]
            s_l = [len(story[ : 4])   for i, story in enumerate(batch.story_ids) ]
            right_ending_probabs.append(r_e_p)
            wrong_ending_probabs.append(w_e_p)
            cond_right_ending_probabs.append(cr_e_p)
            cond_wrong_ending_probabs.append(cw_e_p)
            right_ending_lengths.append(r_e_l)
            wrong_ending_lengths.append(w_e_l)
            story_lengths.append(s_l)

        right_ending_probabs = np.hstack([np.array(r) for r in right_ending_probabs])
        wrong_ending_probabs = np.hstack([np.array(r) for r in wrong_ending_probabs])
        cond_right_ending_probabs = np.hstack([np.array(r) for r in cond_right_ending_probabs])
        cond_wrong_ending_probabs = np.hstack([np.array(r) for r in cond_wrong_ending_probabs])
        right_ending_lengths = np.hstack([np.array(r) for r in right_ending_lengths])
        wrong_ending_lengths = np.hstack([np.array(r) for r in wrong_ending_lengths])
        story_lengths = np.hstack([np.array(r) for r in story_lengths])

        L = dataset_train.data_size + dataset_val.data_size
        assert L == len(right_ending_probabs) == len(wrong_ending_probabs) \
               == len(cond_right_ending_probabs) == len(cond_wrong_ending_probabs)  \
               == len(story_lengths) == len(right_ending_lengths) == len(wrong_ending_lengths)


        return right_ending_probabs, wrong_ending_probabs, \
               cond_right_ending_probabs, cond_wrong_ending_probabs, \
               right_ending_lengths, wrong_ending_lengths, story_lengths



def mean(data):
    """Return the sample arithmetic mean of data."""
    n = len(data)
    if n < 1:
        raise ValueError('mean requires at least one data point')
    return sum(data)/n # in Python 2 use sum(data)/float(n)

def _ss(data):
    """Return sum of square deviations of sequence data."""
    c = mean(data)
    ss = sum((x-c)**2 for x in data)
    return ss

def stddev(data, ddof=0):
    """Calculates the population standard deviation
    by default; specify ddof=1 to compute the sample
    standard deviation."""
    n = len(data)
    if n < 2:
        raise ValueError('variance requires at least two data points')
    ss = _ss(data)
    pvar = ss/(n-ddof)
    return pvar**0.5


def create_perp_plots(r_p, w_p, c_r_p, c_w_p, r_e_l, w_e_l, s_l, output_dir):


    r_p_mean = np.mean(r_p)
    w_p_mean = np.mean(w_p)
    c_r_p_mean =  np.mean(c_r_p)
    c_w_p_mean = np.mean(c_w_p)
    r_p_std = stddev(r_p.astype(np.float64))
    w_p_std = stddev(w_p.astype(np.float64))
    c_r_p_std =  stddev(c_r_p.astype(np.float128))
    c_w_p_std = stddev(c_w_p.astype(np.float128))

    timestamp = datetime.datetime.now().strftime("%y-%b-%d_%Hh%M-%S")
    f = plt.figure()
    plt.errorbar(['log p(real ending)', 'log p(wrong ending)'], [r_p_mean, w_p_mean],
                 [r_p_std, w_p_std], linestyle='None', marker='^', elinewidth=0.5)
    #plt.bar(['p(real ending)', 'p(wrong ending)'],
    #        [r_p_mean, w_p_mean])
    f.savefig(os.path.join(output_dir, "ending_probabs"+timestamp+".png"))
    plt.show()
    f = plt.figure()
    plt.errorbar([ 'log p(real | sentence)', 'log p(wrong | sentence'], [ c_r_p_mean, c_w_p_mean],
                 [c_r_p_std, c_w_p_std],elinewidth=0.5, linestyle='None', marker='^')
    f.savefig(os.path.join(output_dir, "conditional_ending_probabs"+timestamp+".png"))
    plt.show()

    # get where the probabs of the true ending are higher
    right_higher = np.sum(r_p > w_p)
    right_lower = np.sum(r_p < w_p)
    right_c_higher = np.sum(c_r_p > c_w_p)
    right_c_lower = np.sum(c_r_p < c_w_p)

    print("number of times where right ending probab was higher:")
    print( str(right_higher) + " / " + str(len(w_p)))
    print("number of times where wrong ending probab was higher:")
    print( str(right_lower) + " / " + str(len(w_p)))
    print("number of times where conditional right ending probab was higher:")
    print( str(right_c_higher) + " / " + str(len(w_p)))
    print("number of times where conditional wrong ending probab was higher:")
    print( str(right_c_lower) + " / " + str(len(w_p)))


    print("so much plot")


if __name__ == '__main__':

    from config_full import config
    rnn_config = config['rnn_config']
    static_features = config['static_features']
    static_features['sentence_lengths'] = False
    static_features['sentiment'] = False
    config['static_features'] = static_features

    output_dir = '../analysis/perplexity/'
    timestamp = datetime.datetime.now().strftime("%y-%b-%d_%Hh%M-%S")  # str(int(time.time()))
    output_dir = os.path.abspath(
        os.path.join(output_dir, timestamp))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("write to {}\n".format(output_dir), flush=True)

    shutil.copy2("config_full.py", os.path.join(output_dir, 'theconfigforanalysis.txt'))

    r_p, w_p, c_r_p, c_w_p, r_e_l, w_e_l, s_l = get_perplexities_right_wrong(config, rnn_config, output_dir)

    create_perp_plots(r_p, w_p, c_r_p, c_w_p, r_e_l, w_e_l, s_l, output_dir)