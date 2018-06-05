'''
Copied (possibly modified) from model.py from project 1
Train the RNN to reuse it in full model
'''

# Neural Language Model
import numpy as np
import tensorflow as tf
import gensim
import nltk
import time
import os
import glob
import sys
import datetime
from utils import id2word, load_data, build_dict, convert_text_data, get_batch_data
from RNN_model import RNNModel
from dataset import StoryDataset, storydata_from_csv, Preprocessor

def train(model_train, model_val, rnn_config, train_dataset, val_dataset, id2word_dict):
    #assert val_dataset.data_size == val_dataset.feeder.batch_size

    learning_rate = rnn_config['learning_rate']
    is_use_embedding = rnn_config['is_use_embedding']
    is_add_layer = rnn_config['is_add_layer']
    embedding_path = rnn_config['embedding_path']
    vocab_len = rnn_config['vocab_size']
    max_grad_norm = rnn_config['max_grad_norm']
    embedding_dim = rnn_config['embedding_dim']
    num_epochs = rnn_config['num_epochs']

    # set the output dir
    timestamp = datetime.datetime.now().strftime("%y-%b-%d_%Hh%M-%S")  # str(int(time.time()))
    rnn_config['model_dir'] = os.path.abspath(os.path.join(rnn_config['output_dir'], rnn_config['name'] + '/' + timestamp))
    print("write to {}\n".format(rnn_config['model_dir']), flush=True)

    # Training
    sess = tf.Session()
    with sess.as_default():
        # define the training process
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(model_train.minimize_loss, tvars), max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.apply_gradients(zip(grads, tvars),
            global_step=global_step)
        sess.run(tf.global_variables_initializer())

        #  add summary
        grad_summaries = []
        for g, v in zip(grads,tvars):
            if g is not None:
                grad_hist_summary = tf.summary.histogram('/grad/hist/%s' % v.name, g)
                grad_summaries.append(grad_hist_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)


        # summary for the loss
        loss_summary = tf.summary.scalar("print_perplexity", model_train.print_perplexity)

        # train_summary
        train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
        out_summary_dir = os.path.join(rnn_config['model_dir'], "summary")
        train_summary_writer = tf.summary.FileWriter(out_summary_dir,sess.graph)

        # saver
        checkpoint_dir = os.path.join(rnn_config['model_dir'],"checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=rnn_config['n_checkpoints_to_keep'])
        with open(os.path.join(rnn_config['model_dir'],'config.txt'),'w') as f:
            for item in rnn_config.items():
                f.write(str(item))
                f.write('\n')
        # load embedding
        wordemb = gensim.models.KeyedVectors.load_word2vec_format(embedding_path,binary=False)
        my_embedding_matrix = np.random.uniform(-0.25, 0.25, (vocab_len,embedding_dim))
        if is_use_embedding:
            for id, word in id2word_dict.items():
                if word in wordemb.vocab:
                    my_embedding_matrix[id,:] = wordemb[word]
                else:
                    my_embedding_matrix[id,:] = np.random.uniform(-0.25, 0.25,embedding_dim)

        word_embedding = tf.placeholder(tf.float32,[None,None], name="pretrained_embeddings")
        set_x = model_train.word_embeddings.assign(word_embedding)
        set_x_val = model_val.word_embeddings.assign(word_embedding)

        sess.run(set_x, feed_dict={word_embedding:my_embedding_matrix})
        sess.run(set_x_val, feed_dict={word_embedding:my_embedding_matrix})


        try:
            epoch = 0
            while epoch < num_epochs:
                train_loss = 0
                for ib, ibatch in enumerate(train_dataset.all_batches(shuffle=True)):
                    #X, Y, batch_seq_lengths = ibatch.get_padded_data() # also todo: why are val and train loss so incredibly different?
                    feed_dict = model_train.get_feed_dict_train(ibatch)  # todo: why different than below?
                    _, summary, step, loss = sess.run([train_op, train_summary_op, global_step, model_train.print_perplexity],
                                                        feed_dict=feed_dict)
                    train_summary_writer.add_summary(summary=summary, global_step=step)
                    print('\ribatch {:d}, max loss {:f}, loss {:f}'.format(ib, np.max(loss), np.mean(loss)), end='')
                    train_loss += loss
                # -- validate --
                valid_loss = 0.
                for ib, ibatch in enumerate(val_dataset.all_batches()):
                #val_allbatch = val_dataset.next_batch(shuffle=False)
                    #X_v, Y_v, val_seq_lengths = ibatch.get_padded_data()
                    feed_dict = model_val.get_feed_dict_train(ibatch)
                    valid_loss += sess.run([model_val.print_perplexity], feed_dict=feed_dict)[0]#{model_val.input_x: X_v,
                                                                           #model_val.input_y: Y_v,
                                                                           #model_val.sequence_length_list: val_seq_lengths})[0]
                print("\nep:", epoch, "train_loss", train_loss / train_dataset.data_size, flush=True)
                print("ep:", epoch, "valid_loss", valid_loss / val_dataset.data_size, flush=True)
                sys.stdout.flush()
                if (epoch + 1) % rnn_config['save_checkpoints_every_epoch'] == 0:
                    ckpt_path = saver.save(sess, checkpoint_dir+"_ep"+str(epoch)+"/", global_step)
                    print('Model saved to file {}'.format(ckpt_path))
                epoch += 1
        finally:
            if not config['train_data_file'] == 'train_stories_sample.csv':
                ckpt_path = saver.save(sess,checkpoint_dir,global_step=step)
                print("Stored trained model at: \n"+ckpt_path)
            else:
                print("train data file indicates this is a debug run; not saving the model at shutdown.")
    return ckpt_path

def evaluate(sess_path, eva_data, result_ptr):
    data_x,data_y,length_list = eva_data
    n_batch = len(data_x)
    sess = tf.Session()
    graph_path = os.path.join(sess_path,'*.meta')
    graph_name = glob.glob(graph_path)
    saver = tf.train.import_meta_graph(graph_name[0])
    saver.restore(sess, graph_name[0].split('.')[0])
    graph = tf.get_default_graph()

    # get ops
    perplexity = graph.get_tensor_by_name("eva_perplexity:0")
    input_x = graph.get_tensor_by_name("input_x:0")
    input_y = graph.get_tensor_by_name("input_y:0")
    sequence_length_list = graph.get_tensor_by_name("sequence_length_list:0")

    for ibatch in range(n_batch):
        this_perplexity = sess.run(perplexity, feed_dict={input_x: data_x[ibatch,:,:],
                                                          input_y: data_y[ibatch,:,:],
                                                          sequence_length_list: length_list[ibatch,:]})
        print(np.mean(this_perplexity),np.max(this_perplexity),flush=True)
        for i_per in this_perplexity:
            result_ptr.write(str(i_per)+'\n')


def generate(sess_path, cfg, contin_data,result_ptr,id2word_dict):

    vocab_len = cfg['vocab_len']
    embedding_dim = cfg['embedding_dim']
    hidden_size = cfg['hidden_size']
    is_add_layer = cfg['is_add_layer']
    max_generate_length = cfg['max_generate_length']
    batch_size = cfg['batch_size']
    max_length = cfg['max_length']

    contin_data,_,length_list = contin_data
    # contin_data = np.squeeze(contin_data, axis=1)  #we process sentence one by one
    # length_list = np.squeeze(length_list, axis=1)
    sess = tf.Session()
    graph_path = os.path.join(sess_path,'*.meta')
    graph_name = glob.glob(graph_path)

    word_embeddings = tf.get_variable("word_embeddings", [vocab_len, embedding_dim])
    if is_add_layer:
        rnncell = tf.nn.rnn_cell.LSTMCell(num_units=2*hidden_size)
        W_middle = tf.get_variable("rnn/W_middle", shape=[2 * hidden_size, hidden_size])
    else:
        rnncell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

    state=rnncell.zero_state(batch_size=batch_size,dtype=tf.float32)
    init_input = tf.constant(0,shape=(batch_size,embedding_dim),dtype=tf.float32)
    with tf.variable_scope('rnn'):
        rnncell(init_input,state) # for create rnn kerner/ bias for parameter load

    W_out = tf.get_variable("W_out", shape=[hidden_size, vocab_len],
                                 initializer=tf.contrib.layers.xavier_initializer())
    b_out = tf.Variable(tf.constant(0.1, shape=[vocab_len, ]), name='b_out')

    saver = tf.train.Saver()
    ckpt_name = graph_name[0].split('.')[0]  # something like checkpoints-40
    saver.restore(sess,ckpt_name)
    sentence_id = 0
    for sentence_id_list,L in zip(contin_data,length_list):
        state = rnncell.zero_state(batch_size=batch_size, dtype=tf.float32)
        output_list = []
        state_list = []

        # lookup
        embedded_tokens = tf.nn.embedding_lookup(word_embeddings, sentence_id_list)
        # split by the timestamp
        embedded_tokens = tf.unstack(embedded_tokens, num=max_length, axis=1)

        l = 0
        L_max = np.max(L)
        for _input in embedded_tokens:
            output, state = rnncell(_input, state)
            output_list.append(output)
            state_list.append(state)
            l += 1
            if l == L_max:
                break

        for idx,iL in enumerate(L):
            generate_length = 0
            output = tf.gather_nd(output_list[iL-1], [[idx]])
            c_all,h_all = state_list[iL-1]
            c = tf.gather_nd(c_all, [[idx]])
            h = tf.gather_nd(h_all,[[idx]])
            state = (c,h)
            for i in range(iL, max_generate_length):
                if is_add_layer:
                    middle_output = tf.matmul(output,W_middle)
                    final_output = tf.add(tf.matmul(middle_output, W_out),b_out)
                    word_id = sess.run(tf.argmax(final_output,axis=1))
                else:
                    word_id = sess.run(tf.argmax(tf.add(tf.matmul(output, W_out),b_out),axis=1))
                sentence_id_list[idx][i]=word_id[0]
                generate_length += 1
                if word_id[0] == 3: # <eos>
                    break
                input = tf.nn.embedding_lookup(word_embeddings,word_id)
                output,state = rnncell(input,state)
            sentence = id2word(sentence_id_list[idx],id2word_dict)
            print (sentence_id, ' '.join(sentence[:iL+generate_length]),flush=True)
            result_ptr.write(' '.join(sentence[:iL+generate_length])+'\n')
            sentence_id += 1






# ------ Main ----- #

def main(config):
    train_data_file = os.path.join(config['data_dir'], config['train_data_file'])
    story_dataset_train, story_dataset_val = \
        storydata_from_csv(train_data_file, config['rnn_config']['batch_size'], has_titles=True, has_ending_labels=False)
    prep = Preprocessor(config, dataset=story_dataset_train)
    story_dataset_train.preprocess(prep)
    story_dataset_val.preprocess(prep)

    model_train = RNNModel(config['rnn_config'])
    val_rnn_config = config['rnn_config'].copy()
    val_rnn_config['mode'] = 'validate_RNN'
    model_val = RNNModel(val_rnn_config)

    out_dir = train(model_train, model_val, config['rnn_config'], id2word_dict=prep.id2word_dict,
                    train_dataset=story_dataset_train, val_dataset=story_dataset_val)

#    quiz_file = os.path.join(config['data_dir'], config['StoryCloze_file'])
#    story_dataset_quiz_train, story_dataset_quiz_val = \
#        storydata_from_csv(quiz_file, config['rnn_config']['batch_size'], has_titles=False, has_ending_labels=True)
#    story_dataset_quiz_train.preprocess(prep)
#    story_dataset_quiz_val.preprocess(prep)



if __name__ == '__main__':

    from config import config
    main(config)