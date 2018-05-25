import tensorflow as tf


class RNNModel():
    def __init__(self, rnn_config):

        self.config = rnn_config
        self.embedding_dim = rnn_config['embedding_dim']
        #self.sequence_length = rnn_config['sequence_length']
        self.hidden_size = rnn_config['hidden_size']
        self.vocab_size = rnn_config['vocab_size']
            # Isnt't this missing a dimension?
        self.input_x = tf.placeholder(dtype=tf.int64, shape=[None, None], name="input_x") #self.sequence_length], name="input_x")
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None, None], name="input_y")#self.sequence_length], name="input_y")
        self.sequence_length_list = tf.placeholder(dtype=tf.int32, shape=[None,],name='sequence_length_list')
        self.batch_size = tf.shape(self.input_x)[0]  # dynamic size#rnn_config['batch_size']
        self.max_seq_length = tf.shape(self.input_x)[1]  # dynamic size
        self.sequence_mask = tf.sequence_mask(self.sequence_length_list, None, dtype=tf.float32)

        self.word_embeddings = tf.get_variable("word_embeddings", [self.vocab_size, self.embedding_dim])
        self.embedded_tokens = tf.nn.embedding_lookup(self.word_embeddings, self.input_x)

        # split by the timestamp # Todo: Understand & try to remove sequence_length here --
        # TODO: self.embedded_tokens = tf.unstack(self.embedded_tokens, num=self.max_seq_length, axis=1)# num=self.sequence_length, axis=1)
        # rnncell = tf.nn.rnn_cell.BasicRNNCell(num_units=self.hidden_size)
        with tf.variable_scope("rnn"):
            if rnn_config['is_add_layer']:
                rnncell = tf.nn.rnn_cell.LSTMCell(num_units=2*self.hidden_size)
                W_middle = tf.get_variable("W_middle", shape=[2 * self.hidden_size, self.hidden_size],
                                           initializer=tf.contrib.layers.xavier_initializer())
            else:
                rnncell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size)
            state = rnncell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            outputs, state = tf.nn.dynamic_rnn(rnncell, self.embedded_tokens, sequence_length=self.sequence_length_list,
                                               initial_state=state)
            # TODO: test the above

        if rnn_config['is_add_layer']:
            outputs = tf.reshape(outputs,[-1,2*self.hidden_size])
            self.outputs = tf.matmul(outputs,W_middle)
        else:
            self.outputs = tf.reshape(outputs,[-1,self.hidden_size])  # shape: (batch_size*time_step, self.hidden_size)
        self.W_out = tf.get_variable("W_out", shape=[self.hidden_size, self.vocab_size],
                                initializer=tf.contrib.layers.xavier_initializer())
        self.b_out = tf.Variable(tf.constant(0.1, shape=[self.vocab_size,]), name='b_out')

        logits = tf.nn.xw_plus_b(self.outputs,self.W_out, self.b_out)   # TODO: check the replacement works (of self.seq_length -> self.max_seq_length
        logits = tf.reshape(logits, shape=[self.max_seq_length, -1, self.vocab_size])  # (time_step,batch_size,vocab_size)
        self.logits = tf.transpose(logits, perm=[1, 0, 2])
        self.prediction = tf.argmax(logits, 1, name='prediction')
        self.loss = tf.contrib.seq2seq.sequence_loss(
                            self.logits,
                            self.input_y,
                            self.sequence_mask,
                            average_across_timesteps=True,
                            average_across_batch=False,name="loss")

        self.eva_perplexity = tf.exp(self.loss, name="eva_perplexity")
        self.minimize_loss = tf.reduce_mean(self.loss,name="minize_loss")
        self.print_perplexity = tf.reduce_mean(self.eva_perplexity, name="print_perplexity")

        self.sequence_probab = self.eva_perplexity