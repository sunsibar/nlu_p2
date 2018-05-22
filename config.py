
# ---- RNN configuration parameters ---- #
rnn_config = {}
rnn_config['data_dir'] = '../data'
rnn_config['output_dir'] = '../trained_models/RNN'

rnn_config["learning_rate"] = 1e-3
rnn_config["batch_size"] = 32
rnn_config["num_epochs"] = 10
rnn_config['max_grad_norm'] = 5

rnn_config['save_checkpoints_every_epoch'] = 5
rnn_config['n_keep_checkpoints'] = 2

rnn_config['model_type'] = 'simple' # RNN model type; one of: 'simple', ... (to come)

rnn_config['mode'] = 'train_RNN' # one of 'train_RNN', 'validate_RNN', 'infer_RNN'

if rnn_config['model_type'] == 'simple':
    rnn_config['hidden_size'] = 100
    rnn_config['num_layers'] = 3
    rnn_config['no_dropout_ids'] = [0] # don't use dropout between input and first layer

    rnn_config['embedding_dim'] = 100
    rnn_config['is_use_embedding'] = True
    rnn_config['embedding_path'] = "../../embs/wordembeddings-dim100.word2vec"
    rnn_config['is_add_layer'] = False

    rnn_config['name'] = rnn_config['model_type'] + "-" + str(rnn_config['num_layers']) + "L-" + str(
        rnn_config['hidden_size']) + "h_useE-"+str(rnn_config['is_use_embedding'])+"_addL-"+str(rnn_config['is_add_layer'])


else:
    raise ValueError("Error, unknown model type: "+rnn_config['model_type'])

assert 'model_dir' not in rnn_config.keys() # will be written at train time and appended with a date

# ---- toggle static features ---- #

static_features = {}
static_features['sentence_lengths'] = True


# ---- Full model configuration parameters ---- #
config = {}
config['data_dir'] = '../data'
config['output_dir'] = '../trained_models/full'
config['mode'] = 'training' # one of 'training', 'validation', 'inference'
config['rnn_config'] = rnn_config
config['static_features'] = static_features
config['max_sentence_length'] = 30 # in words, including special tokens and sentence endings ## TODO: unused so far...
config['vocab_size'] = rnn_config['vocab_size'] = 20000
config['limit_num_samples'] = 20   # Unused ; None, or a positive number to reduce the number of samples during training

# In training the final classifier, needs an RNN model that's already been trained
config['rnn_model_dir'] = '../trained_models/RNN/myfancymodel' # add path to model here
config['rnn_model_id'] = None # None, then use latest checkpoint, or add the checkpoint ID here

assert 'model_dir' not in config.keys() # will be created during training

# For inference:
infer_config = config.copy()
infer_config['model_dir'] = '../trained_models/full/myfancymodel' # add path to model here
infer_config['model_id'] = None # None, then use latest checkpoint, or add the checkpoint ID here
