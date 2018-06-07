
# ---- RNN configuration parameters ---- #
rnn_config = {}
rnn_config['data_dir'] = '../data'
rnn_config['output_dir'] = '../trained_models/RNN'

rnn_config["learning_rate"] = 1e-3
rnn_config["batch_size"] = 32
rnn_config["num_epochs"] = 100
rnn_config['max_grad_norm'] = 5

rnn_config['save_checkpoints_every_epoch'] = 5
rnn_config['n_checkpoints_to_keep'] = 2

rnn_config['model_type'] = 'simple' # RNN model type; one of: 'simple', ... (to come)

rnn_config['mode'] = 'train_RNN' # one of 'train_RNN', 'validate_RNN', 'infer_RNN'

if rnn_config['model_type'] == 'simple':
    rnn_config['hidden_size'] = 100
    #rnn_config['num_layers'] = 3
    #rnn_config['no_dropout_ids'] = [0] # unused # don't use dropout between input and first layer

    rnn_config['embedding_dim'] = 100
    rnn_config['is_use_embedding'] = True
    rnn_config['embedding_path'] = "./embs/wordembeddings-dim100.word2vec"
    rnn_config['is_add_layer'] = False
    rnn_config['is_dropout'] = False

    rnn_config['name'] = rnn_config['model_type'] + "-" + "L-" + str(
        rnn_config['hidden_size']) + "h_useE-"+str(rnn_config['is_use_embedding'])+"_addL-"+str(rnn_config['is_add_layer'])


else:
    raise ValueError("Error, unknown model type: "+rnn_config['model_type'])

assert 'model_dir' not in rnn_config.keys() # will be written at train time and appended with a date

# ---- Full model configuration parameters ---- #
config = {}
config['model_type'] = "simple"
config['mode'] = 'training' # one of 'training', 'validation', 'inference'
config['use_rnn'] = True
config["learning_rate"] = 1e-2
config["batch_size"] = rnn_config['batch_size']
config["num_epochs"] = 100
config['max_grad_norm'] = 5

rnn_config['save_checkpoints_every_epoch'] = 5
rnn_config['n_checkpoints_to_keep'] = 2
config['data_dir'] = '../data'
#config['output_dir'] = '../trained_models/full'
config['train_data_file'] = 'train_stories.csv'
config['story_cloze_file'] = 'cloze_test_val__spring2016.csv'
config['rnn_config'] = rnn_config
config['vocab_size'] = rnn_config['vocab_size'] = 20000

# In training the final classifier, needs an RNN model that's already been trained
config['name'] = config['model_type']
if config['use_rnn']:
    config['name'] += "-" + rnn_config['model_type'] + "_rnn"
else:
    config['name'] += "no_rnn"

assert 'model_dir' not in config.keys() # will be created during training

