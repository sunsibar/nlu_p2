
# ---- RNN configuration parameters ---- #
rnn_config = {}
rnn_config['data_dir'] = '../data'
rnn_config['output_dir'] = '../trained_models/RNN'

#rnn_config["learning_rate"] = 1e-3
rnn_config["batch_size"] = 32
#rnn_config["num_epochs"] = 10
#rnn_config['max_grad_norm'] = 5

#rnn_config['save_checkpoints_every_epoch'] = 5
#rnn_config['n_keep_checkpoints'] = 2

rnn_config['model_type'] = 'simple' # RNN model type; one of: 'simple', ... (to come)

rnn_config['mode'] = 'infer_RNN' # one of 'train_RNN', 'validate_RNN', 'infer_RNN'

if rnn_config['model_type'] == 'simple':
    rnn_config['hidden_size'] = 100
    rnn_config['num_layers'] = 3
    rnn_config['no_dropout_ids'] = [0] # unused # don't use dropout between input and first layer

    rnn_config['embedding_dim'] = 100
    rnn_config['is_use_embedding'] = True
    rnn_config['embedding_path'] = "./embs/wordembeddings-dim100.word2vec"
    rnn_config['is_add_layer'] = False

    rnn_config['name'] = rnn_config['model_type'] + "-" + str(rnn_config['num_layers']) + "L-" + str(
        rnn_config['hidden_size']) + "h_useE-"+str(rnn_config['is_use_embedding'])+"_addL-"+str(rnn_config['is_add_layer'])

else:
    raise ValueError("Error, unknown model type: "+rnn_config['model_type'])

assert 'model_dir' not in rnn_config.keys() # will be written at train time and appended with a date

# ---- toggle static features ---- #
static_features = {}
static_features['sentence_lengths'] = True
static_features['sentiment'] = False # TODO
# TODO: if you add new features, also add lines to the feature counting 'num_features(self)' in full_model.py

# ---- Full model configuration parameters ---- #
config = {}
config['model_type'] = "simple"
config['mode'] = 'training' # one of 'training', 'validation', 'inference'
config['use_rnn'] = True
config["learning_rate"] = 1e-2
config["batch_size"] = rnn_config['batch_size']
config["num_epochs"] = 10
config['max_grad_norm'] = 5

config['save_checkpoints_every_epoch'] = 5
config['n_checkpoints_to_keep'] = 2
config['data_dir'] = '../data'
config['output_dir'] = '../trained_models/full'
config['train_data_file'] = 'train_stories.csv'
#config['train_data_file'] = 'train_stories_sample.csv'
config['story_cloze_file'] = 'cloze_test_val__spring2016.csv'
#config['story_cloze_file'] = 'cloze_test_val_sample.csv'
config['rnn_config'] = rnn_config
config['static_features'] = static_features
#config['max_sentence_length'] = 30 # in words, including special tokens and sentence endings ## TODO: unused so far...
config['vocab_size'] = rnn_config['vocab_size'] = 20000
#config['limit_num_samples'] = 20   # Unused ; None, or a positive number to reduce the number of samples during training

# In training the final classifier, needs an RNN model that's already been trained
config['rnn_model_dir'] = '../trained_models/RNN/simple-3L-100h_useE-True_addL-False/18-May-26_00h21-19' # add path to model here
config['rnn_model_id'] = None # None, then use latest checkpoint, or add the checkpoint ID here
config['name'] = config['model_type']
if config['use_rnn']:
    config['name'] += "-" + rnn_config['model_type'] + "_rnn"
else:
    config['name'] += "no_rnn"

assert 'model_dir' not in config.keys() # will be created during training

valid_config = config.copy()
valid_config['mode'] = 'validation'

# For using an existing full model:
infer_config = config.copy()
infer_config['model_dir'] = '../trained_models/full/myfancymodel' # add path to model here
infer_config['model_id'] = None # None, then use latest checkpoint, or add the checkpoint ID here
