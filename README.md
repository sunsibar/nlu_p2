# NLU-p2

Supposed to become an implementation of http://www.aclweb.org/anthology/W17-0907, and then be extended.

full_model.py is for models that supervisedly learn to predict which is the true story ending, from possibly rnn features as well as static features. train.py is for training these models. Static features are only sentence lengths.


There are files for three different tasks:

1. Training the RNN model:
  - RNN_model.py
  - train_RNN.py
  - config.py: Contains hyperparameters and other configurations
  
 2. Training the logistic classifier on RNN-features and/or static features:
  - full_model.py
  - train.py
  - config_full.py
  
 3. Analysing the RNN features:
  - analysis.py: Creates some plots
  - config_full.py: This configuration file will be used to load the trained RNN for analysis

And then there are:
  - dataset.py: Contains data structures used in the remaining code
  - utils.py
 
# To train the RNN:
- Adjust config.py if you want
- Run  python train_RNN.py
- Wait for the progress to be printed to stdout

# To train the full model
- Adjust config_full.py, for example, set the path to the trained RNN model to be used; set the rnn_config to the same as what it was during training of the RNN
- Run python train.py
- Wait for the progress to be printed to stdout

# To analyze a trained RNN:
- Adjust config_full.py, for example, set the path to the trained RNN model to be used; set the rnn_config to the same as what it was during training of the RNN
- Run analysis.py
- Stop the code in debug mode and look more closely at the variables, or just let it run and look at the created plots & output


