'''
Combine RNN and static features and build a classifier
on top of them.
'''
import tensorflow as tf


class SimpleEndingClassifier():

    def __init__(self, config):
        self.config = config


    def build_model(self):
        pass
        # For each batch of data, do:
            # Extract two last-sentence static features
            # Extract two full-sentence static features
            # Extract two last-sentence RNN-probabs p(ending)
            # Extract two full-sentence RNN-probabs p(ending, beginning)
            # Calculate p(ending | beginning) for both endings
            # Set a classifier on top

            # access points: self.classification :
            #   [p1, p2] giving the two probabilities that ending 1 or ending 2 is correct

    def build_loss(self):
        '''
        ? Use cross entropy loss between predicted probabilities which of both endings it is
        '''