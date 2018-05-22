import numpy as np
import pandas as pd
import re
from utils import build_dict, convert_text_data


# The classes StoryDataset and StoryFeeder are basically one thing together (could merge them some time)


class StoryDataset:
    '''
    self.stories: A list of lists(5 sentences) of word lists
    self.story_ids: A list of lists of numpy arrays of shape (sent_length x vocab_size)
    '''
    def __init__(self, stories=None, story_ids=None):
        self.stories = stories
        self.story_ids = story_ids
        self.story_keys = None   # metadata
        self.story_titles = None # metadata
        self.feeder = None # Could have integrated Feeder into this class, but why bother now

    def preprocess(self, preprocessor):
        self.story_ids = preprocessor.preprocess(self.stories)

    @property
    def data_size(self):
        return 0 if self.stories is None else len(self.stories)

    def next_batch(self, shuffle=True):
        assert self.feeder is not None
        return self.feeder.get_batch(shuffle=shuffle)

    def all_batches(self, shuffle=True):
        assert self.feeder is not None
        return self.feeder.all_batches(shuffle=shuffle)

    def get_data(self, indices):
        ''' :param indices: a list of integers in range(0, data_size)
            :return: the corresponding batch'''
        assert np.all(0 <= indices) and np.all(indices < self.data_size)
        return [self.stories[i] for i in indices]



def storydata_from_csv(path, batch_size, val_part=0.1):
    ''' batch_size: For creating the feeder for the training data part
        val_part: how much of the data should be set as validation set'''
    ds_train = StoryDataset()
    ds_val = StoryDataset()
    stories_df = pd.read_csv(path, engine='python')
    n_data = len(stories_df)
    n_val = int(val_part*float(n_data))

    # read in data
    ds_train.story_ids = stories_df.iloc[:,0].values[ : -n_val]
    ds_val.story_ids = stories_df.iloc[:,0].values[ : -n_val]
    ds_train.story_titles = stories_df.iloc[:,1].values[ -n_val : ]
    ds_val.story_titles = stories_df.iloc[:,1].values[ -n_val : ]
    story_mat = stories_df.iloc[:, 2:].values
    all_stories = []

    for story in story_mat:
        sentences = []
        for sentence in story:
            token = sentence.lower()
            token = re.sub(r"[?!.,';\":]", r" \g<0> ", token) # pad special signs with spaces
            token = token.strip().split(' ')
            token = filter(None, token)
            sentences.append(list(token))
        all_stories.append(sentences)
    ds_train.stories = all_stories[ : -n_val]
    ds_val.stories   = all_stories[-n_val : ]

    ds_train.feeder = StoryFeeder(ds_train, batch_size)
    ds_val.feeder = StoryFeeder(ds_val, ds_val.data_size)

    print("Loaded "+str(ds_train.data_size)+" training stories and " +str(ds_val.data_size)+
          " validation stories from "+path)
    return ds_train, ds_val


class StoryFeeder:

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rngen = np.random

        self.n_batches = int(np.ceil(float(dataset.data_size) / float(batch_size)))
        self.batch_ptr = 0
        self.ids_shuffled = np.arange(dataset.data_size)
        self.ids = np.arange(dataset.data_size)
        self.rngen.shuffle(self.ids_shuffled)

    def get_batch(self, shuffle=True):
        all_ids = self.ids_shuffled if shuffle else self.ids
        ptr = self.batch_ptr * self.batch_size
        if self.batch_ptr == self.n_batches - 1:    #if there's not enough data left for a full batch
            ids = all_ids[ptr : ]
        else:
            ids = all_ids[ptr : ptr + self.batch_size]
        return StoryBatch(stories=self.dataset.get_data(ids))

    def adv_batchptr(self):
        self.batch_ptr = self.batch_ptr + 1
        if self.batch_ptr == self.batch_size:
            self.batch_ptr = 0
            self.reshuffle()

    def next_batch(self, shuffle=True):
        batch = self.get_batch(shuffle=shuffle)
        self.adv_batchptr()
        return batch

    def all_batches(self, shuffle=True):
        ''' Yields whole dataset once, in batches'''
        for i in range(self.n_batches):
            yield self.next_batch(shuffle=shuffle)


    def reshuffle(self):
        self.rngen.shuffle(self.ids_shuffled)


class StoryBatch():
    def __init__(self, stories=None):
        self._stories = stories
        self._mask = None
        self._seq_lengths = None # Lengths of every entire sequence, when concatenating all sentences of a story

   # def mask(self):
   ##     """
   #     The mask is used to identify which entries in this minibatch are "real" values and which ones are "padded".
   #     The mask has shape (batch_size, max_seq_length) where an entry is 1 if it's a real value and 0 if it was padded.
   #     """
   #     if self._mask is None:
   #         max_seq_length = max(self.seq_lengths)
   #         ltri = np.tril(np.ones([max_seq_length, max_seq_length]))
   #         self._mask = ltri[self.seq_lengths - 1]
   #     return self._mask

    @property
    def seq_lengths(self):
        if self._seq_lengths is None:
            self._seq_lengths = np.array([np.sum([sent.shape[0] for sent in story]) for story in self.stories ])  # list of sequence lengths per batch entry
        return self._seq_lengths

    @property
    def stories(self):
        return self._stories

    def get_padded_data(self, which_sentences=[0,1,2,3,4], pad_target=True):
        """
        Pads the data with zeros, i.e. returns an np array of shape (batch_size, max_seq_length, dof). `max_seq_length`
        is the maximum occurring sequence length in the batch. Target is only padded if `pad_target` is True.
        :returns an 'input' consisting of the sentences given in :param which_sentences, concatenated to one list
                 a 'target' which is the same sentence-group, but shifted by one

        """
        cur_seq_lengths = np.array([np.sum([story[i].shape[0] for i in which_sentences]) for story in self.stories])
        max_seq_length = max(cur_seq_lengths)

        inputs = []
        targets = []
        for x in self._stories:
            X = np.concatenate([x[i] for i in which_sentences])[ : -1]
            Y = np.concatenate([x[i] for i in which_sentences])[ 1 : ]
            missing = max_seq_length - X.shape[0]
            x_padded, y_padded = X, Y
            if missing > 0:
                # this batch entry has a smaller sequence length then the max sequence length, so pad it with zeros
                voc_len = X.shape[1]
                x_padded = np.concatenate([X, np.zeros(shape=[missing, voc_len])], axis=0)
                if pad_target:
                    y_padded = np.concatenate([Y, np.zeros(shape=[missing, voc_len])], axis=0)
            assert len(x_padded) == max_seq_length
            inputs.append(x_padded)

            if pad_target:
                assert len(y_padded) == max_seq_length
                targets.append(y_padded)

        ltri = np.tril(np.ones([max_seq_length, max_seq_length]))
        mask = ltri[self.seq_lengths - 1]

        return np.array(inputs), np.array(targets), mask # TODO: check shape of these arrays


class Preprocessor():
    def __init__(self, config, dataset=None):
        self.max_sentence_length = config['max_sentence_length']
        self.vocab_size = config['vocab_size']
        self.word2id_dict = None
        self.id2word_dict = None

        if dataset is not None:
            self.set_up(dataset)

    def set_up(self, dataset):
        flat_list = [item for sublist in dataset.stories for item in sublist]
        self.word2id_dict, self.id2word_dict = build_dict(flat_list, vocab_len=self.vocab_size)



    def preprocess(self, storylist):
        '''storylist: A list of lists of word-lists'''
        story_ids = []
        for story in storylist:
            prep_story_vec, seq_l = convert_text_data(story, self.word2id_dict, )
            story_ids.append(prep_story_vec)  #-> list of lists of numpy-arrays, 1d, with word indices
        return story_ids