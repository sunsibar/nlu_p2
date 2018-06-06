import numpy as np
import pandas as pd
import re
from utils import build_dict, convert_text_data
import nltk

# The classes StoryDataset and StoryFeeder are basically one thing together (could merge them some time)

#
# ! For the training set, the stories are always lists of five sentences (in the respective format).
# ! For the validation set, they are lists of six sentences, and an attribute 'wrong_endings' specifies
# ! whether the fifth or the sixth sentence is the correct last sentence.
#

class StoryDataset:
    '''
    self.stories: A list of lists(5 sentences, or 6 if there are wrong endings) of word lists
    self.story_ids: A list of lists of numpy arrays of shape (sent_length x vocab_size)
    '''
    def __init__(self, stories=None, story_ids=None):
        self.stories = stories
        self.story_ids = story_ids
        self.story_keys = None   # metadata
        self.story_titles = None # metadata
        self.ending_labels = None # says whether the correct ending is the fifth sentence ("1") or the sixth ("2")
        self.feeder = None # Could have integrated Feeder into this class, but why bother now

    def preprocess(self, preprocessor):
        self.story_ids = preprocessor.preprocess(self.stories)

    @property
    def data_size(self):
        return 0 if self.stories is None else len(self.stories)

    def next_batch(self, shuffle=True):
        assert self.feeder is not None
        return self.feeder.next_batch(shuffle=shuffle)

    def all_batches(self, shuffle=True):
        assert self.feeder is not None
        return self.feeder.all_batches(shuffle=shuffle)

    def get_batch(self, shuffle=True, return_raw=True):
        assert self.feeder is not None
        return self.feeder.get_batch(shuffle=shuffle, return_raw=return_raw)

    def get_data(self, indices, id=True):
        ''' :param indices: a list of integers in range(0, data_size)
            :param id:  if True, returned data consists of word indices,
                        if False, return one list of sentences per story index
            :return: the corresponding batch'''
        assert np.all(0 <= indices) and np.all(indices < self.data_size)
        if id:
            return [self.story_ids[i] for i in indices]
        else:
            return [self.stories[i] for i in indices]

    @property
    def n_batches(self):
        assert self.feeder is not None
        return self.feeder.n_batches



def storydata_from_csv(path, batch_size, val_part=0.1, has_titles=True, has_ending_labels=False):
    ''' batch_size: For creating the feeder for the training data part
        val_part: how much of the data should be set as validation set
        has_titles: if True, treats the **second field** as titles
        has_ending_labels: if True, expects integer labels in the **last field**'''
    ds_train = StoryDataset()
    ds_val = StoryDataset()
    stories_df = pd.read_csv(path, engine='python')
    n_data = len(stories_df)
    n_val = int(val_part*float(n_data))

    # read in data
    ds_train.story_ids = stories_df.iloc[:,0].values[ : -n_val]
    ds_val.story_ids = stories_df.iloc[:,0].values[ -n_val : ]
    if has_titles:
        ds_train.story_titles = stories_df.iloc[:,1].values[ : -n_val ]
        ds_val.story_titles   = stories_df.iloc[:,1].values[ -n_val : ]
        stories_df.drop(stories_df.columns[1], axis=1, inplace=True)
    if has_ending_labels:
        ds_train.ending_labels = stories_df.iloc[:,-1].values[ : -n_val ] - 1
        ds_val.ending_labels   = stories_df.iloc[:,-1].values[ -n_val : ] - 1
        stories_df.drop(stories_df.columns[-1], axis=1, inplace=True)
    story_mat = stories_df.iloc[:, 1:].values
    all_stories = []

    for story in story_mat:
        sentences = []
        for sentence in story:
            # token = sentence.lower()
            # token = re.sub(r"[?!.,';\":]", r" \g<0> ", token) # pad special signs with spaces
            # token = token.strip().split(' ')
            # token = filter(None, token)
            # sentences.append(list(token))
            sentences.append(nltk.word_tokenize(sentence))
        all_stories.append(sentences)
    ds_train.stories = all_stories[ : -n_val]
    ds_val.stories   = all_stories[-n_val : ]

    ds_train.feeder = StoryFeeder(ds_train, batch_size)
    ds_val.feeder = StoryFeeder(ds_val, batch_size) # just use the same size

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

    def get_batch(self, shuffle=True, return_raw=True):
        ''' The returned batch contains the word indices of the current batch, one list per sentence per story.
            batch contains: word indices, opt: raw sentences, opt: wrong story endings, opt: raw wrong story endings
            :param return_raw: if True, add lists of raw sentences to output batch
            '''
        all_ids = self.ids_shuffled if shuffle else self.ids
        ptr = self.batch_ptr * self.batch_size
        if self.batch_ptr == self.n_batches - 1:    #if there's not enough data left for a full batch
            ids = all_ids[ptr : ]
        else:
            ids = all_ids[ptr : ptr + self.batch_size]
        stories = None
        ending_labels = None if self.dataset.ending_labels is None else self.dataset.ending_labels[ids]
        if return_raw:
            stories = self.dataset.get_data(ids, id=False)
        return StoryBatch(story_ids=self.dataset.get_data(ids), stories=stories, ending_labels=ending_labels)

    def adv_batchptr(self):
        self.batch_ptr = self.batch_ptr + 1
        if self.batch_ptr == self.n_batches:
            self.batch_ptr = 0
            self.reshuffle()
        assert self.batch_ptr < self.n_batches

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
    def __init__(self, story_ids=None, stories=None, ending_labels=None):
        self._story_ids = story_ids
        self._stories = stories
        self._ending_labels = ending_labels # a batch-size-sized list of '1's and '2's, indicating whether the fifth or the sixth sentence is the correct ending
        self._mask = None
        self._seq_lengths = None # Lengths of every entire sequence, when concatenating all sentences of a story
        self._target_ids = None


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
            self._seq_lengths = np.array([np.sum([len(sent) for sent in story]) for story in self.story_ids ])  # list of sequence lengths per batch entry
        return self._seq_lengths

    def get_seq_lengths(self, which_sentences):
        seq_ls = np.array([np.sum([len(story[idx]) for idx in which_sentences]) for story in self.story_ids ])
        return seq_ls

    @property
    def batch_size(self):
        if self.story_ids is not None:
            return len(self.story_ids)
        elif self.stories is not None:
            return len(self.stories)
        else:
            assert self._ending_labels is None, "Error: Found batch with only wrong ending labels, but no data"
            return 0

    def sent_len(self, sent_idx):
        '''
        Length: in number of words.
        for each story in the batch, return the number of tokens (~words) in the sent_idx'th sentence'''
        return [len(story[sent_idx]) for story in self.story_ids]

    def sents_len(self, sent_idces):
        '''
        Length: in number of words.
        For each story in the batch, return the combined length of the senteces indexed by sent_idces.'''
        return np.sum(np.array([self.sent_len(sent_idx) for sent_idx in sent_idces]), axis=0)

    @property
    def num_sentences(self):
        num = len(self.story_ids[0]) # num of sentences in first story
        for story in self.story_ids:
            assert len(story) == num, "Error: number of sentences is not defined for this batch, because it varies."
        return num

    ## No setters - don't modify a batch after creation
    @property   # the stories as lists of word-index-lists
    def story_ids(self):
        return self._story_ids
    @property
    def stories(self):
        return self._stories
    @property
    def ending_labels(self):
        assert self._ending_labels is not None, "Batch was created without data about correct endings "
        return self._ending_labels

    def get_padded_data(self, which_sentences=[0,1,2,3,4], use_next_step_as_target=True, pad_target=True):
        """
        Only for returning id lists.
        Pads the data with zeros, i.e. returns an np array of shape (batch_size, max_seq_length, dof). `max_seq_length`
        is the maximum occurring sequence length in the batch. Target is only padded if `pad_target` is True.
        :returns an 'input' consisting of the sentences given in :param which_sentences, concatenated to one list
                 a 'target' which is the same sentence-group, but shifted by one (not the correct classification of endings!)
                 - 'target' can be a 'None'-np.array if not use_next_step_as_target
                 a 'mask' which masks out padded entries
        """
        assert not pad_target or use_next_step_as_target # if pad_target, make sure there is a target to pad
        cur_seq_lengths = np.array([np.sum([len(story[i]) for i in which_sentences]) for story in self.story_ids])
        max_seq_length = max(cur_seq_lengths)
        # Not-TODO: Evil! Do *not* uncomment the next two lines. (Why is this the max seq length?)
       # if use_next_step_as_target:
       #     max_seq_length = max_seq_length - 1 # remove last step because it doesn't have a target

        inputs = []
        targets = []
        for this_storys_ids in self._story_ids:
            X = [this_storys_ids[i] for i in which_sentences]
            X = np.concatenate(X)
            Y = None
            if use_next_step_as_target:
                X = X[ : -1]
                Y = [this_storys_ids[i] for i in which_sentences]
                Y = np.concatenate(Y)[ 1 : ]
            missing = max_seq_length - X.shape[0]
            x_padded, y_padded = X, Y
            if missing > 0:
                # this batch entry has a smaller sequence length then the max sequence length, so pad it with zeros
                x_padded = np.concatenate([X, np.zeros(shape=[missing])], axis=0)
                if pad_target:
                    y_padded = np.concatenate([Y, np.zeros(shape=[missing])], axis=0)
            assert len(x_padded) == max_seq_length
            inputs.append(x_padded)

            if pad_target:
                assert len(y_padded) == max_seq_length
                targets.append(y_padded)

        #ltri = np.tril(np.ones([max_seq_length, max_seq_length]))
        #mask = ltri[self.seq_lengths - 1]

        return np.array(inputs), np.array(targets), np.array(cur_seq_lengths) # TODO: check shape of these arrays



    def get_ending_labels(self):
        assert self._ending_labels is not None
        raise NotImplementedError


class Preprocessor():
    def __init__(self, config, dataset=None):
        #self.max_sentence_length = config['max_sentence_length']
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
            story_ids.append(prep_story_vec)  #-> list of lists of lists
        return story_ids