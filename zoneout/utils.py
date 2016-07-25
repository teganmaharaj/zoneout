import numpy as np
from fuel.datasets import IterableDataset
from fuel.streams import DataStream
import theano
from fuel.transformers import Transformer
import fuel
floatX = theano.config.floatX



# PTB
_data_cache = dict()


def get_cPTB(which_set):
    if which_set not in _data_cache:
        try:
            data = np.load('/data/lisa/data/PennTreebankCorpus/char_level_penntree.npz')
        except:
            data = np.load('char_level_penntree.npz')
        #data = np.load(path)
        # put the entire thing on GPU in one-hot (takes
        # len(self.vocab) * len(self.data) * sizeof(floatX) bytes
        # which is about 1G for the training set and less for the
        # other sets)
        cudandarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.CudaNdarray
        # (doing it in numpy first because cudandarray doesn't accept
        # lists of indices)
        one_hot_data = np.eye(len(data["vocab"]), dtype=theano.config.floatX)[data[which_set]]
        _data_cache[which_set] = cudandarray(one_hot_data)
    return _data_cache[which_set]

def get_Text8(which_set):
    if which_set not in _data_cache:
        try:
            data = np.load('/data/lisa/data/PennTreebankCorpus/char_level_penntree.npz')
        except:
            data = np.load('tezt8.npz')
        #data = np.load(path)
        # put the entire thing on GPU in one-hot (takes
        # len(self.vocab) * len(self.data) * sizeof(floatX) bytes
        # which is about 1G for the training set and less for the
        # other sets)
        cudandarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.CudaNdarray
        # (doing it in numpy first because cudandarray doesn't accept
        # lists of indices)
        one_hot_data = np.eye(len(data["vocab"]), dtype=theano.config.floatX)[data[which_set]]
        _data_cache[which_set] = cudandarray(one_hot_data)
    return _data_cache[which_set]


class PTB(fuel.datasets.Dataset):
    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, which_set, length, augment=True):
        self.which_set = which_set
        self.length = length
        self.augment = augment
        self.data = get_cPTB(which_set)
        self.num_examples = int(len(self.data) / self.length)
        if self.augment:
            # -1 so we have one self.length worth of room for augmentation
            self.num_examples -= 1
        super(PTB, self).__init__()

    def open(self):
        offset = 0
        if self.augment:
            # choose an offset to get some data augmentation by
            # not always chopping the examples at the same point.
            offset = np.random.randint(self.length)
        # none of this should copy
        data = self.data[offset:]
        # reshape to nonoverlapping examples
        data = (data[:self.num_examples * self.length]
                .reshape((self.num_examples, self.length, self.data.shape[1])))
        # return the data so we will get it as the "state" argument to get_data
        return data

    def get_data(self, state, request):
        if isinstance(request, (tuple, list)):
            request = np.array(request, dtype=np.int64)
            return (state.take(request, 0),)
        return (state[request],)

class Text8(fuel.datasets.Dataset):
    provides_sources = ('features',)
    example_iteration_scheme = None

    def __init__(self, which_set, length, augment=True):
        self.which_set = which_set
        self.length = length
        self.augment = augment
        self.data = get_Text8(which_set)
        self.num_examples = int(len(self.data) / self.length)
        if self.augment:
            # -1 so we have one self.length worth of room for augmentation
            self.num_examples -= 1
        super(Text8, self).__init__()

    def open(self):
        offset = 0
        if self.augment:
            # choose an offset to get some data augmentation by
            # not always chopping the examples at the same point.
            offset = np.random.randint(self.length)
        # none of this should copy
        data = self.data[offset:]
        # reshape to nonoverlapping examples
        data = (data[:self.num_examples * self.length]
                .reshape((self.num_examples, self.length, self.data.shape[1])))
        # return the data so we will get it as the "state" argument to get_data
        return data

    def get_data(self, state, request):
        if isinstance(request, (tuple, list)):
            request = np.array(request, dtype=np.int64)
            return (state.take(request, 0),)
        return (state[request],)


class SampleZoneouts(Transformer):
    def __init__(self, data_stream, z_prob_states, z_prob_cells, drop_prob_igates, hidden_dim,
                 is_for_test, **kwargs):
        super(SampleZoneoutPTB, self).__init__(
            data_stream, **kwargs)
        self.z_prob_states = z_prob_states
        self.z_prob_cells = z_prob_cells
        self.drop_prob_igates = drop_prob_igates
        self.hidden_dim = hidden_dim
        self.is_for_test = is_for_test
        self.produces_examples = False

    def get_data(self, request=None):
        data = next(self.child_epoch_iterator)
        transformed_data = []
        # Now it is: T x B x F
        transformed_data.append(np.swapaxes(data[0], 0, 1))#[:-1])
        #transformed_data.append(np.swapaxes(data[0], 0, 1)[1:])
        T, B, _ = transformed_data[0].shape
        if self.is_for_test:
            zoneouts_states = np.ones((T, B, self.hidden_dim)) * self.z_prob_states
        else:
            zoneouts_states = np.random.binomial(n=1, p=self.z_prob_states,
                                       size=(T, B, self.hidden_dim))
        if self.is_for_test:
            zoneouts_cells = np.ones((T, B, self.hidden_dim)) * self.z_prob_cells
        else:
            zoneouts_cells = np.random.binomial(n=1, p=self.z_prob_cells,
                                       size=(T, B, self.hidden_dim))
        if self.is_for_test:
            zoneouts_igates = np.ones((T, B, self.hidden_dim)) * self.drop_prob_igates
        else:
            zoneouts_igates = np.random.binomial(n=1, p=self.drop_prob_igates,
                                       size=(T, B, self.hidden_dim))
        transformed_data.append(zoneouts_states.astype(floatX))
        transformed_data.append(zoneouts_cells.astype(floatX))
        transformed_data.append(zoneouts_igates.astype(floatX))
        return transformed_data


def get_ptb_stream(which_set, batch_size, length, z_prob_states, z_prob_cells, drop_prob_igates,
                   hidden_dim, for_evaluation, num_examples=None,
                   augment=True):
    
    dataset = PTB(which_set, length=length, augment=augment)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    ds = SampleZoneouts(stream, z_prob_states, z_prob_cells, drop_prob_igates, hidden_dim,
                        for_evaluation)
    ds.sources = ('features',  'zoneouts_states', 'zoneouts_cells', 'zoneouts_igates')#'outputs',
    return ds



def get_text8_stream(which_set, batch_size, length, z_prob_states, z_prob_cells, drop_prob_igates,
                   hidden_dim, for_evaluation, num_examples=None,
                   augment=True):
    
    dataset = Text8(which_set, length=length, augment=augment)
    if num_examples is None or num_examples > dataset.num_examples:
        num_examples = dataset.num_examples
    stream = fuel.streams.DataStream.default_stream(
        dataset,
        iteration_scheme=fuel.schemes.ShuffledScheme(num_examples, batch_size))
    ds = SampleZoneouts(stream, z_prob_states, z_prob_cells, drop_prob_igates, hidden_dim,
                        for_evaluation)
    ds.sources = ('features',  'zoneouts_states', 'zoneouts_cells', 'zoneouts_igates')#'outputs',
    return ds