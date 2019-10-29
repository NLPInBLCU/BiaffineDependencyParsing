from collections import OrderedDict

PAD = '<PAD>'
PAD_ID = 0
UNK = '<UNK>'
UNK_ID = 1
EMPTY = '<EMPTY>'
EMPTY_ID = 2
ROOT = '<ROOT>'
ROOT_ID = 3
VOCAB_PREFIX = [PAD, UNK, EMPTY, ROOT]


class BaseVocab:
    """ A base class for deprecated_common vocabulary operations. Each subclass should at least
    implement its own build_vocab() function."""

    def __init__(self, data=None, lang="", idx=0, cutoff=0, lower=False):
        self.data = data
        self.lang = lang
        self.idx = idx
        self.cutoff = cutoff
        self.lower = lower
        if data is not None:
            self.build_vocab()
        self.state_attrs = ['lang', 'idx', 'cutoff', 'lower', '_unit2id', '_id2unit']

    def build_vocab(self):
        raise NotImplementedError()

    def state_dict(self):
        """ Returns a dictionary containing all states that are necessary to recover
        this vocab. Useful for serialization."""
        state = OrderedDict()
        for attr in self.state_attrs:
            if hasattr(self, attr):
                state[attr] = getattr(self, attr)
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        """ Returns a new Vocab instance constructed from a state dict. """
        new = cls()
        for attr, value in state_dict.items():
            setattr(new, attr, value)
        return new

    def normalize_unit(self, unit):
        if self.lower:
            return unit.lower()
        return unit

    def unit2id(self, unit):
        unit = self.normalize_unit(unit)
        if unit in self._unit2id:
            return self._unit2id[unit]
        else:
            return self._unit2id[UNK]

    def id2unit(self, id):
        return self._id2unit[id]

    def map(self, units):
        return [self.unit2id(x) for x in units]

    def unmap(self, ids):
        return [self.id2unit(x) for x in ids]

    def __len__(self):
        return len(self._id2unit)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.unit2id(key)
        elif isinstance(key, int) or isinstance(key, list):
            return self.id2unit(key)
        else:
            raise TypeError("Vocab key must be one of str, list, or int")

    def __contains__(self, key):
        return key in self._unit2id

    @property
    def size(self):
        return len(self)


class BaseMultiVocab:
    """ A convenient vocab container that can store multiple BaseVocab instances, and support 
    safe serialization of all instances via state dicts. Each subclass of this base class 
    should implement the load_state_dict() function to specify how a saved state dict 
    should be loaded back."""

    def __init__(self, vocab_dict=None):
        self._vocabs = OrderedDict()
        if vocab_dict is None:
            return
        # check all values provided must be a subclass of the Vocab base class
        assert all([isinstance(v, BaseVocab) for v in vocab_dict.values()])
        for k, v in vocab_dict.items():
            self._vocabs[k] = v

    def __setitem__(self, key, item):
        self._vocabs[key] = item

    def __getitem__(self, key):
        return self._vocabs[key]

    def state_dict(self):
        """ Build a state dict by iteratively calling state_dict() of all vocabs. """
        state = OrderedDict()
        for k, v in self._vocabs.items():
            state[k] = v.state_dict()
        return state

    @classmethod
    def load_state_dict(cls, state_dict):
        """ Construct a MultiVocab by reading from a state dict."""
        raise NotImplementedError
