import numpy as np

import cPickle as pickle

class Datum(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def serialize(self):
        return pickle.dumps((self.X.shape, self.X.tobytes(), int(self.y)), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def deserialize(bytes):
        shape, data, label = pickle.loads(bytes)
        return Datum(np.fromstring(data, dtype=np.uint8).reshape(shape), label)

from PIL import Image

class DatumJPEG(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def serialize(self):
        return pickle.dumps((self.X, int(self.y)), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def deserialize(bytes):
        data, label = pickle.loads(bytes)
        return DatumJPEG(Image.open(data), label)

class DatumFloat(object):
    def __init__(self, X):
        self.X = X

    def serialize(self):
        return pickle.dumps((self.X, 0), protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def deserialize(bytes):
        data, label = pickle.loads(bytes)
        return DatumJPEG(Image.open(data), label)
