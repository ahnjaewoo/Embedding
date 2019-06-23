# coding: utf-8
import numpy as np

class BaseMaster():
    def __init__(self, redis_con, test_sock, entities, relations, n_dim, precision=0):
        self.redis_con = redis_con
        self.test_sock = test_sock

        self.entities = entities
        self.relations = relations

        self.n_dim = n_dim
        # precision: 0 for single precision, 1 for half precision
        self.precision_string = 'f' if precision == 0 else 'e'
        self.precision_byte = 4 if precision == 0 else 2
        self.np_dtype = np.float32 if precision == 0 else np.float16

    def initialize_vectors(self, *args):
        raise NotImplementedError

    def load_trained_vectors(self, *args):
        raise NotImplementedError

    def send_vectors_for_test(self, *args):
        raise NotImplementedError


class BaseWorker():
    def __init__(self, redis_con, n_dim, np_dtype):
        self.redis_con = redis_con
        self.n_dim = n_dim
        self.np_dtype = np_dtype

    def load_initialized_vectors(self, *args):
        raise NotImplementedError

    def send_entities(self, *args):
        raise NotImplementedError

    def send_relations(self, *args):
        raise NotImplementedError

    def get_entities(self, *args):
        raise NotImplementedError

    def get_relations(self, *args):
        raise NotImplementedError

    def update_vectors(self, *args):
        raise NotImplementedError
