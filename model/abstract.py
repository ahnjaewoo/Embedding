# coding: utf-8


class BaseMaster():
    def __init__(self, redis_con):
        self.redis_con = redis_con
        pass

    def initialize_vectors(self):
        raise NotImplementedError

    def load_trained_vectors(self):
        raise NotImplementedError

    def send_vectors_for_test(self):
        raise NotImplementedError


class BaseWorker():
    def __init__(self, redis_con):
        self.redis_con = redis_con
        pass

    def load_initialized_vectors(self):
        raise NotImplementedError

    def send_entities(self):
        raise NotImplementedError

    def send_relations(self):
        raise NotImplementedError

    def get_entities(self):
        raise NotImplementedError

    def get_relations(self):
        raise NotImplementedError

    def update_vectors(self):
        raise NotImplementedError
