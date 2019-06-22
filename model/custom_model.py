# coding: utf-8

from abstract import BaseMaster, BaseWorker


class TransEMaster(BaseMaster):
    def __init__(self, redis_con):
        super(TransEMaster, self).__init__()
        self.redis_con = redis_con

    def initialize_vectors(self):
        pass

    def load_trained_vectors(self):
        pass

    def send_vectors_for_test(self):
        pass


class TransEWorker(BaseWorker):
    def __init__(self):
        super(TransEWorker, self).__init__()

    def load_initialized_vectors(self):
        pass

    def send_entities(self):
        pass

    def send_relations(self):
        pass

    def get_entities(self):
        pass

    def get_relations(self):
        pass

    def update_vectors(self):
        pass
