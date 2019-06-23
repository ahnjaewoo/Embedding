# coding: utf-8
from pickle import HIGHEST_PROTOCOL, dumps
import numpy as np
from sklearn.preprocessing import normalize

from abstract import BaseMaster, BaseWorker
from struct import pack, unpack
from utils import iter_mget
from utils import iter_mset


class TransEMaster(BaseMaster):
    def __init__(self, redis_con, test_sock, entities, relations, n_dim=50):
        super(TransEMaster, self).__init__(redis_con, test_sock, entities, relations, n_dim)

    def initialize_vectors(self):
        self.redis_con.set('entities', dumps(self.entities, protocol=HIGHEST_PROTOCOL))
        entities_initialized = normalize(np.random.randn(len(self.entities), self.n_dim).astype(self.np_dtype))
        iter_mset(self.redis_con, {f'{entity}_v': v.tostring() for v, entity in zip(entities_initialized, self.entities)})
        
        self.redis_con.set('relations', dumps(self.relations, protocol=HIGHEST_PROTOCOL))
        relations_initialized = normalize(np.random.randn(len(self.relations), self.n_dim).astype(self.np_dtype))
        iter_mset(self.redis_con, {f'{relation}_v': v.tostring() for v, relation in zip(relations_initialized,
            self.relations)})

    def load_trained_vectors(self):
        self.entity_vectors = iter_mget(self.redis_con, [f'{entity}_v' for entity in self.entities])
        self.entity_vectors = np.stack([np.fromstring(v, dtype=self.np_dtype) for v in self.entity_vectors])
        
        self.relation_vectors = iter_mget(self.redis_con, [f'{relation}_v' for relation in self.relations])
        self.relation_vectors = np.stack([np.fromstring(v, dtype=self.np_dtype) for v in self.relation_vectors])

    def send_vectors_for_test(self):
        for vector in self.entity_vectors:
            self.test_sock.send(pack(self.precision_string * len(vector), * vector))

        for vector in self.relation_vectors:
            self.test_sock.send(pack(self.precision_string * len(vector), *vector))


class TransGMaster(BaseMaster):
    def __init__(self, redis_con, entities, relations, n_cluster,
                 n_dim=50):
        super(TransGMaster, self).__init__(redis_con, entities, relations, n_dim)
        self.n_cluster = n_cluster

    def initialize_vectors(self):
        self.redis_con.set('entities', dumps(self.entities, protocol=HIGHEST_PROTOCOL))
        entities_initialized = normalize(np.random.randn(len(self.entities), self.n_dim).astype(self.np_dtype))
        iter_mset(self.redis_con, {f'{entity}_v': v.tostring() for v, entity in zip(entities_initialized, self.entities)})
        
        self.redis_con.set('relations', dumps(self.relations, protocol=HIGHEST_PROTOCOL))
        
        # xavier initialization
        embedding_clusters = np.random.random((len(self.relations), 21 * self.n_dim)).astype(self.np_dtype)
        embedding_clusters = (2 * embedding_clusters - 1) * np.sqrt(6 / self.n_dim)
        iter_mset(self.redis_con, {f'{relation}_cv': v.tostring() for v, relation in zip(embedding_clusters,
            self.relations)})

        weights_clusters = np.zeros((len(self.relations), 21)).astype(self.np_dtype)
        weights_clusters[:, :self.n_cluster] = 1
        normalize(weights_clusters, norm='l1', copy=False)
        iter_mset(self.redis_con, {f'{relation}_wv': v.tostring() for v, relation in zip(weights_clusters,
            self.relations)})

        size_clusters = np.full(len(self.relations), self.n_cluster, dtype=np.int32)
        iter_mset(self.redis_con, {f'{relation}_s': v.tostring() for v, relation in zip(size_clusters, self.relations)})

    def load_trained_vectors(self):
        self.entity_vectors = iter_mget(self.redis_con, [f'{entity}_v' for entity in self.entities])
        self.entity_vectors = np.stack([np.fromstring(v, dtype=self.np_dtype) for v in self.entity_vectors])
        
        self.embedding_clusters = iter_mget(self.redis_con, [f'{relation}_cv' for relation in self.relations])
        self.embedding_clusters = np.stack([np.fromstring(v, dtype=self.np_dtype) for v in self.embedding_clusters])
        self.weights_clusters = iter_mget(self.redis_con, [f'{relation}_wv' for relation in self.relations])
        self.weights_clusters = np.stack([np.fromstring(v, dtype=self.np_dtype) for v in self.weights_clusters])
        self.size_clusters = iter_mget(self.redis_con, [f'{relation}_s' for relation in self.relations])
        self.size_clusters = np.stack([np.fromstring(v, dtype=np.int32) for v in self.size_clusters])

    def send_vectors_for_test(self):
        for vector in self.entity_vectors:
            self.test_sock.send(pack(self.precision_string * len(vector), * vector))
        
        for vector in self.embedding_clusters:
            self.test_sock.send(pack(self.precision_string * len(vector), *vector))

        for vector in self.weights_clusters:
            self.test_sock.send(pack(self.precision_string * len(vector), *vector))

        size_clusters = self.size_clusters.reshape(-1)
        self.test_sock.send(pack('!' + 'i' * len(size_clusters), *size_clusters))


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


class TransGWorker(BaseWorker):
    def __init__(self):
        super(TransGWorker, self).__init__()

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
