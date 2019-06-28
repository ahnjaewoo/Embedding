# coding: utf-8
from pickle import HIGHEST_PROTOCOL, dumps
import numpy as np
from sklearn.preprocessing import normalize

from .abstract import BaseMaster, BaseWorker
from struct import pack, unpack

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import iter_mget, iter_mset, sockRecv, loads


class TransEMaster(BaseMaster):
    def __init__(self, redis_con, entities, relations, n_dim=50):
        super(TransEMaster, self).__init__(redis_con, entities, relations, n_dim)

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

    def send_vectors_for_test(self, test_sock):
        for vector in self.entity_vectors:
            test_sock.send(pack(self.precision_string * len(vector), * vector))

        for vector in self.relation_vectors:
            test_sock.send(pack(self.precision_string * len(vector), *vector))


class TransGMaster(BaseMaster):
    def __init__(self, redis_con, entities, relations, n_cluster, n_dim=50):
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

    def send_vectors_for_test(self, test_sock):
        for vector in self.entity_vectors:
            test_sock.send(pack(self.precision_string * len(vector), * vector))
        
        for vector in self.embedding_clusters:
            test_sock.send(pack(self.precision_string * len(vector), *vector))

        for vector in self.weights_clusters:
            test_sock.send(pack(self.precision_string * len(vector), *vector))

        size_clusters = self.size_clusters.reshape(-1)
        test_sock.send(pack('!' + 'i' * len(size_clusters), *size_clusters))


class TransEWorker(BaseWorker):
    def __init__(self, redis_con, embedding_sock, embedding_dim):
        super(TransEWorker, self).__init__(redis_con, embedding_sock, embedding_dim)

    def load_initialized_vectors(self):
        self.entities = np.array(loads(self.redis_con.get('entities')))
        entity_vectors = iter_mget(self.redis_con, [f'{entity}_v' for entity in self.entities])
        self.entity_vectors = np.stack([np.fromstring(v, dtype=self.np_dtype) for v in entity_vectors])

        self.relations = np.array(loads(self.redis_con.get('relations')))
        relation_vectors = iter_mget(self.redis_con, [f'{relation}_v' for relation in self.relations])
        self.relation_vectors = np.stack([np.fromstring(v, dtype=self.np_dtype) for v in relation_vectors])

    def send_entities(self):
        for vector in self.entity_vectors:
            self.embedding_sock.send(pack(self.precision_string * self.embedding_dim, *vector))

        self.entity_vectors = None

    def send_relations(self):
        for vector in self.relation_vectors:
            self.embedding_sock.send(pack(self.precision_string * self.embedding_dim, *vector))

        self.relation_vectors = None

    def get_entities(self):
        count_entity = unpack('!i', sockRecv(self.embedding_sock, 4))[0]
        entity_id_list = unpack('!' + 'i' * count_entity, sockRecv(self.embedding_sock, count_entity * 4))
        
        entity_vector_list = unpack(self.precision_string * count_entity * self.embedding_dim,
            sockRecv(self.embedding_sock, self.precision_byte * self.embedding_dim * count_entity))
        
        entity_vector_list = np.array(entity_vector_list, dtype=self.np_dtype).reshape(count_entity, self.embedding_dim)
        self.entity_vectors = {f"{self.entities[id_]}_v": v.tostring() for v, id_ in zip(entity_vector_list, entity_id_list)}

    def get_relations(self):
        count_relation = unpack('!i', sockRecv(self.embedding_sock, 4))[0]

        relation_id_list = unpack('!' + 'i' * count_relation, sockRecv(self.embedding_sock, count_relation * 4))
        relation_vector_list = unpack(self.precision_string * count_relation * self.embedding_dim,
            sockRecv(self.embedding_sock, self.precision_byte * self.embedding_dim * count_relation))
        
        # relation_vectors 전송
        relation_vector_list = np.array(relation_vector_list, dtype=self.np_dtype).reshape(count_relation, self.embedding_dim)
        self.relation_vectors = {f"{self.relations[id_]}_v": v.tostring() for v, id_ in zip(relation_vector_list, relation_id_list)}
        
    def update_entities(self):
        iter_mset(self.redis_con, self.entity_vectors)
    
    def update_relations(self):
        iter_mset(self.redis_con, self.relation_vectors)


class TransGWorker(BaseWorker):
    def __init__(self, redis_con, embedding_sock, embedding_dim):
        super(TransGWorker, self).__init__(redis_con, embedding_sock, embedding_dim)

    def load_initialized_vectors(self):
        self.entities = np.array(loads(self.redis_con.get('entities')))
        entity_vectors = iter_mget(self.redis_con, [f'{entity}_v' for entity in self.entities])
        self.entity_vectors = np.stack([np.fromstring(v, dtype=self.np_dtype) for v in entity_vectors])

        self.relations = np.array(loads(self.redis_con.get('relations')))
        embedding_clusters = iter_mget(self.redis_con, [f'{relation}_cv' for relation in self.relations])
        self.embedding_clusters = np.stack([np.fromstring(v, dtype=self.np_dtype) for v in embedding_clusters])
        weights_clusters = iter_mget(self.redis_con, [f'{relation}_wv' for relation in self.relations])
        self.weights_clusters = np.stack([np.fromstring(v, dtype=self.np_dtype) for v in weights_clusters])
        size_clusters = iter_mget(self.redis_con, [f'{relation}_s' for relation in self.relations])
        self.size_clusters = np.stack([np.fromstring(v, dtype=np.int32) for v in size_clusters])
        
    def send_entities(self):
        for vector in self.entity_vectors:
            self.embedding_sock.send(pack(self.precision_string * self.embedding_dim, *vector))

        self.entity_vectors = None

    def send_relations(self):
        # embedding_clusters 전송 - GeometricModel load    
        for vector in self.embedding_clusters:

            self.embedding_sock.send(pack(self.precision_string * len(vector), *vector))

        # weights_clusters 전송 - GeometricModel load
        for vector in self.weights_clusters:

            self.embedding_sock.send(pack(self.precision_string * len(vector), *vector))

        # size_clusters 전송 - GeometricModel load
        size_clusters = self.size_clusters.reshape(-1)
        self.embedding_sock.send(pack('!' + 'i' * len(size_clusters), *size_clusters))

        self.embedding_clusters = None
        self.weights_clusters = None
        self.size_clusters = None

    def get_entities(self):
        count_entity = unpack('!i', sockRecv(self.embedding_sock, 4))[0]

        entity_id_list = unpack('!' + 'i' * count_entity, sockRecv(self.embedding_sock, count_entity * 4))
        entity_vector_list = unpack(self.precision_string * count_entity * self.embedding_dim,
            sockRecv(self.embedding_sock, self.precision_byte * self.embedding_dim * count_entity))
        
        entity_vector_list = np.array(entity_vector_list, dtype=self.np_dtype).reshape(count_entity, self.embedding_dim)
        self.entity_vectors = {f"{self.entities[id_]}_v": v.tostring() for v, id_ in zip(entity_vector_list, entity_id_list)}

    def get_relations(self):
        count_relation = unpack('!i', sockRecv(self.embedding_sock, 4))[0]
        relation_id_list = unpack('!' + 'i' * count_relation, sockRecv(self.embedding_sock, count_relation * 4))
        
        # embedding_clusters 전송
        cluster_vector_list = unpack(self.precision_string * count_relation * self.embedding_dim * 21,
            sockRecv(self.embedding_sock, 21 * self.precision_byte * self.embedding_dim * count_relation))
        cluster_vector_list = np.array(cluster_vector_list, dtype=self.np_dtype).reshape(count_relation, 21 * self.embedding_dim)
        self.cluster_vectors = {f"{self.relations[id_]}_cv": v.tostring() for v, id_ in
                zip(cluster_vector_list, relation_id_list)}
        
        # weights_clusters 전송
        weights_clusters_list = unpack(self.precision_string * count_relation * 21,
            sockRecv(self.embedding_sock, self.precision_byte * 21 * count_relation))
        weights_clusters_list = np.array(weights_clusters_list, dtype=self.np_dtype).reshape(count_relation, 21)
        self.weights_clusters = {f"{self.relations[id_]}_wv": v.tostring() for v, id_ in zip(weights_clusters_list, relation_id_list)}
        
        # size_clusters 전송
        size_clusters_list = unpack('!' + 'i' * count_relation, sockRecv(self.embedding_sock, 4 * count_relation))
        size_clusters_list = np.array(size_clusters_list, dtype=np.int32)
        self.size_clusters = {f"{self.relations[id_]}_s": v.tostring() for v, id_ in zip(size_clusters_list, relation_id_list)}

    def update_entities(self):
        iter_mset(self.redis_con, self.entity_vectors)
    
    def update_relations(self):
        iter_mset(self.redis_con, self.cluster_vectors)
        iter_mset(self.redis_con, self.weights_clusters)
        iter_mset(self.redis_con, self.size_clusters)
