# coding: utf-8
from subprocess import Popen
import numpy as np
import redis
import pickle
import sys


chunk_data = sys.argv[1]
worker_id = sys.argv[2]
cur_iter = sys.argv[3]
embedding_dim = sys.argv[4]
learning_rate = sys.argv[5]
margin = sys.argv[6]
root_dir = "/home/rudvlf0413/distributedKGE/Embedding"


# redis에서 embedding vector들 받아오기
r = redis.StrictRedis(host='163.152.20.66', port=6379, db=0)
entities = pickle.loads(r.get('entities'))
relations = pickle.loads(r.get('relations'))
entity_id = r.mget(entities)
relation_id = r.mget(relations)
entities_initialized = r.mget([entity+'_v' for entity in entities])
relations_initialized = r.mget([relation+'_v' for relation in relations])

entity_id = {entity: int(entity_id[i]) for i, entity in enumerate(entities)}
relation_id = {relation: int(relation_id[i]) for i, relation in enumerate(relations)}

entities_initialized = [pickle.loads(v) for v in entities_initialized]
relations_initialized = [pickle.loads(v) for v in relations_initialized]


if int(cur_iter) % 2 == 0:
    with open(f"{root_dir}/tmp/maxmin_{worker_id}.txt", 'w') as f:
        f.write(chunk_data)
else:
    sub_graphs = pickle.loads(r.get(f'sub_graph_{worker_id}'))
    with open(f"{root_dir}/tmp/sub_graph_{worker_id}.txt", 'w') as f:
        for (head_id, relation_id, tail_id) in sub_graphs:
            f.write(f"{head_id} {relation_id} {tail_id}\n")


# matrix를 text로 빨리 저장하는 법 찾기!
with open("./tmp/entity_vectors.txt", 'w') as f:
    for i, vector in enumerate(entities_initialized):
        f.write(str(entities[i]) + "\t")
        f.write(" ".join([str(v) for v in vector]) + '\n')

with open("./tmp/relation_vectors.txt", 'w') as f:
    for i, relation in enumerate(relations_initialized):
        f.write(str(relations[i]) + "\t")
        f.write(" ".join([str(v) for v in relation]) + '\n')


del entities_initialized
del relations_initialized


# 여기서 C++ 프로그램 호출
proc = Popen([
    f"{root_dir}/MultiChannelEmbedding/Embedding.out", 
    worker_id, cur_iter, embedding_dim, learning_rate, margin],
    cwd=f'{root_dir}/preprocess/')
proc.wait()


entity_vectors = {}
with open(f"{root_dir}/tmp/entity_vectors_updated.txt", 'r') as f:
    for line in f:
        line = line[:-1].split()
        entity_vectors[line[0] + '_v'] = pickle.dumps(np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)

r.mset(entity_vectors)

relation_vectors = {}
with open(f"{root_dir}/tmp/relation_vectors_updated.txt", 'r') as f:
    for line in f:
        line = line[:-1].split()
        relation_vectors[line[0] + '_v'] = pickle.dumps(np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)

r.mset(relation_vectors)

print("finished!")
