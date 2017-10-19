# coding: utf-8
from subprocess import Popen
import numpy as np
import redis
import pickle
import sys


chunk_data = sys.argv[1]
worker_id = sys.argv[2]
cur_iter = sys.argv[3]
root_dir = "/home/rudvlf0413/distributedKGE/Embedding"

with open(f"{root_dir}/tmp/maxmin_{worker_id}.txt", 'w') as f:
    f.write(chunk_data)


# redis에서 embedding vector들 받아오기
print("redis...")
r = redis.StrictRedis(host='163.152.20.66', port=6379, db=0)

entities = pickle.loads(r.get('entities'))
relations = pickle.loads(r.get('relations'))

entity_id = r.mget(entities)
entity_id = {entity: int(entity_id[i]) for i, entity in enumerate(entities)}

relation_id = r.mget(relations)
relation_id = {relation: int(relation_id[i]) for i, relation in enumerate(relations)}

entities_initialized = r.mget([entity+'_v' for entity in entities])
entities_initialized = [pickle.loads(v) for v in entities_initialized]

relations_initialized = r.mget([relation+'_v' for relation in relations])
relations_initialized = [pickle.loads(v) for v in relations_initialized]


print("save file...")
# 좀 더 빨리 처리 하는 법 찾기!
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
proc = Popen(f"{root_dir}/MultiChannelEmbedding/Embedding.out", cwd=f'{root_dir}/preprocess/')
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
