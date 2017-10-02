# coding: utf-8

from subprocess import Popen
from time import time
import redis
import pickle
import sys


worker_id = sys.argv[1]
cur_epoch = sys.argv[2]


# redis에서 embedding vector들 받아오기
print("redis...")
t = time()
r = redis.StrictRedis(host='163.152.20.66', port=6379, db=0, password='davian!')

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
print(time()-t)


print("save file...")
t = time()

# 좀 더 빨리 처리 하는 법 찾기
with open("tmp/entity_vectors.txt", 'w') as f:
    for i, vector in enumerate(entities_initialized):
        f.write(str(entities[i]) + "\t")
        f.write(" ".join([str(v) for v in vector]) + '\n')

with open("tmp/relation_vectors.txt", 'w') as f:
    for i, relation in enumerate(relations_initialized):
        f.write(str(relations[i]) + "\t")
        f.write(" ".join([str(v) for v in relation]) + '\n')

print(time()-t)

del entities_initialized
del relations_initialized


"""
# 여기서 C++ 프로그램 호출
proc = Popen(["bash", "test.sh"])
proc.wait()


entity_vectors = {}
with open("tmp/entity_vectors_updated", 'r') as f:
    for line in f:
        line = line[:-1].split()
        entity_vectors[line[0]] = pickle.dumps(np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)

r.mset(entity_vectors)

relation_vectors = {}
with open("tmp/relation_vectors_updated", 'r') as f:
    for line in f:
        line = line[:-1].split()
        relation_vectors[line[0]] = pickle.dumps(np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)

r.mset(relation_vectors)
"""

print("finished!")
