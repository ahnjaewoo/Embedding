# coding: utf-8

from subprocess import Popen
from time import time
import redis
import pickle
import shlex


r = redis.StrictRedis(host='163.152.20.66', port=6379, db=0, password='davian!')

# redis에서 embedding vector들 받아오기
print("redis...")
t = time()
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
print(time())
# 파일로 저장, c에서 불러와야!
with open(f"tmp/entity_vectors.txt", 'w') as f:
    for i, vector in enumerate(entities_initialized):
        f.write(str(entities[i])+f"\t")
        f.write(" ".join([str(v) for v in vector])+'\n')

with open(f"tmp/relation_vectors.txt", 'w') as f:
    for i, vector in enumerate(relations_initialized):
        f.write(str(relations[i])+f"\t")
        f.write(" ".join([str(v) for v in vector])+'\n')

print(time()-t)

print("finished!")


"""
# c 프로그램 호출
command = "bash test.sh"
args = shlex.split(command)
proc = Popen(args)
proc.wait()
