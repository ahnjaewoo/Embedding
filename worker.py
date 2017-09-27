# coding: utf-8

from subprocess import Popen
import redis
import pickle
import shlex


r = redis.StrictRedis(host='163.152.20.66', port=6379, db=0, password='davian!')

# redis에서 embedding vector들 받아오기
entities = pickle.loads(r.get('entities'))
relations = pickle.loads(r.get('relations'))

entity_id = r.mget(entities)
entity_id = {entity: int(entity_id[i]) for i, entity in entities}

relation_id = r.mget(relations)
relation_id = {relation: int(relation_id[i]) for i, relation in relations}

entities_initialized = r.mget([entity+'_v' for entity in entities])
entities_initialized = [pickle.loads(v) for v in entities_initialized]

relations_initialzied = r.mget([relation+'_v' for relation in relations])
relations_initialized = [pickle.loads(v) for v in relations_initialized]

# 파일로 저장, c에서 불러와야!
with open(f"tmp/entity_vectors.txt", 'w') as f:
    for i, vector in enumerate(entities_initialized):
        f.write(str(entities[i])+f"\t")
        f.write(" ".join(vector)+'\n')

with open(f"tmp/relation_vectors.txt", 'w') as f:
    for i, relation in enumerate(relations_initialized):
        f.write(str(relations[i])+f"\t")
        f.write(" ".join(vector)+'\n')


"""
# c 프로그램 호출
command = "bash test.sh"
args = shlex.split(command)
proc = Popen(args)
proc.wait()


# embedding 학습 끝나면 c에서 업데이트한 것들 불러 오기

# redis에 해당 학습된 벡터들 업데이트하기
r.mset({f'_{i}': pickle.dumps(v) for i, v in enumerate(a)})
"""
