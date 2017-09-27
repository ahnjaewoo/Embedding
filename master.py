# coding: utf-8

from distributed import Client
from sklearn.preprocessing import normalize
from subprocess import Popen
import numpy as np
import redis
import pickle

# master node에 redis db 실행해야함
# master에서 dask-scheduler 실행
# 각 worker pc에서는 dask-worker <마스터 ip>:8786

root = 'fb15k/'
data_files = ['train.txt', 'dev.txt', 'test.txt']

entities = set()
relations = set()

print("read files")
for file in data_files:
    with open(root+file, 'r') as f:
        for line in f:
            head, relation, tail = line[:-1].split("\t")
            entities.add(head)
            entities.add(tail)
            relations.add(relation)

entities = sorted(entities)
entity_id = {e: i for i, e in enumerate(entities)}
relations = sorted(relations)
relation_id = {r: i for i, r in enumerate(relations)}


print("redis...")
r = redis.StrictRedis(host='163.152.20.66', port=6379, db=0, password='davian!')

r.mset(entity_id)
r.mset(relation_id)
r.set('entities', pickle.dumps(entities))
r.set('relations', pickle.dumps(relations))

entities_initialized = normalize(np.random.random((len(entities), 300)))
relations_initialized = normalize(np.random.random((len(relations), 300)))

r.mset({entity+'_v': pickle.dumps(entities_initialized[i]) for i, entity in enumerate(entities)})
r.mset({relation+'_v': pickle.dumps(relations_initialized[i]) for i, relation in enumerate(relations)})


print("distributed...")
client = Client('163.152.20.66:8786', asynchronous=True)

def work(i):
    proc = Popen(["python", "worker.py"])
    proc.wait()
    return "process {}: finished".format(i)

# 작업 배정
results = []
for i in range(10):
    # worker.py 호출
    results.append(client.submit(work, i, pure=False))

print("aa")
# max-min cut 실행

# worker들 작업 끝나면 anchor 등 재분배
for result in results:
    print(result.result())
