# coding: utf-8

from distributed import Client
from distributed.client import as_completed
from sklearn.preprocessing import normalize
from subprocess import Popen
from time import time
import numpy as np
import redis
import pickle
import os


root = 'fb15k/'
data_files = ['train.txt', 'dev.txt', 'test.txt']
epoch = 1
n_dim = 20

entities = set()
relations = set()

print("read files")
t = time()
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
print(time()-t)


print("redis...")
t = time()
r = redis.StrictRedis(host='163.152.20.66', port=6379, db=0)

r.mset(entity_id)
r.mset(relation_id)
r.set('entities', pickle.dumps(entities, protocol=pickle.HIGHEST_PROTOCOL))
r.set('relations', pickle.dumps(relations, protocol=pickle.HIGHEST_PROTOCOL))

entities_initialized = normalize(np.random.randn(len(entities), n_dim))
relations_initialized = normalize(np.random.randn(len(relations), n_dim))

r.mset({
    entity+'_v': pickle.dumps(
        entities_initialized[i],
        protocol=pickle.HIGHEST_PROTOCOL) for i, entity in enumerate(entities)})
r.mset({
    relation+'_v': pickle.dumps(
        relations_initialized[i],
        protocol=pickle.HIGHEST_PROTOCOL) for i, relation in enumerate(relations)})

print(time()-t)


def install():
    # install redis in worker machine
    os.system("pip install redis")
    os.system("pip install hiredis")


def work(worker_id, cur_epoch):
    # worker.py 호출
    proc = Popen(["python", "/home/rudvlf0413/distributedKGE/Embedding/worker.py", str(worker_id), str(cur_epoch)])
    proc.wait()
    return "process {}: finished".format(worker_id)


print("distributed...")
client = Client('163.152.20.66:8786', asynchronous=True, name='Embedding')

# install redis
client.run(install)

for e in range(epoch):
    # 작업 배정
    results = []
    for worker_id in range(10):
        results.append(client.submit(work, worker_id, e))

    # max-min cut 실행, anchor 등 재분배
    print("aa")

    for result in as_completed(results)  :
        print(result.result())
