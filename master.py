# coding: utf-8

from distributed import Client
from distributed.client import as_completed
from sklearn.preprocessing import normalize
from subprocess import Popen
import numpy as np
import redis
import pickle
import os
import sys


num_worker = int(sys.argv[1])
root = 'fb15k/'
data_files = ['train.txt', 'dev.txt', 'test.txt']
epoch = 1
n_dim = 20

entities = set()
relations = set()
entity_graph = []

# 여기서 전처리 C++ 프로그램 비동기 호출
# 조금 시간 걸림
print("Preprocessing start...")
proc = Popen(
    "/home/rudvlf0413/distributedKGE/Embedding/preprocess/preprocess.out",
    cwd='/home/rudvlf0413/distributedKGE/Embedding/preprocess/')


print("read files")
for file in data_files:
    with open(root+file, 'r') as f:
        for line in f:
            head, relation, tail = line[:-1].split("\t")
            entities.add(head)
            entities.add(tail)
            relations.add(relation)

            entity_graph.append((head, tail))

entities = sorted(entities)
entity_id = {e: i for i, e in enumerate(entities)}
relations = sorted(relations)
relation_id = {r: i for i, r in enumerate(relations)}


print("redis...")
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


def install():
    # install redis in worker machine
    os.system("pip install redis")
    os.system("pip install hiredis")


def work(worker_id, cur_epoch):
    proc = Popen(["python", "/home/rudvlf0413/distributedKGE/Embedding/worker.py", str(worker_id), str(cur_epoch)])
    proc.wait()
    return "process {}: finished".format(worker_id)


def savePreprocessedData(worker_id):
    print("bb")
    with open(f"/home/rudvlf0413/distributedKGE/Embedding/tmp/data_model_{worker_id}.bin", "wb") as f:
        f.write(r.get('prep_data'))

    return f"{worker_id} finish saving file!"


# 전처리 끝날때까지 대기
proc.wait()
print("Finished preocessing!")
with open("/home/rudvlf0413/distributedKGE/Embedding/tmp/data_model.bin", 'rb') as f:
    r.set('prep_data', f.read())


client = Client('163.152.20.66:8786', asynchronous=True, name='Embedding')
# install libraries
client.run(install)


results = []
for i in range(num_worker):
    worker_id = 'worker-%02d' % (i+1)
    results.append(client.submit(savePreprocessedData, worker_id, workers=[worker_id]))

for result in as_completed(results):
    print(result.result())


for e in range(epoch):
    # 작업 배정
    results = []
    for i in range(num_worker):
        worker_id = 'worker-%02d' % (i+1)
        results.append(client.submit(work, worker_id, e, workers=[worker_id]))

    # max-min cut 실행, anchor 등 재분배
    print("aa")

    for result in as_completed(results):
        print(result.result())
