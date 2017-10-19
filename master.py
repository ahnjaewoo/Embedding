# coding: utf-8
from distributed import Client
from distributed.client import as_completed
from sklearn.preprocessing import normalize
from subprocess import Popen
from argparse import ArgumentParser
import numpy as np
import redis
import pickle


parser = ArgumentParser(description='Distributed Knowledge Graph Embedding')
parser.add_argument('--num_worker', type=int, default=2, help='number of workers')
parser.add_argument('--data_root', type=str, default='./fb15k', help='root directory of data')
parser.add_argument('--niter', type=int, default=1, help='total number of training iterations')
parser.add_argument('--install', default=False, help='install libraries in each worker')
parser.add_argument('--ndim', type=int, default=20, help='dimension of embeddings')
args = parser.parse_args()

install = args.install
data_root = args.data_root
root_dir = "/home/rudvlf0413/distributedKGE/Embedding"
data_files = ['/train.txt', '/dev.txt', '/test.txt']
num_worker = args.num_worker
niter = args.niter
n_dim = args.ndim


entities = set()
relations = set()

# 여기서 전처리 C++ 프로그램 비동기 호출
print("Preprocessing start...")
proc = Popen(f"{root_dir}/preprocess/preprocess.out", cwd=f'{root_dir}/preprocess/')


print("read files")
for file in data_files:
    with open(data_root+file, 'r') as f:
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


def install_libs():
    import os
    os.system("pip install redis")
    os.system("pip install hiredis")


def work(chunk_data, worker_id, cur_iter):
    proc = Popen(["python", f"{root_dir}/worker.py", chunk_data, str(worker_id), str(cur_iter)])
    proc.wait()
    return "process {}: finished".format(worker_id)


def savePreprocessedData(data, worker_id):
    from threading import Thread
    
    def saveFile(data):
        with open(f"{root_dir}/tmp/data_model_{worker_id}.bin", 'wb') as f:
            f.write(data)

    thread = Thread(target=saveFile, args=(data, ))
    thread.start()
    thread.join()

    return f"{worker_id} finish saving file!"


client = Client('163.152.20.66:8786', asynchronous=True, name='Embedding')
if install:
    client.run(install_libs)


# 전처리 끝날때까지 대기
proc.wait()
with open(f"{root_dir}/tmp/data_model.bin", 'rb') as f:
    data = f.read()


workers = []
for i in range(num_worker):
    worker_id = f'worker-{i}'
    workers.append(client.submit(savePreprocessedData, data, worker_id))

for worker in as_completed(workers):
    print(worker.result())


# max-min cut 실행, anchor 분배
proc = Popen(["/home/rudvlf0413/pypy/bin/pypy", 'maxmin.py', str(num_worker)])
proc.wait()
with open(f"{root_dir}/tmp/maxmin_output.txt") as f:
    lines = f.read().splitlines()
    anchors, chunks = lines[0], lines[1:]

for cur_iter in range(niter):
    # 작업 배정
    workers = []
    for i in range(num_worker):
        worker_id = f'worker_{i}'
        chunk_data = "{}\n{}".format(anchors, chunks[i])
        workers.append(client.submit(work, chunk_data, worker_id, cur_iter))

    # max-min cut 실행, anchor 등 재분배
    proc = Popen(["/home/rudvlf0413/pypy/bin/pypy", 'maxmin.py', str(num_worker)])
    proc.wait()

    with open(f"{root_dir}/tmp/maxmin_output.txt") as f:
        lines = f.read().splitlines()
        anchors, chunks = lines[0], lines[1:]

    for worker in as_completed(workers):
        print(worker.result())
