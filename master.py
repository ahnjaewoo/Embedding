# coding: utf-8
from distributed import Client
from distributed.client import as_completed
from sklearn.preprocessing import normalize
from subprocess import Popen
from argparse import ArgumentParser
from random import shuffle
from collections import defaultdict
import numpy as np
import redis
import pickle
from time import time

t_ = time()

parser = ArgumentParser(description='Distributed Knowledge Graph Embedding')
parser.add_argument('--num_worker', type=int, default=2, help='number of workers')
parser.add_argument('--data_root', type=str, default='./fb15k', help='root directory of data')
parser.add_argument('--niter', type=int, default=2, help='total number of training iterations')
parser.add_argument('--install', default=False, help='install libraries in each worker')
parser.add_argument('--ndim', type=int, default=20, help='dimension of embeddings')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--margin', type=int, default=2, help='margin')
args = parser.parse_args()

install = args.install
data_root = args.data_root
root_dir = "/home/rudvlf0413/distributedKGE/Embedding"
data_files = ['/fb15k/train.txt', '/fb15k/dev.txt', '/fb15k/test.txt']
num_worker = args.num_worker
niter = args.niter
n_dim = args.ndim
lr = args.lr
margin = args.margin


# 여기서 전처리 C++ 프로그램 비동기 호출
print("Preprocessing start...")
proc = Popen(f"{root_dir}/preprocess/preprocess.out", cwd=f'{root_dir}/preprocess/')


print("read files")
entities = set()
relations = set()
entity2id = dict()
relation2id = dict()
entity_cnt = 0
relations_cnt = 0

for file in data_files:
    with open(root_dir+file, 'r') as f:
        for line in f:
            head, relation, tail = line[:-1].split("\t")
            entities.add(head)
            entities.add(tail)
            relations.add(relation)
            if head not in entity2id:
                entity2id[head] = entity_cnt
                entity_cnt += 1
            if tail not in entity2id:
                entity2id[tail] = entity_cnt
                entity_cnt += 1
            if relation not in relation2id:
                relation2id[relation] = relations_cnt
                relations_cnt += 1


relation_triples = defaultdict(list)
with open(root_dir+data_files[0], 'r') as f:
    for line in f:
        head, relation, tail = line[:-1].split("\t")
        head, relation, tail = entity2id[head], relation2id[relation], entity2id[tail]
        relation_triples[relation].append((head, tail))

relation_each_num = [(k, len(v)) for k, v in relation_triples.items()]
relation_each_num = sorted(relation_each_num, key=lambda x: x[1])
allocated_relation_worker = [[[], 0] for i in range(num_worker)]
for i, (relation, num) in enumerate(relation_each_num):
    allocated_relation_worker = sorted(allocated_relation_worker, key=lambda x: x[1])
    allocated_relation_worker[0][0].append(relation)
    allocated_relation_worker[0][1] += num

sub_graphs = {}
for c, (relation_list, num) in enumerate(allocated_relation_worker):
    g = []
    for relation in relation_list:
        g.append((head, relation, tail))

    sub_graphs[f'sub_graph_worker_{c}'] = pickle.dumps(g, protocol=pickle.HIGHEST_PROTOCOL)


r = redis.StrictRedis(host='163.152.29.73', port=6379, db=0)
r.mset(sub_graphs)

del relation_each_num
del relation_triples
del allocated_relation_worker
del sub_graphs


entities = list(entities)
relations = list(relations)

r.mset(entity2id)
r.mset(relation2id)
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


def work(chunk_data, worker_id, cur_iter, n_dim, lr, margin):
    proc = Popen([
        "python", f"{root_dir}/worker.py", chunk_data,
        str(worker_id), str(cur_iter), str(n_dim), str(lr), str(margin)])
    proc.wait()
    return f"{worker_id}: {cur_iter} iteration finished"


def savePreprocessedData(data, worker_id):
    from threading import Thread
    
    def saveFile(data):
        with open(f"{root_dir}/tmp/data_model_{worker_id}.bin", 'wb') as f:
            f.write(data)

    thread = Thread(target=saveFile, args=(data, ))
    thread.start()
    thread.join()

    return f"{worker_id} finish saving file!"


client = Client('163.152.29.73:8786', asynchronous=True, name='Embedding')
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
proc = Popen(["/home/rudvlf0413/pypy/bin/pypy", 'maxmin.py', str(num_worker), '0'])
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
        workers.append(client.submit(work, chunk_data, worker_id, cur_iter, n_dim, lr, margin))


    if cur_iter % 2 == 1:
        # entity partitioning: max-min cut 실행, anchor 등 재분배
        proc = Popen(["/home/rudvlf0413/pypy/bin/pypy", 'maxmin.py', str(num_worker), str(cur_iter)])
        proc.wait()

        with open(f"{root_dir}/tmp/maxmin_output.txt") as f:
            lines = f.read().splitlines()
            anchors, chunks = lines[0], lines[1:]
    else:
        # relation partitioning
        chunk_data = ''
        

    for worker in as_completed(workers):
        print(worker.result())

print("Totally finished! - Elapsed time: {}".format((time()-t_)))
