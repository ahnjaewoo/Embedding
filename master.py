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


import socket
import time as tt
import struct

parser = ArgumentParser(description='Distributed Knowledge Graph Embedding')
parser.add_argument('--num_worker', type=int, default=2, help='number of workers')
parser.add_argument('--data_root', type=str, default='./fb15k', help='root directory of data')
parser.add_argument('--niter', type=int, default=2, help='total number of training iterations')
parser.add_argument('--install', default=False, help='install libraries in each worker')
parser.add_argument('--ndim', type=int, default=20, help='dimension of embeddings')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--margin', type=int, default=2, help='margin')
parser.add_argument('--anchor_num', type=int, default=5, help='number of anchor during entity training')
parser.add_argument('--anchor_interval', type=int, default=6, help='number of epoch that anchors can rest as non-anchor')
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
anchor_num = args.anchor_num
anchor_interval = args.anchor_interval
use_socket = False

# 여기서 전처리 C++ 프로그램 비동기 호출
t_ = time()
print("Preprocessing start...")
proc = Popen(f"{root_dir}/preprocess/preprocess.out", cwd=f'{root_dir}/preprocess/')


print("read files")
entities = list()
relations = list()
entity2id = dict()
relation2id = dict()
entity_cnt = 0
relations_cnt = 0

for file in data_files:
    with open(root_dir+file, 'r') as f:
        for line in f:
            head, relation, tail = line[:-1].split("\t")
            if head not in entity2id:
            	entities.append(head)
            	entity2id[head] = entity_cnt
            	entity_cnt += 1
            if tail not in entity2id:
            	entities.append(tail)
            	entity2id[tail] = entity_cnt
            	entity_cnt += 1
            if relation not in relation2id:
            	relations.append(relation)
            	relation2id[relation] = relations_cnt
            	relations_cnt += 1


relation_triples = defaultdict(list)
with open(root_dir+data_files[0], 'r') as f:
    for line in f:
        head, relation, tail = line[:-1].split("\t")
        head, relation, tail = entity2id[head], relation2id[relation], entity2id[tail]
        relation_triples[relation].append((head, tail))

relation_each_num = [(k, len(v)) for k, v in relation_triples.items()]
relation_each_num = sorted(relation_each_num, key=lambda x: x[1], reverse=True)
allocated_relation_worker = [[[], 0] for i in range(num_worker)]
for i, (relation, num) in enumerate(relation_each_num):
    allocated_relation_worker = sorted(allocated_relation_worker, key=lambda x: x[1])
    allocated_relation_worker[0][0].append(relation)
    allocated_relation_worker[0][1] += num

#printing # of relations per each partitions
print('# of relations per each partitions: [%s]' %
    " ".join([str(len(relation_list)) for relation_list, num in allocated_relation_worker]))

sub_graphs = {}
for c, (relation_list, num) in enumerate(allocated_relation_worker):
    g = []
    for relation in relation_list:
        for (head, tail) in relation_triples[relation]:
            g.append((head, relation, tail))
    sub_graphs[f'sub_graph_worker_{c}'] = pickle.dumps(g, protocol=pickle.HIGHEST_PROTOCOL)

r = redis.StrictRedis(host='163.152.29.73', port=6379, db=0)
r.mset(sub_graphs)

del relation_each_num
del relation_triples
del allocated_relation_worker
del sub_graphs


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


def work(chunk_data, worker_id, cur_iter, n_dim, lr, margin, is_final):
    
    # 첫 iter 에서 embedding.cpp 를 실행해놓음
    # 
    if use_socket and cur_iter == 0:
        proc = Popen([f"{root_dir}/MultiChannelEmbedding/Embedding.out", worker_id, \
            str(cur_iter), str(n_dim), str(lr), str(margin), str(is_final)], cwd=f'{root_dir}/preprocess/')

    proc = Popen([
        "python", f"{root_dir}/worker.py", chunk_data,
        str(worker_id), str(cur_iter), str(n_dim), str(lr), str(margin), str(is_final)])
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

print("preprocessing time: %f" % (time()-t_))

workers = []
for i in range(num_worker):
    worker_id = f'worker_{i}'
    workers.append(client.submit(savePreprocessedData, data, worker_id))

for worker in as_completed(workers):
    print(worker.result())

# max-min process 실행, socket 연결
# maxmin.cpp 가 server
# master.py 는 client
if use_socket:

    proc = Popen(["/home/rudvlf0413/pypy/bin/pypy", 'maxmin.py', str(num_worker), '0', str(anchor_num), str(anchor_interval)])
    tt.sleep(3)

    maxmin_addr = '127.0.0.1'
    maxmin_port = 7847
    maxmin_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    maxmin_sock.connect((maxmin_addr, maxmin_port))

# max-min process 의 socket 으로 anchor 분배, 실행
if use_socket:

    # try 가 들어가야 함 line 184 ~ 239 까지를 감쌈

    maxmin_sock.send(b'0')
    maxmin_sock.send(struct.pack('!i', num_worker))
    maxmin_sock.send(struct.pack('!i', 0))     # 이 부분은 첫 send 에서는 "0" 으로 교체
    maxmin_sock.send(struct.pack('!i', anchor_num))
    maxmin_sock.send(struct.pack('!i', anchor_interval))

    # 원래 maxmin_output.txt 로 받았던 결과를 socket 으로 다시 받을 수 있음, socket 과 파일의 결과 전달 속도를 비교할 필요가 있음
    # socket 으로 결과를 전달한다면 list type 을 string 으로 바꿔 보내고 recv 후 eval 하면 됨
    # recv 의 인자로 들어가는 수는 read 할 데이터 길이, byte 단위
    # 그리고 recv 에서의 길이에 주의, 결과를 담은 문자열의 길이가 얼마나 될 지 모름
    # 만약 이것이 너무 길다면, 여러 번에 걸쳐서 나눠 받고 합쳐서 사용해야 함
    # 하지만 master.py 가 maxmin.py 에 보내는 인자의 문자열 길이는 짧아서 괜찮음
    # anchors = eval(maxmin_sock.recv(1024))
    # chuncks = eval(maxmin_sock.recv(1024))







    # maxmin 의 결과를 파일로 전송하지 않고 소켓으로 전송하면 아래 줄은 필요 없음
    maxmin_iter_end = maxmin_sock.recv(1).decode()



    """
    # writing output file
    with open(output_file, "w") as fwrite:
        fwrite.write(" ".join([str(i) for i in anchor])+"\n")
        print(len(" ".join([str(i) for i in anchor])))
        for nas in parts:
            fwrite.write(" ".join([str(i) for i in nas])+"\n")
            print(len(" ".join([str(i) for i in nas])))
    """




    # maxmin 의 결과를 파일로 전송하지 않고 소켓으로 전송
    with open(f"{root_dir}/tmp/maxmin_output.txt") as f:
        lines = f.read().splitlines()
        anchors, chunks = lines[0], lines[1:]


# line 201 ~ 205 을 line 181 ~ 196 의 socket 통신으로 대체
# max-min cut 실행, anchor 분배
if not use_socket:
    proc = Popen(["/home/rudvlf0413/pypy/bin/pypy", 'maxmin.py', str(num_worker), '0', str(anchor_num), str(anchor_interval)])
    proc.wait()
    with open(f"{root_dir}/tmp/maxmin_output.txt") as f:
        lines = f.read().splitlines()
        anchors, chunks = lines[0], lines[1:]













for cur_iter in range(niter):
    t_ = time()

    # 작업 배정
    workers = []
    for i in range(num_worker):
        worker_id = f'worker_{i}'
        chunk_data = "{}\n{}".format(anchors, chunks[i])

        # work 함수가 mapping 되어 실행될 때, worker.py 프로세스를 새로 실행
        # worker.py 는 다시 embedding.cpp 프로세스를 새로 실행
        # embedding.cpp 프로세스가 새로 실행되던 걸 개선하더라도 worker.py 가 계속 바뀌므로 socket 을 새로 연결하는 오버헤드가 있음
        # 또한 이 상태에선 embedding.cpp 프로세스를 생성하는 주체가 master 가 되어야 하는데, 연결하는 주체는 worker.py 이므로 애매함이 좀 있음
        # worker.py 와 embedding.cpp 간의 socket 에 관한 addr, port 를 관리해야 함


        # embedding_sock_setting[i] 를 함께 전달
        if cur_iter < niter - 1:
            workers.append(client.submit(work, chunk_data, worker_id, cur_iter, n_dim, lr, margin, 0))
            #workers.append(client.submit(work, chunk_data, worker_id, cur_iter, n_dim, lr, margin, 0, embedding_sock_setting[i][0], embedding_sock_setting[i][1]))
        else:
            workers.append(client.submit(work, chunk_data, worker_id, cur_iter, n_dim, lr, margin, 1))
            #workers.append(client.submit(work, chunk_data, worker_id, cur_iter, n_dim, lr, margin, 1, embedding_sock_setting[i][0], embedding_sock_setting[i][1]))

    if cur_iter % 2 == 1:

        # line 227 ~ 233 을 line 181 ~ 193 의 socket 통신으로 대체
        # entity partitioning: max-min cut 실행, anchor 등 재분배
        if not use_socket:

            proc = Popen(["/home/rudvlf0413/pypy/bin/pypy", 'maxmin.py', str(num_worker), str(cur_iter), str(anchor_num), str(anchor_interval)])
            proc.wait()

            with open(f"{root_dir}/tmp/maxmin_output.txt") as f:
                lines = f.read().splitlines()
                anchors, chunks = lines[0], lines[1:]

        else:

            maxmin_sock.send(b'0')
            maxmin_sock.send(struct.pack('!i', num_worker))
            maxmin_sock.send(struct.pack('!i', cur_iter))     # 이 부분은 첫 send 에서는 "0" 으로 교체
            maxmin_sock.send(struct.pack('!i', anchor_num))
            maxmin_sock.send(struct.pack('!i', anchor_interval))



            # maxmin 결과를 소켓으로 전송하면 아래 줄은 필요 없음
            maxmin_iter_end = maxmin_sock.recv(1).decode()

            #if maxmin_iter_end == '0':

            with open(f"{root_dir}/tmp/maxmin_output.txt") as f:
                lines = f.read().splitlines()
                anchors, chunks = lines[0], lines[1:]


    else:
        # relation partitioning
        chunk_data = ''

    for worker in as_completed(workers):
        print(worker.result())

    print("iteration time: %f" % (time()-t_))

# except KeyboardInterrupt:
#   maxmin_sock.close()

# maxmin.py 과의 socket 을 close
# socket 을 사용하는 코드 전체를 try except 로 감싸고 close 를 한 번 더 사용해줘야 함 (비정상 종료 때문)
# finally:
# maxmin_sock.close()

if use_socket:

    maxmin_sock.send(b'1')
    maxmin_sock.close()
