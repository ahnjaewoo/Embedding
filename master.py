# coding: utf-8
from distributed import Client
from distributed.diagnostics import progress
from sklearn.preprocessing import normalize
from subprocess import Popen
from argparse import ArgumentParser
from collections import defaultdict
import logging
import numpy as np
import redis
import pickle
from time import time
import socket
import time as tt
import struct
import sys
import threading

# argument parse
parser = ArgumentParser(description='Distributed Knowledge Graph Embedding')
parser.add_argument('--num_worker', type=int,
                    default=2, help='number of workers')
parser.add_argument('--data_root', type=str, default='/fb15k',
                    help='root directory of data(must include a name of dataset)')
parser.add_argument('--niter', type=int, default=2,
                    help='total number of masters iterations')
parser.add_argument('--train_iter', type=int, default=10,
                    help='total number of workers(actual) training iterations')
parser.add_argument('--install', default='True',
                    help='install libraries in each worker')
parser.add_argument('--ndim', type=int, default=20,
                    help='dimension of embeddings')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--margin', type=int, default=2, help='margin')
parser.add_argument('--anchor_num', type=int, default=5,
                    help='number of anchor during entity training')
parser.add_argument('--anchor_interval', type=int, default=6,
                    help='number of epoch that anchors can rest as non-anchor')
parser.add_argument('--root_dir', type=str,
                    default="/home/rudvlf0413/distributedKGE/Embedding", help='project directory')
parser.add_argument('--temp_dir', type=str, default='', help='temp directory')
parser.add_argument('--pypy_dir', type=str,
                    default="/home/rudvlf0413/pypy/bin/pypy", help='pypy directory')
parser.add_argument('--redis_ip', type=str,
                    default='163.152.29.73', help='redis ip address')
parser.add_argument('--scheduler_ip', type=str,
                    default='163.152.29.73:8786', help='dask scheduler ip:port')
parser.add_argument('--use_scheduler_config_file', default='False',
                    help='wheter to use scheduler config file or use scheduler ip directly')
args = parser.parse_args()

install = args.install
data_root = args.data_root
if data_root[0] != '/':
    print("[error] data root directory must start with /")
    sys.exit(1)

root_dir = args.root_dir


logging.basicConfig(filename='%s/master.log' %
                    root_dir, filemode='w', level=logging.DEBUG)
logger = logging.getLogger()
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)

loggerOn = False

def printt(str):

    global loggerOn

    print(str)

    if loggerOn:
        logger.warning(str + '\n')

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

preprocess_folder_dir = "%s/preprocess/" % root_dir
train_code_dir = "%s/MultiChannelEmbedding/Embedding.out" % root_dir
test_code_dir = "%s/MultiChannelEmbedding/Test.out" % root_dir
worker_code_dir = "%s/worker.py" % root_dir

if args.temp_dir == '':
    temp_folder_dir = "%s/tmp" % root_dir

pypy_dir = args.pypy_dir
redis_ip_address = args.redis_ip
dask_ip_address = args.scheduler_ip
use_scheduler_config_file = args.use_scheduler_config_file

data_files = ['%s/train.txt' % data_root, '%s/dev.txt' %
              data_root, '%s/test.txt' % data_root]
num_worker = args.num_worker
niter = args.niter
train_iter = args.train_iter
n_dim = args.ndim
lr = args.lr
margin = args.margin
anchor_num = args.anchor_num
anchor_interval = args.anchor_interval

def data2id(data_root):
    data_root_split = [x.lower() for x in data_root.split('/')]
    if 'fb15k' in data_root_split:
        return 0
    elif 'wn18' in data_root_split:
        return 1
    elif 'dbpedia' in data_root_split:
        return 2
    else:
        print("[error] data root mismatch")
        sys.exit(1)

data_root_id = data2id(data_root)

# 여기서 전처리 C++ 프로그램 비동기 호출
master_start = time()
t_ = time()

printt('Preprocessing start! - master.py')
proc = Popen(["%spreprocess.out" % preprocess_folder_dir,
              str(data_root_id)], cwd=preprocess_folder_dir)

printt('Read files - master.py')
entities = list()
relations = list()
entity2id = dict()
relation2id = dict()
entity_cnt = 0
relations_cnt = 0

for file in data_files:
    with open(root_dir + file, 'r') as f:
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

with open(root_dir + data_files[0], 'r') as f:
    for line in f:
        head, relation, tail = line[:-1].split("\t")
        head, relation, tail = entity2id[head], relation2id[relation], entity2id[tail]
        relation_triples[relation].append((head, tail))

relation_each_num = [(k, len(v)) for k, v in relation_triples.items()]
relation_each_num = sorted(relation_each_num, key=lambda x: x[1], reverse=True)
allocated_relation_worker = [[[], 0] for i in range(num_worker)]

for i, (relation, num) in enumerate(relation_each_num):
    allocated_relation_worker = sorted(
        allocated_relation_worker, key=lambda x: x[1])
    allocated_relation_worker[0][0].append(relation)
    allocated_relation_worker[0][1] += num

# printing # of relations per each partitions

printt('# of relations per each partitions: [%s]' %
      " ".join([str(len(relation_list)) for relation_list, num in allocated_relation_worker]))

sub_graphs = {}

for c, (relation_list, num) in enumerate(allocated_relation_worker):
    g = []
    for relation in relation_list:
        for (head, tail) in relation_triples[relation]:
            g.append((head, relation, tail))
    sub_graphs['sub_graph_worker_%d' % c] = pickle.dumps(
        g, protocol=pickle.HIGHEST_PROTOCOL)

r = redis.StrictRedis(host=redis_ip_address, port=6379, db=0)
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
    entity + '_v': pickle.dumps(
        entities_initialized[i],
        protocol=pickle.HIGHEST_PROTOCOL) for i, entity in enumerate(entities)})
r.mset({
    relation + '_v': pickle.dumps(
        relations_initialized[i],
        protocol=pickle.HIGHEST_PROTOCOL) for i, relation in enumerate(relations)})

def install_libs():
    import os
    os.system("pip install redis")
    os.system("pip install hiredis")

def work(chunk_data, worker_id, cur_iter, n_dim, lr, margin, train_iter, data_root_id):
    # 첫 iter 에서 embedding.cpp 를 실행해놓음
    
    # dask 에 submit 하는 함수에는 logger.warning 을 사용하면 안됨
    #printt('work function called - master.py')
    print('work function called - master.py')

    if cur_iter == 0:
        proc = Popen([train_code_dir, worker_id,
                      str(cur_iter), str(n_dim), str(lr), str(margin), str(train_iter), str(data_root_id)], cwd=preprocess_folder_dir)
        
        # dask 에 submit 하는 함수에는 logger.warning 을 사용하면 안됨
        #printt('in first iteration, create embedding.cpp process - master.py')
        print('in first iteration, create embedding.cpp process - master.py')
    
    proc = Popen([
        "python", worker_code_dir, chunk_data,
        str(worker_id), str(cur_iter), str(n_dim), str(lr), str(margin), str(train_iter), redis_ip_address, root_dir, str(data_root_id)])
    proc.wait()

    return "%s: %d iteration finished: %f" % (worker_id, cur_iter, time())


# def savePreprocessedData(data, worker_id):
#     from threading import Thread
#     def saveFile(data):
#         with open("%s/data_model_%s.bin" % (temp_folder_dir, worker_id), 'wb') as f:
#             f.write(data)

#     thread = Thread(target=saveFile, args=(data, ))
#     thread.start()
#     thread.join()

#     return "%s finish saving file!" % worker_id

if use_scheduler_config_file == 'True':
    client = Client(scheduler_file=temp_folder_dir +
                    '/scheduler.json', name='Embedding')
else:
    client = Client(dask_ip_address, name='Embedding')

if install == 'True':
    client.run(install_libs)

# 전처리 끝날때까지 대기
proc.wait()

# with open("%s/data_model.bin" % temp_folder_dir, 'rb') as f:
#     data = f.read()

printt('Preprocessing time : %f' % (time() - t_))

# workers = list()

# for i in range(num_worker):
#     worker_id = 'worker_%d' % i
#     workers.append(client.submit(savePreprocessedData, data, worker_id))

# for worker in as_completed(workers):
#     print(worker.result())

# max-min process 실행, socket 연결
# maxmin.cpp 가 server
# master.py 는 client
anchors = ""
chunks = list()

proc = Popen([pypy_dir, 'maxmin.py', str(num_worker),
              '0', str(anchor_num), str(anchor_interval), root_dir, data_root])
printt('popen maxmin.py complete - master.py')

maxmin_addr = '127.0.0.1'
maxmin_port = 7847
tt.sleep(2)

printt('try to connect maxmin socket - master.py')

while True:
    try:
        maxmin_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        break
    except (TimeoutError, ConnectionRefusedError):
        printt('exception occured in master and maxmin connection - master.py')
        tt.sleep(1)

while True:
    try:
        maxmin_sock.connect((maxmin_addr, maxmin_port))
        break
    except (TimeoutError, ConnectionRefusedError):
        printt('exception occured in master and maxmin connection - master.py')
        tt.sleep(1)

printt('socket between master and maxmin connected - master.py')

maxmin_sock.send(struct.pack('!i', 0))
maxmin_sock.send(struct.pack('!i', num_worker))
maxmin_sock.send(struct.pack('!i', 0))
maxmin_sock.send(struct.pack('!i', anchor_num))
maxmin_sock.send(struct.pack('!i', anchor_interval))

# maxmin 의 결과를 소켓으로 받음
anchor_len = struct.unpack('!i', maxmin_sock.recv(4))[0]
printt('anchor_len : ' + str(anchor_len) + ' - master.py')

for anchor_idx in range(anchor_len):
    anchors += str(struct.unpack('!i', maxmin_sock.recv(4))[0]) + " "
anchors = anchors[:-1]

for part_idx in range(num_worker):
    chunk = ""
    chunk_len = struct.unpack('!i', maxmin_sock.recv(4))[0]

    for nas_idx in range(chunk_len):
        chunk += str(struct.unpack('!i', maxmin_sock.recv(4))[0]) + " "
    chunk = chunk[:-1]
    chunks.append(chunk)

printt('maxmin finished - master.py')
printt('worker training iteration epoch: {} - master.py'.format(train_iter))

for cur_iter in range(niter):
    printt('%d iteration - master.py' % cur_iter)

    t_ = time()
    if cur_iter > 0:
        avg_idle_t = 0
        for worker in workers:
            idle_t = t_ - float(worker.result().split(':')[-1])
            avg_idle_t += idle_t
            printt('%s idle time : %f - master.py' % (':'.join(worker.result().split(':')[:2]), idle_t))
        printt('average idle time : %f - master.py' % (avg_idle_t / len(workers)))

    # 작업 배정
    # chunk_data, worker_id, cur_iter, n_dim, lr, margin, train_iter, data_root_id
    workers = [client.submit(work,
                             "{}\n{}".format(anchors, chunks[i]),
                             'worker_%d' % i,
                             cur_iter, n_dim, lr, margin, train_iter,
                             data_root_id
                             ) for i in range(num_worker)]

    if cur_iter % 2 == 1:
        # entity partitioning: max-min cut 실행, anchor 등 재분배
        anchors = ""
        chunks = list()

        # try 가 들어가야 함
        maxmin_sock.send(struct.pack('!i', 0))
        maxmin_sock.send(struct.pack('!i', num_worker))
        # 이 부분은 첫 send 에서는 "0" 으로 교체
        maxmin_sock.send(struct.pack('!i', cur_iter))
        maxmin_sock.send(struct.pack('!i', anchor_num))
        maxmin_sock.send(struct.pack('!i', anchor_interval))

        # maxmin 의 결과를 소켓으로 받음
        anchor_len = struct.unpack('!i', maxmin_sock.recv(4))[0]
        logger.warning(str(anchor_len)+'\n')

        for anchor_idx in range(anchor_len):
            anchors += str(struct.unpack('!i',
                                         maxmin_sock.recv(4))[0]) + " "
        anchors = anchors[:-1]

        for part_idx in range(num_worker):
            chunk = ""
            chunk_len = struct.unpack('!i', maxmin_sock.recv(4))[0]

            for nas_idx in range(chunk_len):
                chunk += str(struct.unpack('!i',
                                           maxmin_sock.recv(4))[0]) + " "
            chunk = chunk[:-1]
            chunks.append(chunk)

    else:
        # relation partitioning
        chunk_data = ''

    progress(workers)

    for worker in workers:
        printt(worker.result() + ' - master.py')

    printt('iteration time : %f - master.py' % (time() - t_))

# test part
printt('test start - master.py')

# load entity vector
entities = pickle.loads(r.get('entities'))
relations = pickle.loads(r.get('relations'))
entity_id = r.mget(entities)
relation_id = r.mget(relations)
entities_initialized = r.mget([entity + '_v' for entity in entities])
relations_initialized = r.mget([relation + '_v' for relation in relations])

entity_id = {entity: int(entity_id[i]) for i, entity in enumerate(entities)}
relation_id = {relation: int(relation_id[i])
               for i, relation in enumerate(relations)}

entities_initialized = [pickle.loads(v) for v in entities_initialized]
relations_initialized = [pickle.loads(v) for v in relations_initialized]

worker_id = 'worker_0'
proc = Popen([
    test_code_dir,
    worker_id, str(cur_iter), str(n_dim), str(lr), str(margin), str(data_root_id)],
    cwd=preprocess_folder_dir)

maxmin_sock.send(struct.pack('!i', 1))
maxmin_sock.close()

test_addr = '0.0.0.0'
test_port = 7874  # 임의로 7874 로 포트를 정함
tt.sleep(2)

while True:
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        break
    except (TimeoutError, ConnectionRefusedError):
        tt.sleep(1)

while True:
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        break
    except (TimeoutError, ConnectionRefusedError):
        tt.sleep(1)

test_sock.send(struct.pack('!i', 0))                        # 연산 요청 메시지
# int 임시 땜빵, 매우 큰 문제
test_sock.send(struct.pack('!i', int(worker_id.split('_')[1])))
test_sock.send(struct.pack('!i', int(cur_iter)))            # int
test_sock.send(struct.pack('!i', int(n_dim)))       # int
test_sock.send(struct.pack('d', float(lr)))      # double
test_sock.send(struct.pack('d', float(margin)))             # double
test_sock.send(struct.pack('!i', int(data_root_id)))    # int

# DataModel 생성자 -> GeometricModel load 메소드 -> GeometricModel save 메소드 순서로 통신

checksum = 0

if int(cur_iter) % 2 == 0:
    # entity 전송 - DataModel 생성자
    chunk_anchor, chunk_entity = chunk_data.split('\n')
    chunk_anchor = chunk_anchor.split(' ')
    chunk_entity = chunk_entity.split(' ')

    if len(chunk_anchor) is 1 and chunk_anchor[0] is '':
        chunk_anchor = []

    while checksum != 1:

        test_sock.send(struct.pack('!i', len(chunk_anchor)))

        for iter_anchor in chunk_anchor:
            test_sock.send(struct.pack('!i', int(iter_anchor)))

        test_sock.send(struct.pack('!i', len(chunk_entity)))

        for iter_entity in chunk_entity:
            test_sock.send(struct.pack('!i', int(iter_entity)))

        checksum = struct.unpack('!i', test_sock.recv(4))[0]

        if checksum == 1234:

            printt('phase 1 finished - master.py (for test)')
            checksum = 1

        elif checksum == 9876:

            printt('retry phase 1 - master.py (for test)')
            checksum = 0

        else:

            printt('unknown error in phase 1 - master.py (for test)')
            checksum = 0

else:
    # relation 전송 - DataModel 생성자
    sub_graphs = pickle.loads(r.get('sub_graph_{}'.format(worker_id)))
    test_sock.send(struct.pack('!i', len(sub_graphs)))

    while checksum != 1:

        for (head_id, relation_id, tail_id) in sub_graphs:
            test_sock.send(struct.pack('!i', int(head_id)))
            test_sock.send(struct.pack('!i', int(relation_id)))
            test_sock.send(struct.pack('!i', int(tail_id)))

        checksum = struct.unpack('!i', test_sock.recv(4))[0]

        if checksum == 1234:

            printt('phase 1 finished - master.py (for test)')
            checksum = 1

        elif checksum == 9876:

            printt('retry phase 1 - master.py (for test)')
            checksum = 0

        else:

            printt('unknown error in phase 1 - master.py (for test)')
            checksum = 0

printt('chunk or relation sent to DataModel - master.py (for test)')

checksum = 0

# entity_vector 전송 - GeometricModel load
while checksum != 1:

    for i, vector in enumerate(entities_initialized):
        entity_name = str(entities[i])
        test_sock.send(struct.pack('!i', len(entity_name)))
        test_sock.send(str.encode(entity_name))    # entity string 자체를 전송


        #test_sock.send(struct.pack('!i', entity2id[entity_name])) # entity id 를 int 로 전송


        for v in vector:
            test_sock.send(struct.pack('d', float(v)))

    checksum = struct.unpack('!i', test_sock.recv(4))[0]

    if checksum == 1234:

        printt('phase 2 (entity) finished - master.py (for test)')
        checksum = 1

    elif checksum == 9876:

        printt('retry phase 2 (entity) - master.py (for test)')
        checksum = 0

    else:

        printt('unknown error in phase 2 (entity) - master.py (for test)')
        checksum = 0

printt('entity_vector sent to GeometricModel load function - master.py (for test)')

checksum = 0

# relation_vector 전송 - GeometricModel load
while checksum != 1:

    for i, relation in enumerate(relations_initialized):
        relation_name = str(relations[i])
        test_sock.send(struct.pack('!i', len(relation_name)))
        test_sock.send(str.encode(relation_name))  # relation string 자체를 전송


        #test_sock.send(struct.pack('!i', relation2id[relation_name])) # relation id 를 int 로 전송


        for v in relation:
            test_sock.send(struct.pack('d', float(v)))

    checksum = struct.unpack('!i', test_sock.recv(4))[0]

    if checksum == 1234:

        printt('phase 2 (relation) finished - master.py (for test)')
        checksum = 1

    elif checksum == 9876:

        printt('retry phase 2 (relation) - master.py (for test)')
        checksum = 0

    else:

        printt('unknown error in phase 2 (relation) - master.py (for test)')
        checksum = 0

printt('relation_vector sent to Geome tricModel load function - master.py (for test)')

del entities_initialized
del relations_initialized

proc.wait()
printt('Total elapsed time: %f - master.py' % (time() - master_start))