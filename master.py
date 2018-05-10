# coding: utf-8
from distributed import Client
from distributed.diagnostics import progress
from sklearn.preprocessing import normalize
from subprocess import Popen
from subprocess import check_output
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
    print("[error] master.py > data root directory must start with /")
    sys.exit(1)

root_dir = args.root_dir


logging.basicConfig(filename='%s/master.log' %
                    root_dir, filemode='w', level=logging.DEBUG)
logger = logging.getLogger()
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)

loggerOn = True

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
        print("[error] master.py > data root mismatch")
        sys.exit(1)

data_root_id = data2id(data_root)

# 여기서 전처리 C++ 프로그램 비동기 호출
master_start = time()
t_ = time()

printt('[info] master.py > Preprocessing started')
proc = Popen(["%spreprocess.out" % preprocess_folder_dir,
              str(data_root_id)], cwd=preprocess_folder_dir)

printt('[info] master.py > Read files')
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

printt('[info] master.py > # of relations per each partitions : [%s]' %
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
    os.system("pip install --upgrade pip")
    os.system("pip install --upgrade redis")
    os.system("pip install --upgrade hiredis")

def work(chunk_data, worker_id, cur_iter, n_dim, lr, margin, train_iter, data_root_id):

    # dask 에 submit 하는 함수에는 logger.warning 을 사용하면 안됨
    socket_port = 50000 + 5 * int(worker_id.split('_')[1]) + (cur_iter % 5)
    print('[info] master.py > work function called, cur_iter = ' + str(cur_iter) + ', port = ' + str(socket_port))

    embedding_proc = Popen([train_code_dir, 
                                    worker_id,
                                    str(cur_iter),
                                    str(n_dim),
                                    str(lr),
                                    str(margin),
                                    str(train_iter),
                                    str(data_root_id),
                                    str(socket_port)],
                                    cwd=preprocess_folder_dir)

    worker_proc = Popen(["python",
                                worker_code_dir,
                                chunk_data,
                                str(worker_id),
                                str(cur_iter),
                                str(n_dim),
                                str(lr),
                                str(margin),
                                str(train_iter),
                                redis_ip_address,
                                root_dir,
                                str(data_root_id),
                                str(socket_port)])

    print('[info] master.py > embedding.cpp, worker.py generated - ' + str(worker_id))

    embedding_proc.wait()
    worker_proc.wait()

    embedding_return = int(embedding_proc.returncode)
    worker_return = int(worker_proc.returncode)

    if embedding_return < 0 or worker_return < 0:

        # embedding.cpp 또는 worker.py 가 비정상 종료
        # 이번 이터레이션을 취소, 한 번 더 수행
        return (False, None)

    else:

        # 모두 성공적으로 수행
        # worker_return 은 string 형태? byte 형태? 의 pickle 을 가지고 있음
        return (True, '[info] master.py > %s: iteration %d finished, time: %f' % (worker_id, cur_iter, time()))

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
proc.communicate()

# with open("%s/data_model.bin" % temp_folder_dir, 'rb') as f:
#     data = f.read()

printt('[info] master.py > preprocessing time : %f' % (time() - t_))

# max-min process 실행, socket 연결
# maxmin.cpp 가 server
# master.py 는 client
anchors = ""
chunks = list()

proc = Popen([pypy_dir,
            'maxmin.py',
            str(num_worker),
            '0',
            str(anchor_num),
            str(anchor_interval),
            root_dir,
            data_root])

printt('[info] master.py > popen maxmin.py complete')

while True:

    try:

        maxmin_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        break
    
    except Exception as e:
    
        tt.sleep(1)
        printt('[error] master.py > exception occured in master <-> maxmin (%s)' % worker_num)
        printt('[error] master.py > ' + str(e))

while True:
    
    try:
    
        maxmin_sock.connect(('127.0.0.1', 7847))
        break
    
    except Exception as e:
    
        tt.sleep(1)
        printt('[error] master.py > exception occured in master <-> maxmin')
        printt('[error] master.py > ' + str(e))

printt('[info] master.py > socket connected (master <-> maxmin)')

maxmin_sock.send(struct.pack('!i', 0))
maxmin_sock.send(struct.pack('!i', num_worker))
maxmin_sock.send(struct.pack('!i', 0))
maxmin_sock.send(struct.pack('!i', anchor_num))
maxmin_sock.send(struct.pack('!i', anchor_interval))

# maxmin 의 결과를 소켓으로 받음
anchor_len = struct.unpack('!i', maxmin_sock.recv(4))[0]
printt('[info] master.py > anchor_len = ' + str(anchor_len))

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

printt('[info] master.py > maxmin finished')
printt('[info] master.py > worker training iteration epoch : {}'.format(train_iter))

cur_iter = 0
trial  = 0
success = False

entities = pickle.loads(r.get('entities'))
relations = pickle.loads(r.get('relations'))
entity_id = r.mget(entities)
relation_id = r.mget(relations)

while True:

    if cur_iter == niter:

        break

    printt('=====================================================================')
    printt('=====================================================================')
    printt('[info] master.py > iteration %d' % cur_iter)

    # 이터레이션이 실패할 경우를 대비해 redis 의 값을 백업
    entities_initialized_bak = r.mget([entity + '_v' for entity in entities])
    entities_initialized_bak = [pickle.loads(v) for v in entities_initialized_bak]
    relations_initialized_bak = r.mget([relation + '_v' for relation in relations])
    relations_initialized_bak = [pickle.loads(v) for v in relations_initialized_bak]

    t_ = time()

    if trial == 5:

        printt('[error] master.py > training failed, exit')
        maxmin_sock.send(struct.pack('!i', 1))
        maxmin_sock.close()
        sys.exit(-1)
    
    if cur_iter > 0 and success == True:
        
        idle_times = [t_ - float(e[1].split(':')[-1]) for e in result_iter]
        avg_idle_time = sum(idle_times) / len(idle_times)
            
        for idx in range(len(result_iter)):

            printt('[info] master.py > %s' % result_iter[idx][1])
            printt('[info] master.py > idle time : %f' % idle_times[idx])
        
        success = False
        printt('[info] master.py > average idle time : %f' % avg_idle_time)

    # 작업 배정
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
        maxmin_sock.send(struct.pack('!i', cur_iter))
        maxmin_sock.send(struct.pack('!i', anchor_num))
        maxmin_sock.send(struct.pack('!i', anchor_interval))

        # maxmin 의 결과를 소켓으로 받음
        anchor_len = struct.unpack('!i', maxmin_sock.recv(4))[0]
        printt('[info] master.py > anchor_len = ' + str(anchor_len))

        for anchor_idx in range(anchor_len):
            
            anchors += str(struct.unpack('!i', maxmin_sock.recv(4))[0]) + ' '
        
        anchors = anchors[:-1]

        for part_idx in range(num_worker):

            chunk = ''
            chunk_len = struct.unpack('!i', maxmin_sock.recv(4))[0]

            for nas_idx in range(chunk_len):
            
                chunk += str(struct.unpack('!i', maxmin_sock.recv(4))[0]) + ' '
            
            chunk = chunk[:-1]
            chunks.append(chunk)

    else:
        # relation partitioning
        chunk_data = ''

    progress(workers)
    printt('')
    result_iter = [worker.result() for worker in workers]

    printt('[info] master.py result array : ' + str([e[0] for e in result_iter]))

    if all([e[0] for e in result_iter]) == True:

        # 이터레이션 성공
        t_ = time()

        for log in [e[1] for e in result_iter]:

            printt(log)

        printt('[info] master.py > iteration %d finished' % cur_iter)
        printt('[info] master.py > iteration time : %f' % (time() - t_))
        success = True
        trial = 0
        cur_iter = cur_iter + 1

    else:

        # 이터레이션 실패
        # redis 에 저장된 결과를 백업된 값으로 되돌림
        trial = trial + 1
        r.mset({entities[i]: pickle.dumps(entities_initialized_bak[i]) for i in range(len(entities_initialized_bak))})
        r.mset({relations[i]: pickle.dumps(relations_initialized_bak[i]) for i in range(len(relations_initialized_bak))})
        
        printt('[error] master.py > iteration %d is failed' % cur_iter)
        printt('[Info] master.py > retry iteration %d' % cur_iter)

# test part
printt('[info] master.py > test start')

# load entity vector
entities_initialized = r.mget([entity + '_v' for entity in entities])
relations_initialized = r.mget([relation + '_v' for relation in relations])

entity_id = {entity: int(entity_id[i]) for i, entity in enumerate(entities)}
relation_id = {relation: int(relation_id[i])
               for i, relation in enumerate(relations)}

entities_initialized = [pickle.loads(v) for v in entities_initialized]
relations_initialized = [pickle.loads(v) for v in relations_initialized]

maxmin_sock.send(struct.pack('!i', 1))
maxmin_sock.close()

###############################################################################
###############################################################################

worker_id = 'worker_0'
proc = Popen([test_code_dir,
            worker_id,
            str(cur_iter),
            str(n_dim),
            str(lr),
            str(margin),
            str(data_root_id)],
            cwd=preprocess_folder_dir)

while True:

    try:

        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        break

    except Exception as e:

        tt.sleep(1)
        printt('[error] master.py > exception occured in master <-> test')
        printt('[error] master.py > ' + str(e))

while True:

    try:

        test_sock.connect(('0.0.0.0', 7874))
        break

    except Exception as e:

        tt.sleep(1)
        printt('[error] master.py > exception occured in master <-> test')
        printt('[error] master.py > ' + str(e))

# DataModel 생성자 -> GeometricModel load 메소드 -> GeometricModel save 메소드 순서로 통신

checksum = 0
success = 0

if int(cur_iter) % 2 == 0:
    # entity 전송 - DataModel 생성자
    #chunk_anchor, chunk_entity = chunk_data.split('\n')
    #chunk_anchor = chunk_anchor.split(' ')
    #chunk_entity = chunk_entity.split(' ')
    chunk_anchor = list()
    chunk_entity = list()

    if len(chunk_anchor) is 1 and chunk_anchor[0] is '':
        
        chunk_anchor = []

    while success != 1:

        test_sock.send(struct.pack('!i', len(chunk_anchor)))

        for iter_anchor in chunk_anchor:
            
            test_sock.send(struct.pack('!i', int(iter_anchor)))

        test_sock.send(struct.pack('!i', len(chunk_entity)))

        for iter_entity in chunk_entity:
            
            test_sock.send(struct.pack('!i', int(iter_entity)))

        checksum = struct.unpack('!i', test_sock.recv(4))[0]

        if checksum == 1234:

            printt('[info] master.py > phase 1 finished (for test)')
            success = 1

        elif checksum == 9876:

            printt('[error] master.py > retry phase 1 (for test)')
            success = 0

        else:

            printt('[error] master.py > unknown error in phase 1 (for test)')
            success = 0

else:
    # relation 전송 - DataModel 생성자
    sub_graphs = pickle.loads(r.get('sub_graph_{}'.format(worker_id)))
    test_sock.send(struct.pack('!i', len(sub_graphs)))

    while success != 1:

        for (head_id, relation_id, tail_id) in sub_graphs:
            test_sock.send(struct.pack('!i', int(head_id)))
            test_sock.send(struct.pack('!i', int(relation_id)))
            test_sock.send(struct.pack('!i', int(tail_id)))

        checksum = struct.unpack('!i', test_sock.recv(4))[0]

        if checksum == 1234:

            printt('[info] master.py > phase 1 finished (for test)')
            success = 1

        elif checksum == 9876:

            printt('[error] master.py > retry phase 1 (for test)')
            success = 0

        else:

            printt('[error] master.py > unknown error in phase 1 (for test)')
            success = 0

printt('[info] master.py > chunk or relation sent to DataModel (for test)')

checksum = 0
success = 0

# entity_vector 전송 - GeometricModel load
while success != 1:

    for i, vector in enumerate(entities_initialized):
        entity_name = str(entities[i])
        #test_sock.send(struct.pack('!i', len(entity_name)))
        #test_sock.send(str.encode(entity_name))    # entity string 자체를 전송


        test_sock.send(struct.pack('!i', entity2id[entity_name])) # entity id 를 int 로 전송


        for v in vector:
            test_sock.send(struct.pack('d', float(v)))

    checksum = struct.unpack('!i', test_sock.recv(4))[0]

    if checksum == 1234:

        printt('[info] master.py > phase 2 (entity) finished (for test)')
        success = 1

    elif checksum == 9876:

        printt('[error] master.py > retry phase 2 (entity) (for test)')
        success = 0

    else:

        printt('[error] master.py > unknown error in phase 2 (entity) (for test)')
        success = 0

printt('[info] master.py > entity_vector sent to GeometricModel load function (for test)')

checksum = 0
success = 0

# relation_vector 전송 - GeometricModel load
while success != 1:

    for i, relation in enumerate(relations_initialized):
        relation_name = str(relations[i])
        #test_sock.send(struct.pack('!i', len(relation_name)))
        #test_sock.send(str.encode(relation_name))  # relation string 자체를 전송


        test_sock.send(struct.pack('!i', relation2id[relation_name])) # relation id 를 int 로 전송


        for v in relation:
            test_sock.send(struct.pack('d', float(v)))

    checksum = struct.unpack('!i', test_sock.recv(4))[0]

    if checksum == 1234:

        printt('[info] master.py > phase 2 (relation) finished (for test)')
        success = 1

    elif checksum == 9876:

        printt('[error] master.py > retry phase 2 (relation) (for test)')
        success = 0

    else:

        printt('[error] master.py > unknown error in phase 2 (relation) (for test)')
        success = 0

printt('[info] master.py > relation_vector sent to Geome tricModel load function (for test)')

del entities_initialized
del relations_initialized

test_return = proc.communicate()

if test_return == -1:

    printt('[error] master.py > test failed, exit')
    sys.exit(-1)
    
printt('[info] master.py > Total elapsed time : %f' % (time() - master_start))