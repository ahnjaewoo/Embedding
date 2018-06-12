# coding: utf-8
from distributed import Client
from sklearn.preprocessing import normalize
from subprocess import Popen
from argparse import ArgumentParser
from collections import defaultdict
from zlib import compress
from zlib import decompress
import logging
import numpy as np
import redis
import pickle
from time import time
from time import sleep
import socket
import timeit
import struct
import sys
import os


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
parser.add_argument('--debugging', type=str, default='yes', help='debugging mode or not')
args = parser.parse_args()


if args.debugging == 'yes':

    logging.basicConfig(filename='%s/master.log' % args.root_dir, filemode='w', level=logging.DEBUG)
    logger = logging.getLogger()
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)
    
    def printt(str):

        print(str)
        logger.warning(str + '\n')

    def handle_exception(exc_type, exc_value, exc_traceback):

        if issubclass(exc_type, KeyboardInterrupt):

            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

elif args.debugging == 'no':
    
    printt = print


def sockRecv(sock, length):

    data = b''

    while len(data) < length:

        buff = sock.recv(length - len(data))

        if not buff:

            return None

        data = data + buff

    return data

def data2id(data_root):

    data_root_split = [x.lower() for x in data_root.split('/')]

    if 'fb15k' in data_root_split:

        return 0

    elif 'wn18' in data_root_split:

        return 1

    elif 'dbpedia' in data_root_split:

        return 2

    else:

        print("[error] master > data root mismatch")
        sys.exit(1)

def install_libs():

    import os
    from pkgutil import iter_modules

    modules = set([m[1] for m in iter_modules()])

    if 'redis' not in modules:
        os.system("pip install --upgrade redis")
    if 'hiredis' not in modules:
        os.system("pip install --upgrade hiredis")


def work(chunk_data, worker_id, cur_iter, n_dim, lr, margin, train_iter, data_root_id, redis_ip, root_dir, debugging):

    # dask 에 submit 하는 함수에는 logger.warning 을 사용하면 안됨
    socket_port = 50000 + 5 * int(worker_id.split('_')[1]) + (cur_iter % 5)
    # print('master > work function called, cur_iter = ' + str(cur_iter) + ', port = ' + str(socket_port))
    log_dir = os.path.join(args.root_dir, 'logs/embedding_log_{}_iter_{}.txt'.format(worker_id, cur_iter))

    workStart = timeit.default_timer()

    embedding_proc = Popen([train_code_dir, 
                            worker_id,
                            str(cur_iter),
                            str(n_dim),
                            str(lr),
                            str(margin),
                            str(train_iter),
                            str(data_root_id),
                            str(socket_port),
                            log_dir],
                            cwd=preprocess_folder_dir)

    worker_proc = Popen(["python",
                         worker_code_dir,
                         chunk_data,
                         worker_id,
                         str(cur_iter),
                         str(n_dim),
                         redis_ip,
                         root_dir,
                         str(socket_port),
                         debugging])

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
        timeNow = timeit.default_timer()
        return (True, timeNow - workStart)

if args.data_root[0] != '/':

    printt("[error] master > data root directory must start with /")
    sys.exit(1)

preprocess_folder_dir = "%s/preprocess/" % args.root_dir
train_code_dir = "%s/MultiChannelEmbedding/Embedding.out" % args.root_dir
test_code_dir = "%s/MultiChannelEmbedding/Test.out" % args.root_dir
worker_code_dir = "%s/worker.py" % args.root_dir

if args.temp_dir == '':

    temp_folder_dir = "%s/tmp" % args.root_dir

data_files = ['%s/train.txt' % args.data_root, '%s/dev.txt' % args.data_root, '%s/test.txt' % args.data_root]
num_worker = args.num_worker
niter = args.niter
train_iter = args.train_iter
n_dim = args.ndim
lr = args.lr
margin = args.margin
anchor_num = args.anchor_num
anchor_interval = args.anchor_interval

entities = list()
relations = list()
entity2id = dict()
relation2id = dict()
entity_cnt = 0
relations_cnt = 0
data_root_id = data2id(args.data_root)

masterStart = timeit.default_timer()
# 여기서 전처리 C++ 프로그램 비동기 호출
proc = Popen(["%spreprocess.out" % preprocess_folder_dir,
              str(data_root_id)], cwd=preprocess_folder_dir)
# printt('master > Preprocessing started')

for file in data_files:

    with open(args.root_dir + file, 'r') as f:

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

with open(args.root_dir + data_files[0], 'r') as f:

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

# printt('master > # of relations per each partitions : [%s]' %
#       " ".join([str(len(relation_list)) for relation_list, num in allocated_relation_worker]))

sub_graphs = {}

for c, (relation_list, num) in enumerate(allocated_relation_worker):

    g = []
    for relation in relation_list:
        for (head, tail) in relation_triples[relation]:
            g.append((head, relation, tail))
    sub_graphs['sub_g_worker_%d' % c] = compress(pickle.dumps(
        g, protocol=pickle.HIGHEST_PROTOCOL), 9)

r = redis.StrictRedis(host = args.redis_ip, port = 6379, db = 0)
r.mset(sub_graphs)

del relation_each_num
del relation_triples
del allocated_relation_worker
del sub_graphs

r.mset(entity2id)
r.mset(relation2id)

r.set('entities', compress(pickle.dumps(entities, protocol=pickle.HIGHEST_PROTOCOL), 9))
r.set('relations', compress(pickle.dumps(relations, protocol=pickle.HIGHEST_PROTOCOL), 9))

entities_initialized = normalize(np.random.randn(len(entities), n_dim))
relations_initialized = normalize(np.random.randn(len(relations), n_dim))

r.mset({
    entity + '_v': compress(pickle.dumps(
        entities_initialized[i],
        protocol=pickle.HIGHEST_PROTOCOL), 9) for i, entity in enumerate(entities)})
r.mset({
    relation + '_v': compress(pickle.dumps(
        relations_initialized[i],
        protocol=pickle.HIGHEST_PROTOCOL), 9) for i, relation in enumerate(relations)})

if args.use_scheduler_config_file == 'True':

    client = Client(scheduler_file=temp_folder_dir + '/scheduler.json', name='Embedding')

else:

    client = Client(args.scheduler_ip, name='Embedding')

if args.install == 'True':

    client.run(install_libs)

# 전처리 끝날때까지 대기
proc.communicate()
preprocessingTime = timeit.default_timer() - masterStart
printt('master > preprocessing time : %f' % preprocessingTime)

maxminTimes = list()
iterTimes = list()

# max-min process 실행, socket 연결
# maxmin.cpp 가 server
# master.py 는 client
anchors = ""
chunks = list()

proc = Popen([args.pypy_dir,
            'maxmin.py',
            str(num_worker),
            '0',
            str(anchor_num),
            str(anchor_interval),
            args.root_dir,
            args.data_root,
            args.debugging])

while True:

    try:

        maxmin_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        break
    
    except Exception as e:
    
        sleep(1)
        printt('[error] master > exception occured in master <-> maxmin')
        printt('[error] master > ' + str(e))

while True:
    
    try:
    
        maxmin_sock.connect(('127.0.0.1', 7847))
        printt('master > maxmin connection succeed')
        break
    
    except Exception as e:
    
        sleep(1)
        printt('[error] master > exception occured in master <-> maxmin')
        printt('[error] master > ' + str(e))

# printt('master > socket connected (master <-> maxmin)')

timeNow = timeit.default_timer()
maxmin_sock.send(struct.pack('!i', 0))
#maxmin_sock.send(struct.pack('!i', num_worker))
maxmin_sock.send(struct.pack('!i', 0))
#maxmin_sock.send(struct.pack('!i', anchor_num))
#maxmin_sock.send(struct.pack('!i', anchor_interval))

# maxmin 의 결과를 소켓으로 받음
anchor_len = struct.unpack('!i', sockRecv(maxmin_sock, 4))[0]

for _ in range(anchor_len):

    anchors += str(struct.unpack('!i', sockRecv(maxmin_sock, 4))[0]) + " "

anchors = anchors[:-1]

for _ in range(num_worker):

    chunk = ""
    chunk_len = struct.unpack('!i', sockRecv(maxmin_sock, 4))[0]

    for _ in range(chunk_len):

        chunk += str(struct.unpack('!i', sockRecv(maxmin_sock, 4))[0]) + " "
    
    chunk = chunk[:-1]
    chunks.append(chunk)

maxminTimes.append(timeit.default_timer() - timeNow)

# printt('master > maxmin finished')
# printt('master > worker training iteration epoch : {}'.format(train_iter))

cur_iter = 0
trial  = 0
success = False

entities = pickle.loads(decompress(r.get('entities')))
relations = pickle.loads(decompress(r.get('relations')))

trainStart = timeit.default_timer()

while True:

    # 이터레이션이 실패할 경우를 대비해 redis 의 값을 백업
    # entities_initialized_bak = r.mget([entity + '_v' for entity in entities])
    # entities_initialized_bak = [pickle.loads(v) for v in entities_initialized_bak]
    # relations_initialized_bak = r.mget([relation + '_v' for relation in relations])
    # relations_initialized_bak = [pickle.loads(v) for v in relations_initialized_bak]

    if cur_iter == niter:

        break

    if trial == 5:

        printt('[error] master > training failed, exit')
        maxmin_sock.send(struct.pack('!i', 1))
        maxmin_sock.close()
        sys.exit(-1)

    # 작업 배정
    printt('master > iteration %d' % cur_iter)
    iterStart = timeit.default_timer()
    
    workers = [client.submit(work,
                             "{}\n{}".format(anchors, chunks[i]),
                             'worker_%d' % i,
                             cur_iter, n_dim, lr, margin, train_iter,
                             data_root_id, args.redis_ip, args.root_dir, args.debugging
                             ) for i in range(num_worker)]

    if cur_iter % 2 == 1:
        # entity partitioning: max-min cut 실행, anchor 등 재분배
        anchors = ""
        chunks = list()

        maxminStart = timeit.default_timer()

        # try 가 들어가야 함
        maxmin_sock.send(struct.pack('!i', 0))
        #maxmin_sock.send(struct.pack('!i', num_worker))
        maxmin_sock.send(struct.pack('!i', cur_iter))
        #maxmin_sock.send(struct.pack('!i', anchor_num))
        #maxmin_sock.send(struct.pack('!i', anchor_interval))

        # maxmin 의 결과를 소켓으로 받음
        anchor_len = struct.unpack('!i', sockRecv(maxmin_sock, 4))[0]
        # printt('master > anchor_len = ' + str(anchor_len))

        for _ in range(anchor_len):
            
            anchors += str(struct.unpack('!i', sockRecv(maxmin_sock, 4))[0]) + ' '
        
        anchors = anchors[:-1]

        for _ in range(num_worker):

            chunk = ''
            chunk_len = struct.unpack('!i', sockRecv(maxmin_sock, 4))[0]

            for _ in range(chunk_len):
            
                chunk += str(struct.unpack('!i', sockRecv(maxmin_sock, 4))[0]) + ' '
            
            chunk = chunk[:-1]
            chunks.append(chunk)

        maxminTimes.append(timeit.default_timer() - maxminStart)

    else:
        # relation partitioning
        chunk_data = ''
    
    client.gather(workers)
    result_iter = [worker.result() for worker in workers]
    iterTimes.append(timeit.default_timer() - iterStart)

    if all([e[0] for e in result_iter]) == True:

        # 이터레이션 성공
        printt('master > iteration time : %f' % (timeit.default_timer() - timeNow))
        success = True
        trial = 0
        cur_iter = cur_iter + 1

        workTimes = [e[1] for e in result_iter]

        # embedding.cpp 에서 model->run() 실행 시간을 worker.py 로 전송해서 그걸 소켓으로 전송

        printt('master > Total embedding times : ' + str(workTimes))
        # printt('master > Average total embedding time : ' + str(np.mean(workTimes)))

    else:

        # 이터레이션 실패
        # redis 에 저장된 결과를 백업된 값으로 되돌림
        trial = trial + 1
        # r.mset({str(entities[i]) + '_bak' : pickle.dumps(entities_initialized_bak[i], protocol=pickle.HIGHEST_PROTOCOL) for i in range(len(entities_initialized_bak))})
        # r.mset({str(relations[i]) + '_bak' : pickle.dumps(relations_initialized_bak[i], protocol=pickle.HIGHEST_PROTOCOL) for i in range(len(relations_initialized_bak))})
        printt('[error] master > iteration %d is failed, retry' % cur_iter)
        
trainTime = timeit.default_timer() - trainStart

# test part
# printt('master > test start')

# load entity vector
entities_initialized = r.mget([entity + '_v' for entity in entities])
relations_initialized = r.mget([relation + '_v' for relation in relations])

entities_initialized = [pickle.loads(decompress(v)) for v in entities_initialized]
relations_initialized = [pickle.loads(decompress(v)) for v in relations_initialized]

maxmin_sock.send(struct.pack('!i', 1))
maxmin_sock.close()

###############################################################################
###############################################################################

worker_id = 'worker_0'
log_dir = os.path.join(args.root_dir, 'logs/test_log.txt')
proc = Popen([test_code_dir,
            worker_id,
            '0',
            str(n_dim),
            str(lr),
            str(margin),
            str(data_root_id),
            str(log_dir)],
            cwd=preprocess_folder_dir)

while True:

    try:

        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        break

    except Exception as e:

        sleep(1)
        printt('[error] master > exception occured in master <-> test')
        printt('[error] master > ' + str(e))

while True:

    try:

        test_sock.connect(('127.0.0.1', 7874))
        break

    except Exception as e:

        sleep(1)
        printt('[error] master > exception occured in master <-> test')
        printt('[error] master > ' + str(e))

# DataModel 생성자 -> GeometricModel load 메소드 -> GeometricModel save 메소드 순서로 통신
# entity 전송 - DataModel 생성자
chunk_anchor = list()
chunk_entity = list()

checksum = 0
success = 0

# entity 전송 - DataModel 생성자
chunk_anchor = list()
chunk_entity = list()

if len(chunk_anchor) == 1 and chunk_anchor[0] == '':
    
    chunk_anchor = []

while success != 1:

    test_sock.send(struct.pack('!i', len(chunk_anchor)))

    for iter_anchor in chunk_anchor:
        
        test_sock.send(struct.pack('!i', int(iter_anchor)))

    test_sock.send(struct.pack('!i', len(chunk_entity)))

    for iter_entity in chunk_entity:
        
        test_sock.send(struct.pack('!i', int(iter_entity)))

    checksum = struct.unpack('!i', sockRecv(test_sock, 4))[0]

    if checksum == 1234:

        # printt('master > phase 1 finished (for test)')
        success = 1

    elif checksum == 9876:

        printt('[error] master > retry phase 1 (for test)')
        success = 0

    else:

        printt('[error] master > unknown error in phase 1 (for test)')
        success = 0

printt('master > chunk or relation sent to DataModel (for test)')

checksum = 0
success = 0

# entity_vector 전송 - GeometricModel load
while success != 1:

    for i, vector in enumerate(entities_initialized):
        entity_name = str(entities[i])
        #test_sock.send(struct.pack('!i', len(entity_name)))         # entity string 자체를 전송
        #test_sock.send(str.encode(entity_name))                     # entity string 자체를 전송

        test_sock.send(struct.pack('!i', entity2id[entity_name]))  # entity id 를 int 로 전송

        for v in vector:

            test_sock.send(struct.pack('f', float(v)))

    checksum = struct.unpack('!i', sockRecv(test_sock, 4))[0]

    if checksum == 1234:

        # printt('master > phase 2 (entity) finished (for test)')
        success = 1

    elif checksum == 9876:

        printt('[error] master > retry phase 2 (entity) (for test)')
        success = 0

    else:

        printt('[error] master > unknown error in phase 2 (entity) (for test)')
        success = 0

# printt('master > entity_vector sent to GeometricModel load function (for test)')

checksum = 0
success = 0

# relation_vector 전송 - GeometricModel load
while success != 1:

    for i, relation in enumerate(relations_initialized):
        relation_name = str(relations[i])
        #test_sock.send(struct.pack('!i', len(relation_name)))           # relation string 자체를 전송
        #test_sock.send(str.encode(relation_name))                       # relation string 자체를 전송

        test_sock.send(struct.pack('!i', relation2id[relation_name]))  # relation id 를 int 로 전송

        for v in relation:

            test_sock.send(struct.pack('f', float(v)))

    checksum = struct.unpack('!i', sockRecv(test_sock, 4))[0]

    if checksum == 1234:

        printt('master > phase 2 (relation) finished (for test)')
        success = 1

    elif checksum == 9876:

        printt('[error] master > retry phase 2 (relation) (for test)')
        success = 0

    else:

        printt('[error] master > unknown error in phase 2 (relation) (for test)')
        success = 0


# printt('master > relation_vector sent to Geome tricModel load function (for test)')

del entities_initialized
del relations_initialized

test_return = proc.communicate()

if test_return == -1:

    printt('[error] master > test failed, exit')
    sys.exit(-1)

totalTime = timeit.default_timer() - masterStart
printt('master > Total elapsed time : %f' % (totalTime))

workerLogKeys = ['worker_' + str(n) + '_' + str(i) for i in range(niter) for n in range(num_worker)]
workerLogs = r.mget(workerLogKeys)

redisConnTime = list()
datamodelTime = list()
sockLoadTime = list()
embeddingTime = list()
modelRunTime = list()
sockSaveTime = list()
redisTime = list()
workerTotalTime = list()

for worker_times in workerLogs:
    worker_times = pickle.loads(decompress(worker_times))

    datamodelTime.append(worker_times["datamodel_sock"])
    sockLoadTime.append(worker_times["socket_load"])
    embeddingTime.append(worker_times["embedding"])
    modelRunTime.append(worker_times["model_run"])
    sockSaveTime.append(worker_times["socket_save"])
    redisTime.append(worker_times["redis"])
    workerTotalTime.append(worker_times["worker_total"])

with open("logs/test_log.txt", 'a') as f:
    
    f.write("\n== preprocessing_time = {}\n".format(preprocessingTime))                             # master.py 의 preprocessTime
    f.write("\n== train_time = {}\n".format(trainTime))                                             # master.py 의 iteration while 문 안의 시간
    f.write("\n== avg_work_time = {}\n".format(str(np.mean(workTimes))))
    f.write("\n== avg_worker_time = {}\n".format(str(np.mean(workerTotalTime))))                          # master.py 의 work 를 측정한 avg workTimes
    f.write("\n== avg_maxmin_time = {}\n".format(str(np.mean(maxminTimes))))                        # master.py 의 iteration while 에서 측정한 maxminTimes
    f.write("\n== avg_datamodel_sock_time = {}\n".format(str(np.mean(datamodelTime))))              # worker.py 에서 측정한 datamodelTime
    f.write("\n== avg_socket_load_time = {}\n".format(str(np.mean(sockLoadTime))))                  # worker.py 에서 측정한 sockLoadTime
    f.write("\n== avg_embedding_time = {}\n".format(str(np.mean(embeddingTime))))                   # worker.py 에서 측정한 embeddingTime
    f.write("\n== avg_model_run_time = {}\n".format(str(np.mean(modelRunTime))))                    # embedding.cpp 에서 측정한 modelRunTime
    f.write("\n== avg_socket_save_time = {}\n".format(str(np.mean(sockSaveTime))))                  # worker.py 에서 측정한 sockSaveTime
    f.write("\n== avg_redis_time = {}\n".format(str(np.mean(redisTime))))                           # worker.py 에서 측정한 redisTime   
