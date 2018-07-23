# coding: utf-8
from distributed import Client
from sklearn.preprocessing import normalize
from subprocess import Popen
from argparse import ArgumentParser
from collections import defaultdict
from zlib import compress, decompress
from pickle import dumps, loads, HIGHEST_PROTOCOL
from struct import pack, unpack
from port_for import select_random
import logging
import numpy as np
import redis
from time import sleep
import socket
import timeit
import sys
import os


# argument parse
parser = ArgumentParser(description='Distributed Knowledge Graph Embedding')
parser.add_argument('--num_worker', type=int, default=2, help='number of workers')
parser.add_argument('--data_root', type=str, default='/fb15k',
                    help='root directory of data(must include a name of dataset)')
parser.add_argument('--train_model', type=str, default='TransE',
                    help='training model(TransE/TransG)')
parser.add_argument('--niter', type=int, default=2,
                    help='total number of masters iterations')
parser.add_argument('--train_iter', type=int, default=10,
                    help='total number of workers(actual) training iterations')
parser.add_argument('--install', default='True', help='install libraries in each worker')
parser.add_argument('--ndim', type=int, default=20, help='dimension of embeddings')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--margin', type=int, default=2, help='margin')
parser.add_argument('--n_cluster', type=int, default=10, help='number of clusters in TransG model')
parser.add_argument('--crp', type=float, default=0.05, help='crp factor in TransG model')
parser.add_argument('--anchor_num', type=int, default=5,
                    help='number of anchor during entity training')
parser.add_argument('--anchor_interval', type=int, default=6,
                    help='number of epoch that anchors can rest as non-anchor')
parser.add_argument('--root_dir', type=str,
                    default="/home/rudvlf0413/distributedKGE/Embedding", help='project directory')
parser.add_argument('--temp_dir', type=str, default='', help='temp directory')
parser.add_argument('--pypy_dir', type=str,
                    default="/home/rudvlf0413/pypy2-v6.0.0-linux64/bin/pypy", help='pypy directory')
parser.add_argument('--redis_ip', type=str,
                    default='163.152.29.73', help='redis ip address')
parser.add_argument('--scheduler_ip', type=str,
                    default='163.152.29.73:8786', help='dask scheduler ip:port')
parser.add_argument('--use_scheduler_config_file', default='False',
                    help='wheter to use scheduler config file or use scheduler ip directly')
parser.add_argument('--debugging', type=str, default='yes', help='debugging mode or not')
parser.add_argument('--precision', type=int, default=0, help='single:0, half: 1')
args = parser.parse_args()

precision = int(args.precision)
precision_string = 'f' if precision == 0 else 'e'
precision_byte = 4 if precision == 0 else 2
np_dtype = np.float32 if precision == 0 else np.float16

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

def model2id(train_model):

    if train_model.lower() == "transe":

        return 0
    
    elif train_model.lower() == "transg":

        return 1

    else:
    
        print("[error] master > train model mismatch")
        sys.exit(1)


def install_libs():

    import os
    from pkgutil import iter_modules

    modules = set([m[1] for m in iter_modules()])

    if 'redis' not in modules:
        os.system("pip install --upgrade redis")
    if 'hiredis' not in modules:
        os.system("pip install --upgrade hiredis")


def work(chunk_data, worker_id, cur_iter, n_dim, lr, margin, train_iter, data_root_id,
         redis_ip, root_dir, debugging, precision, train_model, n_cluster, crp):

    # socket_port = init_port + (cur_iter + 1) * int(worker_id.split('_')[1])
    socket_port = str(select_random())
    # print('master > work function called, cur_iter = ' + str(cur_iter) + ', port = ' + str(socket_port))
    log_dir = os.path.join(root_dir, f'logs/embedding_log_{worker_id}_iter_{cur_iter}.txt')

    workStart = timeit.default_timer()

    embedding_proc = Popen([train_code_dir, worker_id, str(cur_iter), str(n_dim), str(lr),
                            str(margin), str(train_iter), str(data_root_id), socket_port,
                            log_dir, str(precision), str(train_model), str(n_cluster), str(crp)],
                            cwd=preprocess_folder_dir)

    worker_proc = Popen(["python", worker_code_dir, chunk_data, worker_id, str(cur_iter), str(n_dim),
                         redis_ip, root_dir, socket_port, debugging, str(precision)])

    worker_proc.wait()
    worker_return = worker_proc.returncode

    if worker_return > 0:
        # worker.py가 비정상 종료 (embedding이 중간에 비정상이면 worker.py도 비정상)

        if embedding_proc.poll() is None:
            embedding_proc.kill()

        return (False, None)

    else:

        embedding_proc.wait()
        embedding_return = embedding_proc.returncode

        if embedding_return < 0:
            # worker.py는 정상 종료 되었지만 embedding이 비정상 정료
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
train_model = model2id(args.train_model)
niter = args.niter
train_iter = args.train_iter
n_dim = args.ndim
lr = args.lr
margin = args.margin
n_cluster = args.n_cluster
crp = args.crp
anchor_num = args.anchor_num
anchor_interval = args.anchor_interval

entities = list()
entities_append = entities.append
relations = list()
relations_append = relations.append
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

                entities_append(head)
                entity2id[head] = entity_cnt
                entity_cnt += 1

            if tail not in entity2id:

                entities_append(tail)
                entity2id[tail] = entity_cnt
                entity_cnt += 1

            if relation not in relation2id:

                relations_append(relation)
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

    allocated_relation_worker = sorted(allocated_relation_worker, key=lambda x: x[1])
    allocated_relation_worker[0][0].append(relation)
    allocated_relation_worker[0][1] += num

# printt('master > # of relations per each partitions : [%s]' %
#       " ".join([str(len(relation_list)) for relation_list, num in allocated_relation_worker]))

sub_graphs = {}

for c, (relation_list, num) in enumerate(allocated_relation_worker):

    g = []
    g_append = g.append
    for relation in relation_list:
        for (head, tail) in relation_triples[relation]:
            g_append((head, relation, tail))
    sub_graphs['sub_g_worker_%d' % c] = compress(dumps(
        g, protocol=HIGHEST_PROTOCOL), 9)

r = redis.StrictRedis(host=args.redis_ip, port=6379, db=0)
r.mset(sub_graphs)

del relation_each_num
del relation_triples
del allocated_relation_worker
del sub_graphs

r.mset(entity2id)
r.mset(relation2id)

r.set('entities', compress(dumps(entities, protocol=HIGHEST_PROTOCOL), 9))
entity_ids = np.array(list(entity2id.values()), dtype=np.int32)
entities_initialized = normalize(np.random.randn(len(entities), n_dim))

r.set('relations', compress(dumps(relations, protocol=HIGHEST_PROTOCOL), 9))
relation_ids = np.array(list(relation2id.values()), dtype=np.int32)
relations_initialized = normalize(np.random.randn(len(relations), n_dim))

r.mset({f'{entity}_v': compress(dumps(vector,
        protocol=HIGHEST_PROTOCOL), 9) for vector, entity in zip(entities_initialized, entities)})
r.mset({f'{relation}_v': compress(dumps(vector,
        protocol=HIGHEST_PROTOCOL), 9) for vector, relation in zip(relations_initialized, relations)})

if args.use_scheduler_config_file == 'True':

    client = Client(scheduler_file=f'{temp_folder_dir}/scheduler.json', name='Embedding')

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
maxmin_port = select_random()
proc = Popen([args.pypy_dir, 'maxmin.py', str(num_worker), '0', str(anchor_num),
              str(anchor_interval), args.root_dir, args.data_root, args.debugging,
              str(maxmin_port)])

while True:

    try:

        maxmin_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        maxmin_sock.connect(('127.0.0.1', maxmin_port))
        break
    
    except Exception as e:
    
        sleep(0.5)
        printt('[error] master > exception occured in master <-> maxmin')
        printt('[error] master > ' + str(e))

# printt('master > socket connected (master <-> maxmin)')

timeNow = timeit.default_timer()
maxmin_sock.send(pack('!i', 0))
maxmin_sock.send(pack('!i', 0))

# 원소를 한 번에 받음
chunks = list()
anchor_len = unpack('!i', sockRecv(maxmin_sock, 4))[0]
anchors = unpack('!' + 'i' * int(anchor_len), sockRecv(maxmin_sock, 4 * int(anchor_len)))
anchors = ' '.join(str(e) for e in anchors)

for _ in range(num_worker):

    chunk_len = unpack('!i', sockRecv(maxmin_sock, 4))[0]
    chunk = unpack('!' + 'i' * chunk_len, sockRecv(maxmin_sock, 4 * chunk_len))
    chunk = ' '.join(str(e) for e in chunk)
    chunks.append(chunk)

maxminTimes.append(timeit.default_timer() - timeNow)

# printt('master > maxmin finished')
# printt('master > worker training iteration epoch : {}'.format(train_iter))

cur_iter = 0
trial  = 0
success = False

trainStart = timeit.default_timer()

while True:

    # 이터레이션이 실패할 경우를 대비해 redis 의 값을 백업
    entities_initialized_bak = r.mget([f'{entity}_v' for entity in entities])
    entities_initialized_bak = np.array([loads(decompress(v)) for v in entities_initialized_bak])
    relations_initialized_bak = r.mget([f'{relation}_v' for relation in relations])
    relations_initialized_bak = np.array([loads(decompress(v)) for v in relations_initialized_bak])

    if cur_iter == niter:

        break

    if trial == 5:

        printt('[error] master > training failed, exit')
        maxmin_sock.send(pack('!i', 1))
        maxmin_sock.close()
        sys.exit(1)

    # 작업 배정
    printt('[info] master > iteration %d' % cur_iter)
    iterStart = timeit.default_timer()
    
    workers = [client.submit(work, f"{anchors}\n{chunks[i]}", f'worker_{i}', cur_iter,
                             n_dim, lr, margin, train_iter, data_root_id,
                             args.redis_ip, args.root_dir, args.debugging, args.precision,
                             train_model, n_cluster, crp) for i in range(num_worker)]

    if cur_iter % 2 == 1:
        # entity partitioning: max-min cut 실행, anchor 등 재분배
        
        maxminStart = timeit.default_timer()

        maxmin_sock.send(pack('!i', 0))
        maxmin_sock.send(pack('!i', cur_iter))

        # 원소를 한 번에 받음
        chunks = list()
        
        anchor_len = unpack('!i', sockRecv(maxmin_sock, 4))[0]
        anchors = unpack('!' + 'i' * anchor_len, sockRecv(maxmin_sock, 4 * anchor_len))
        anchors = ' '.join(str(e) for e in anchors)
        
        for _ in range(num_worker):
        
            chunk_len = unpack('!i', sockRecv(maxmin_sock, 4))[0]
            chunk = unpack('!' + 'i' * chunk_len, sockRecv(maxmin_sock, 4 * chunk_len))
            chunk = ' '.join(str(e) for e in chunk)
            chunks.append(chunk)

        maxminTimes.append(timeit.default_timer() - maxminStart)

    else:
        # relation partitioning
        chunk_data = ''

    client.gather(workers)
    result_iter = [worker.result() for worker in workers]
    iterTimes.append(timeit.default_timer() - iterStart)

    if all([e[0] for e in result_iter]):

        # 이터레이션 성공
        printt('master > iteration time : %f' % (timeit.default_timer() - timeNow))
        success = True
        trial = 0
        cur_iter += 1

        workTimes = [e[1] for e in result_iter]

        # embedding.cpp 에서 model->run() 실행 시간을 worker.py 로 전송해서 그걸 소켓으로 전송

        printt('master > Total embedding times : ' + str(workTimes))
        # printt('master > Average total embedding time : ' + str(np.mean(workTimes)))

    else:

        # 이터레이션 실패
        # redis 에 저장된 결과를 백업된 값으로 되돌림
        trial += 1

        r.mset({
            f'{entity}_v' : compress(dumps(vector, protocol=HIGHEST_PROTOCOL))
            for vector, entity in zip(entities_initialized_bak, entities)})
        r.mset({
            f'{relation}_v' : compress(dumps(vector, protocol=HIGHEST_PROTOCOL))
            for vector, relation in zip(relations_initialized_bak, relations)})
        printt('[error] master > iteration %d is failed, retry' % cur_iter)


trainTime = timeit.default_timer() - trainStart

###############################################################################
###############################################################################

# test part
# printt('master > test start')

# load entity vector
entities_initialized = r.mget([f'{entity}_v' for entity in entities])
entities_initialized = np.array([loads(decompress(v)) for v in entities_initialized], dtype=np_dtype)
relations_initialized = r.mget([f'{relation}_v' for relation in relations])
relations_initialized = np.array([loads(decompress(v)) for v in relations_initialized], dtype=np_dtype)

maxmin_sock.send(pack('!i', 1))
maxmin_sock.close()

worker_id = 'worker_0'
log_dir = os.path.join(args.root_dir, 'logs/test_log.txt')
test_port = select_random()
proc = Popen([test_code_dir, worker_id, '-1', str(n_dim), str(lr), str(margin),
              str(data_root_id), str(log_dir), str(precision), str(train_model),
              str(n_cluster), str(crp), str(test_port)], cwd=preprocess_folder_dir)

while True:

    try:

        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        break

    except Exception as e:

        sleep(0.5)
        printt('[error] master > exception occured in master <-> test')
        printt('[error] master > ' + str(e))

while True:

    try:

        test_sock.connect(('127.0.0.1', test_port))
        break

    except Exception as e:

        sleep(0.5)
        printt('[error] master > exception occured in master <-> test')
        printt('[error] master > ' + str(e))

# DataModel 생성자 -> GeometricModel load 메소드 -> GeometricModel save 메소드 순서로 통신
checksum = 0
success = 0

# entity_vector 전송 - GeometricModel load
while success != 1:

    # 원소를 한 번에 전송 - 2 단계
    for id_, vector in zip(entity_ids, entities_initialized):
        
            test_sock.send(pack('!i', id_))
            test_sock.send(pack(precision_string * len(vector), * vector))
    
    checksum = unpack('!i', sockRecv(test_sock, 4))[0]

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

    # 원소를 한 번에 전송 - 2 단계
    for id_, vector in zip(relation_ids, relations_initialized):
        
            test_sock.send(pack('!i', id_))
            test_sock.send(pack(precision_string * len(vector), *vector))

    checksum = unpack('!i', sockRecv(test_sock, 4))[0]

    if checksum == 1234:

        # printt('master > phase 2 (relation) finished (for test)')
        success = 1

    elif checksum == 9876:

        printt('[error] master > retry phase 2 (relation) (for test)')
        success = 0

    else:

        printt('[error] master > unknown error in phase 2 (relation) (for test)')
        success = 0

# printt('master > relation_vector sent to Geome tricModel load function (for test)')

test_return = proc.communicate()
test_sock.close()

if test_return == -1:

    printt('[error] master > test failed, exit')
    sys.exit(1)

totalTime = timeit.default_timer() - masterStart
printt('master > Total elapsed time : %f' % (totalTime))

workerLogKeys = [f'worker_{n}_{i}' for i in range(niter) for n in range(num_worker)]
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

    worker_times = loads(decompress(worker_times))
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

sys.exit(0)
