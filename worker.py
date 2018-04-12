# coding: utf-8
from subprocess import Popen
from time import time
import logging
import numpy as np
import redis
import pickle
import sys
import socket
import time as tt
import struct

chunk_data = sys.argv[1]
worker_id = sys.argv[2]
cur_iter = sys.argv[3]
embedding_dim = sys.argv[4]
learning_rate = sys.argv[5]
margin = sys.argv[6]
train_iter = sys.argv[7]
redis_ip_address = sys.argv[8]
root_dir = sys.argv[9]
data_root_id = sys.argv[10] 
logging.basicConfig(filename='%s/worker_%s.log' % (root_dir, worker_id), filemode='w', level=logging.DEBUG)
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
temp_folder_dir = "%s/tmp" % root_dir

t_ = time()
# redis에서 embedding vector들 받아오기
r = redis.StrictRedis(host=redis_ip_address, port=6379, db=0)
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

printt('redis server connection time: %f' % (time() - t_))

t_ = time()

# embedding.cpp 와 socket 통신
# worker 가 실행될 때 전달받은 ip 와 port 로 접속
# Embedding.cpp 가 server, 프로세느는 master.py 가 생성
# worker.py 가 client
# 첫 iteration 에서눈 Embedding.cpp 의 실행, 소켓 생성을 기다림

# worker_id 를 기반으로 포트를 생성
embedding_addr = '0.0.0.0'
embedding_port = 49900 + 5 * int(worker_id.split('_')[1]) + int(cur_iter) % 5

printt('port number of ' + worker_id + ' : ' + str(embedding_port) + ' - worker.py')

while True:
    try:
        embedding_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        break
    except (TimeoutError, ConnectionRefusedError):
        tt.sleep(1)

ci = int(cur_iter)

while True:
    try:
        embedding_sock.connect((embedding_addr, embedding_port))
        break
    except (TimeoutError, ConnectionRefusedError):
        tt.sleep(1)
    except Exception e:
        ci = ci + 1
        embedding_port = 49900 + 5 * int(worker_id.split('_')[1]) + ci % 5

printt('socket connected to embedding.cpp to worker.py - worker.py')

# 연산 요청 메시지

embedding_sock.send(struct.pack('!i', 0))
# int 임시 땜빵, 매우 큰 문제
embedding_sock.send(struct.pack('!i', int(worker_id.split('_')[1])))
embedding_sock.send(struct.pack('!i', int(cur_iter)))            # int
embedding_sock.send(struct.pack('!i', int(embedding_dim)))       # int
embedding_sock.send(struct.pack('d', float(learning_rate)))      # double
embedding_sock.send(struct.pack('d', float(margin)))             # double
embedding_sock.send(struct.pack('!i', int(data_root_id)))        # int

printt('sent params to embedding.cpp - worker.py, ' + worker_id)

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

        embedding_sock.send(struct.pack('!i', len(chunk_anchor)))

        for iter_anchor in chunk_anchor:
            embedding_sock.send(struct.pack('!i', int(iter_anchor)))

        embedding_sock.send(struct.pack('!i', len(chunk_entity)))

        for iter_entity in chunk_entity:
            embedding_sock.send(struct.pack('!i', int(iter_entity)))

        checksum = struct.unpack('!i', embedding_sock.recv(4))[0]

        if checksum == 1234:

            printt('phase 1 finished - worker.py, ' + worker_id)
            checksum = 1

        elif checksum == 9876:

            printt('retry phase 1 - worker.py, ' + worker_id)
            checksum = 0

        else:

            printt('unknown error in phase 1 - worker.py, ' + worker_id)
            checksum = 0

else:
    # relation 전송 - DataModel 생성자
    sub_graphs = pickle.loads(r.get('sub_graph_{}'.format(worker_id)))
    embedding_sock.send(struct.pack('!i', len(sub_graphs)))

    while checksum != 1:

        for (head_id, relation_id, tail_id) in sub_graphs:
            embedding_sock.send(struct.pack('!i', int(head_id)))
            embedding_sock.send(struct.pack('!i', int(relation_id)))
            embedding_sock.send(struct.pack('!i', int(tail_id)))

        checksum = struct.unpack('!i', embedding_sock.recv(4))[0]

        if checksum == 1234:

            printt('phase 1 finished - worker.py, ' + worker_id)
            checksum = 1

        elif checksum == 9876:

            printt('retry phase 1 - worker.py, ' + worker_id)
            checksum = 0

        else:

            printt('unknown error in phase 1 - worker.py, ' + worker_id)
            checksum = 0

printt('chunk or relation sent to DataModel - worker.py')

checksum = 0

# entity_vector 전송 - GeometricModel load
while checksum != 1:

    for i, vector in enumerate(entities_initialized):
        entity_name = str(entities[i])
        embedding_sock.send(struct.pack('!i', len(entity_name)))
        embedding_sock.send(str.encode(entity_name))    # entity string 자체를 전송


        #embedding_sock.send(struct.pack('!i', entity2id[entity_name])) # entity id 를 int 로 전송


        for v in vector:
            embedding_sock.send(struct.pack('d', float(v)))

    checksum = struct.unpack('!i', embedding_sock.recv(4))[0]

    if checksum == 1234:

        printt('phase 2 (entity) finished - worker.py, ' + worker_id)
        checksum = 1

    elif checksum == 9876:

        printt('retry phase 2 (entity) - worker.py, ' + worker_id)
        checksum = 0

    else:

        printt('unknown error in phase 2 (entity) - worker.py, ' + worker_id)
        checksum = 0

printt('entity_vector sent to GeometricModel load function - worker.py')

checksum = 0

# relation_vector 전송 - GeometricModel load
while checksum != 1:

    for i, relation in enumerate(relations_initialized):
        relation_name = str(relations[i])
        embedding_sock.send(struct.pack('!i', len(relation_name)))
        embedding_sock.send(str.encode(relation_name))  # relation string 자체를 전송


        #embedding_sock.send(struct.pack('!i', relation2id[relation_name])) # relation id 를 int 로 전송


        for v in relation:
            embedding_sock.send(struct.pack('d', float(v)))

    checksum = struct.unpack('!i', embedding_sock.recv(4))[0]

    if checksum == 1234:

        printt('phase 2 (relation) finished - worker.py, ' + worker_id)
        checksum = 1

    elif checksum == 9876:

        printt('retry phase 2 (relation) - worker.py, ' + worker_id)
        checksum = 0

    else:

        printt('unknown error in phase 2 (relation) - worker.py, ' + worker_id)
        checksum = 0

printt('relation_vector sent to Geome tricModel load function - worker.py')

del entities_initialized
del relations_initialized

w_id = worker_id.split('_')[1]
t_ = time()

tempcount = 0

if int(cur_iter) % 2 == 0:

    success = 0

    while success != 1:

        try:        

            entity_vectors = dict()

            # 처리 결과를 받아옴 - GeometricModel save
            count_entity_data = embedding_sock.recv(4)
            if len(count_entity_data) is not 4:
                printt('length of count_entity_data is ' + str(len(count_entity_data)) + ' - worker.py')
                printt(str(embedding_port) + ' - worker.py')
            count_entity = struct.unpack('!i', count_entity_data)[0]
            printt('count_entity is ' + str(count_entity) + ' - worker.py')

            for entity_idx in range(count_entity):
                temp_entity_vector = list()
                entity_id_len = struct.unpack('!i', embedding_sock.recv(4))[0]
                entity_id = embedding_sock.recv(entity_id_len).decode()


                #entity_id = struct.unpack('!i', embedding_sock.recv(4))[0]     # entity_id 를 int 로 받음


                for dim_idx in range(int(embedding_dim)):
                    temp_entity_double = embedding_sock.recv(8)
                    if len(temp_entity_double) is not 8:
                        printt('length of temp_entity_double is ' + str(len(temp_entity_double)) + ' - worker.py')
                    temp_entity = struct.unpack('d', temp_entity_double)[0]
                    temp_entity_vector.append(temp_entity)

                entity_vectors[entity_id + '_v'] = pickle.dumps(
                    np.array(temp_entity_vector), protocol=pickle.HIGHEST_PROTOCOL)

        except Exception as e:

            if tempcount > 5:

                printt('retry phase 3 (entity) - worker.py, ' + worker_id)
                printt(e.message)

            tempcount = tempcount + 1
            flag = 9876
            embedding_sock.send(struct.pack('!i', flag))
            sucess = 0

        else:

            printt('phase 3 (entity) finished - worker.py, ' + worker_id)
            flag = 1234
            embedding_sock.send(struct.pack('!i', flag))
            sucess = 1
    
    r.mset(entity_vectors)

else:

    success = 0

    while success != 1:

        try:    

            relation_vectors = dict()

            # 처리 결과를 받아옴 - GeometricModel save
            count_relation_data = embedding_sock.recv(4)
            if len(count_relation_data) is not 4:
                printt('length of count_relation_data is ' + str(len(count_relation_data)) + ' - worker.py')
            count_relation = struct.unpack('!i', count_relation_data)[0]
            printt('count_relation is ' + str(count_relation) + ' - worker.py')

            for relation_idx in range(count_relation):
                temp_relation_vector = list()
                relation_id_len = struct.unpack('!i', embedding_sock.recv(4))[0]
                relation_id = embedding_sock.recv(relation_id_len).decode()


                #relation_id = struct.unpack('!i', embedding_sock.recv(4))[0]   # relation_id 를 int 로 바음


                for dim_idx in range(int(embedding_dim)):
                    temp_relation_double = embedding_sock.recv(8)
                    if len(temp_relation_double) is not 8:
                        printt('length of temp_relation_double is ' + len(temp_relation_double) + ' - worker.py')
                    temp_relation = struct.unpack('d', temp_relation_double)[0]
                    temp_relation_vector.append(temp_relation)

                relation_vectors[relation_id + '_v'] = pickle.dumps(
                    np.array(temp_relation_vector), protocol=pickle.HIGHEST_PROTOCOL)
    
        except Exception as e:

            if tempcount > 5:

                printt('retry phase 3 (relation) - worker.py, ' + worker_id)
                printt(e.message)

            tempcount = tempcount + 1
            flag = 9876
            embedding_sock.send(struct.pack('!i', flag))
            sucess = 0

        else:

            printt('phase 3 (relation) finished - worker.py, ' + worker_id)
            flag = 1234
            embedding_sock.send(struct.pack('!i', flag))
            sucess = 1

    r.mset(relation_vectors)

printt('recieved result from GeometricModel save function - worker.py')

printt('redis server connection time : %f - worker.py' % (time() - t_))
printt('{}: {} iteration finished! - worker.py'.format(worker_id, cur_iter))\

sys.exit(0)