# coding: utf-8
import os
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
socket_port = sys.argv[11]
logging.basicConfig(filename='%s/worker_%s.log' % (root_dir, worker_id), filemode='w', level=logging.DEBUG)
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

printt('[info] worker.py > redis server connection time : %f' % (time() - t_))

t_ = time()

# embedding.cpp 와 socket 통신
# worker 가 실행될 때 전달받은 ip 와 port 로 접속
# Embedding.cpp 가 server, 프로세느는 master.py 가 생성
# worker.py 가 client
# 첫 iteration 에서눈 Embedding.cpp 의 실행, 소켓 생성을 기다림

trial = 0
while True:
    
    try:

        embedding_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        break

    except Exception as e:
        
        tt.sleep(1)
        trial = trial + 1
        printt('[error] worker.py > exception occured in worker <-> embedding')
        printt('[error] worker.py > ' + str(e))

    if trial == 5:

        printt('[error] worker.py > iteration ' + str(cur_iter) + ' failed - ' + worker_id)
        printt('[error] worker.py > return -1')
        sys.exit(-1)

trial = 0
while True:
    
    try:
        
        embedding_sock.connect(('0.0.0.0', int(socket_port)))
        break

    except ConnectionRefusedError:
        
        tt.sleep(1)
        trial = trial + 1
        printt('[error] worker.py > exception occured in worker <-> embedding')
        printt('[error] worker.py > ConnectionRefusedError')

    except TimeoutError:
        
        tt.sleep(1)
        trial = trial + 1
        printt('[error] worker.py > exception occured in worker <-> embedding')
        printt('[error] worker.py > TimeoutError')

    except Exception as e:
        
        tt.sleep(1)
        trial = trial + 1
        printt('[error] worker.py > exception occured in worker <-> embedding')
        printt('[error] worker.py > ' + str(e))

    if trial == 5:

        printt('[error] worker.py > iteration ' + str(cur_iter) + ' failed - ' + worker_id)
        printt('[error] worker.py > return -1')
        sys.exit(-1)

printt('[info] worker.py > port number of ' + worker_id + ' = ' + socket_port)
printt('[info] worker.py > socket connected (worker <-> embedding)')

# DataModel 생성자 -> GeometricModel load 메소드 -> GeometricModel save 메소드 순서로 통신
try:

    checksum = 0

    if int(cur_iter) % 2 == 0:
        # entity 전송 - DataModel 생성자
        chunk_anchor, chunk_entity = chunk_data.split('\n')
        chunk_anchor = chunk_anchor.split(' ')
        chunk_entity = chunk_entity.split(' ')

        if len(chunk_anchor) == 1 and chunk_anchor[0] == '':
            
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

                printt('[info] worker.py > phase 1 finished - ' + worker_id)
                checksum = 1

            elif checksum == 9876:

                printt('[error] worker.py > retry phase 1 - ' + worker_id)
                checksum = 0

            else:

                printt('[error] worker.py > unknown error in phase 1 - ' + worker_id)
                printt('[error] worker.py > received checksum = ' + str(checksum) + ' - ' + worker_id)
                printt('[error] worker.py > return -1')
                sys.exit(-1)

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

                printt('[info] worker.py > phase 1 finished - ' + worker_id)
                checksum = 1

            elif checksum == 9876:

                printt('[error] worker.py > retry phase 1 - ' + worker_id)
                checksum = 0

            else:

                printt('[error] worker.py > unknown error in phase 1 - ' + worker_id)
                printt('[error] worker.py > received checksum = ' + str(checksum) + ' - ' + worker_id)
                printt('[error] worker.py > return -1')
                sys.exit(-1)

    printt('[info] worker.py > chunk or relation sent to DataModel')

    checksum = 0

    # entity_vector 전송 - GeometricModel load
    while checksum != 1:

        for i, vector in enumerate(entities_initialized):
            entity_name = str(entities[i])
            #embedding_sock.send(struct.pack('!i', len(entity_name)))
            #embedding_sock.send(str.encode(entity_name))    # entity string 자체를 전송


            embedding_sock.send(struct.pack('!i', entity_id[entity_name])) # entity id 를 int 로 전송


            for v in vector:
                embedding_sock.send(struct.pack('d', float(v)))

        checksum = struct.unpack('!i', embedding_sock.recv(4))[0]

        if checksum == 1234:

            printt('[info] worker.py > phase 2 (entity) finished - ' + worker_id)
            checksum = 1

        elif checksum == 9876:

            printt('[error] worker.py > retry phase 2 (entity) - ' + worker_id)
            checksum = 0

        else:

            printt('[error] worker.py > unknown error in phase 2 (entity) - ' + worker_id)
            printt('[error] worker.py > received checksum = ' + str(checksum) + ' - ' + worker_id)
            printt('[error] worker.py > return -1')
            sys.exit(-1)

    printt('[info] worker.py > entity_vector sent to GeometricModel load function')

    checksum = 0

    # relation_vector 전송 - GeometricModel load
    while checksum != 1:

        for i, relation in enumerate(relations_initialized):
            relation_name = str(relations[i])
            #embedding_sock.send(struct.pack('!i', len(relation_name)))
            #embedding_sock.send(str.encode(relation_name))  # relation string 자체를 전송

            if len(relation_name) > 1000 or len(relation_name) < 0:

                printt('[error] length of relation_name is strange')
                printt('[error] relation_name = ' + str(relation_name))
                printt('[error] len(relation_name) = ' + str(len(relation_name)))


            embedding_sock.send(struct.pack('!i', relation_id[relation_name])) # relation id 를 int 로 전송


            for v in relation:
                embedding_sock.send(struct.pack('d', float(v)))

        checksum = struct.unpack('!i', embedding_sock.recv(4))[0]

        if checksum == 1234:

            printt('[info] worker.py > phase 2 (relation) finished - ' + worker_id)
            checksum = 1

        elif checksum == 9876:

            printt('[error] worker.py > retry phase 2 (relation) - worker.py - ' + worker_id)
            checksum = 0

        else:

            printt('[error] worker.py > unknown error in phase 2 (relation) - ' + worker_id)
            printt('[error] worker.py > received checksum = ' + str(checksum) + ' - ' + worker_id)
            printt('[error] worker.py > return -1')
            sys.exit(-1)

    printt('[info] worker.py > relation_vector sent to GeometricModel load function')

    del entities_initialized
    del relations_initialized

    tempcount = 0

    if int(cur_iter) % 2 == 0:

        success = 0

        while success != 1:

            try:        

                entity_vectors = dict()

                # 처리 결과를 받아옴 - GeometricModel save
                count_entity_data = embedding_sock.recv(4)
                
                if len(count_entity_data) != 4:
                    
                    printt('[info] worker.py > length of count_entity_data = ' + str(len(count_entity_data)))
                    printt('[info] worker.py > embedding_port = ' + socket_port)
                
                count_entity = struct.unpack('!i', count_entity_data)[0]
                printt('[info] worker.py > count_entity = ' + str(count_entity))

                for entity_idx in range(count_entity):
                    
                    temp_entity_vector = list()
                    
                    #entity_name_len = struct.unpack('!i', embedding_sock.recv(4))[0]
                    #entity_name = embedding_sock.recv(entity_id_len).decode()


                    entity_name_id = str(struct.unpack('!i', embedding_sock.recv(4))[0])     # entity_name 를 int 로 받음


                    for dim_idx in range(int(embedding_dim)):
                        
                        temp_entity_double = embedding_sock.recv(8)
                        
                        if len(temp_entity_double) != 8:
                            
                            printt('[info] worker.py > length of temp_entity_double = ' + str(len(temp_entity_double)))
                        
                        temp_entity = struct.unpack('d', temp_entity_double)[0]
                        temp_entity_vector.append(temp_entity)

                    entity_vectors[entity_name_id + '_v'] = pickle.dumps(
                        np.array(temp_entity_vector), protocol=pickle.HIGHEST_PROTOCOL)

            except Exception as e:

                if tempcount < 3:

                    printt('[error] worker.py > retry phase 3 (entity) - ' + worker_id)
                    printt('[error] worker.py > ' + str(e))

                tempcount = tempcount + 1
                flag = 9876
                embedding_sock.send(struct.pack('!i', flag))
                success = 0

            else:

                printt('[info] worker.py > phase 3 (entity) finished - ' + worker_id)
                flag = 1234
                embedding_sock.send(struct.pack('!i', flag))
                success = 1
        
        r.mset(entity_vectors)
        printt('[info] worker.py > entity_vectors updated - ' + worker_id)
        sys.exit(0)

    else:

        success = 0

        while success != 1:

            try:    

                relation_vectors = dict()

                # 처리 결과를 받아옴 - GeometricModel save
                count_relation_data = embedding_sock.recv(4)
                
                if len(count_relation_data) != 4:
                    printt('[info] worker.py > length of count_relation_data = ' + str(len(count_relation_data)))

                count_relation = struct.unpack('!i', count_relation_data)[0]
                printt('[info] worker.py > count_relation is ' + str(count_relation))

                for relation_idx in range(count_relation):
                    
                    temp_relation_vector = list()
                    #relation_name_len = struct.unpack('!i', embedding_sock.recv(4))[0]
                    #relation_name = embedding_sock.recv(relation_id_len).decode()


                    relation_name_id = str(struct.unpack('!i', embedding_sock.recv(4))[0])   # relation_name 를 int 로 받음


                    for dim_idx in range(int(embedding_dim)):
                        
                        temp_relation_double = embedding_sock.recv(8)
                        
                        if len(temp_relation_double) != 8:
                            
                            printt('[info] worker.py > length of temp_relation_double = ' + len(temp_relation_double))

                        temp_relation = struct.unpack('d', temp_relation_double)[0]
                        temp_relation_vector.append(temp_relation)

                    relation_vectors[relation_name_id + '_v'] = pickle.dumps(
                        np.array(temp_relation_vector), protocol=pickle.HIGHEST_PROTOCOL)
        
            except Exception as e:

                if tempcount < 3:

                    printt('[error] worker.py > retry phase 3 (relation) - ' + worker_id)
                    printt('[error] worker.py > ' + str(e))

                tempcount = tempcount + 1
                flag = 9876
                embedding_sock.send(struct.pack('!i', flag))
                success = 0

            else:

                printt('[info] worker.py > phase 3 (relation) finished - ' + worker_id)
                flag = 1234
                embedding_sock.send(struct.pack('!i', flag))
                success = 1

        r.mset(relation_vectors)
        printt('[info] worker.py > relation_vectors updated - ' + worker_id)
        sys.exit(0)

    #printt('[info] worker.py > recieved result from GeometricModel save function')
    #printt('[info] worker.py > {}: {} iteration finished!'.format(worker_id, cur_iter))

except Exception as e:

    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

    printt('[error] worker.py > exception occured in iteration - ' + str(worker_id))
    printt('[error] worker.py > ' + str(e))
    printt('[error] worker.py > exception occured in line ' + str(exc_tb.tb_lineno))
    printt('[error] worker.py > return -1')
    sys.exit(-1)