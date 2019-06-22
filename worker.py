# coding: utf-8
from subprocess import Popen
from time import sleep
from pickle import dumps, loads, load, HIGHEST_PROTOCOL
from struct import pack, unpack
from timeit import default_timer
from .utils import sockRecv
from .utils import iter_mset
from .utils import iter_mget
import logging
import numpy as np
import redis
import pickle
import sys
import os
import socket


worker_id = sys.argv[1]
cur_iter = int(sys.argv[2])
embedding_dim = int(sys.argv[3])
redis_ip_address = sys.argv[4]
root_dir = sys.argv[5]
socket_port = int(sys.argv[6])
debugging = sys.argv[7]
precision = int(sys.argv[8])
precision_string = 'f' if precision == 0 else 'e'
precision_byte = 4 if precision == 0 else 2
np_dtype = np.float32 if precision == 0 else np.float16
train_model = int(sys.argv[9])
n_cluster = int(sys.argv[10])
crp = float(sys.argv[11])
unix_socket_path = sys.argv[12]

if cur_iter % 2 == 0:
    with open(f"{root_dir}/chunk_data_{worker_id}.txt", 'rb') as f:
        #chunk_data = f.read()
        chunk_anchor, chunk_entity = pickle.load(f)

if debugging == 'yes':
    logging.basicConfig(filename='%s/%s_%d.log' % (root_dir,
                                                   worker_id, cur_iter), filemode='w', level=logging.WARNING)
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

elif debugging == 'no':

    printt = print


# embedding.cpp 와 socket 통신
# worker 가 실행될 때 전달받은 ip 와 port 로 접속
# Embedding.cpp 가 server, 프로세느는 master.py 가 생성
# worker.py 가 client
# 첫 iteration 에서눈 Embedding.cpp 의 실행, 소켓 생성을 기다림

connection_trial = 0

while True:

    try:

        printt(f'[info] worker > connecting socket <-> embedding, {worker_id}, {cur_iter}th iter')
        embedding_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        embedding_sock.settimeout(None)
        embedding_sock.connect(('127.0.0.1', socket_port))

    except Exception as e:

        printt(f'[error] worker > exception occured when connecting socket <-> embedding, {worker_id}, {cur_iter}th iter')
        printt('[error] worker > ' + str(e))
        connection_trial = connection_trial + 1
        sleep(5)
        #sys.exit(1)

        if connection_trial > 5:

            printt('[error] worker > cannot connect socket <-> embeding, exit')
            sys.exit(1)

    else:

        break

preprocess_folder_dir = "%s/preprocess/" % root_dir
train_code_dir = "%s/MultiChannelEmbedding/Embedding.out" % root_dir
temp_folder_dir = "%s/tmp" % root_dir

workerStart = default_timer()
# redis에서 embedding vector들 받아오기
if unix_socket_path == '':
    r = redis.StrictRedis(host=redis_ip_address, port=6379, db=0)
else:
    r = redis.StrictRedis(unix_socket_path=unix_socket_path)

entities = np.array(loads(r.get('entities')))
entity_ids = np.array([int(i) for i in iter_mget(r, entities)], dtype=np.int32)
entity_id = {e: i for e, i in zip(entities, entity_ids)}


########## TODO: INTERFACE ##########
entities_initialized = iter_mget(r, [f'{entity}_v' for entity in entities])
entities_initialized = np.stack([np.fromstring(v, dtype=np_dtype) for v in entities_initialized])
relations = np.array(loads(r.get('relations')))
relation_ids = np.array([int(i) for i in iter_mget(r, relations)], dtype=np.int32)
relation_id = {r: i for e, i in zip(relations, relation_ids)}

# transE 에서는 embedding_relation 을 전송
if train_model == 0:

    relations_initialized = iter_mget(r, [f'{relation}_v' for relation in relations])
    relations_initialized = np.stack([np.fromstring(v, dtype=np_dtype) for v in relations_initialized])

# transG 에 추가되는 분기
elif train_model == 1:

    embedding_clusters = iter_mget(r, [f'{relation}_cv' for relation in relations])
    embedding_clusters = np.stack([np.fromstring(v, dtype=np_dtype) for v in embedding_clusters])
    weights_clusters = iter_mget(r, [f'{relation}_wv' for relation in relations])
    weights_clusters = np.stack([np.fromstring(v, dtype=np_dtype) for v in weights_clusters])
    size_clusters = iter_mget(r, [f'{relation}_s' for relation in relations])
    size_clusters = np.stack([np.fromstring(v, dtype=np.int32) for v in size_clusters])

redisTime = default_timer() - workerStart
#printt('worker > redis server connection time : %f' % (redisTime))

#printt('worker > port number of ' + worker_id + ' = ' + socket_port)
#printt('worker > socket connected (worker <-> embedding)')

# DataModel 생성자 -> GeometricModel load 메소드 -> GeometricModel save 메소드 순서로 통신
try:

    timeNow = default_timer()

    # 짝수일 때, entity 전송 - DataModel 생성자
    # 홀수일 때, relation 전송 - DataModel 생성자
    if cur_iter % 2 == 0:

        # 분산 딥러닝 알고리즘을 구현할 때에는 이 부분이 필요 없을듯?
        #
        # overflow 때문에 전체 개수를 따로 보냄
        value_to_send = (len(chunk_anchor), len(chunk_entity))
        embedding_sock.send(pack('ii', *value_to_send))
        value_to_send = (*chunk_anchor, *chunk_entity)
        embedding_sock.send(pack('!' + 'i' * (len(chunk_anchor) + len(chunk_entity)), *value_to_send))

    else:

        sub_graphs = loads(r.get(f'sub_g_{worker_id}'))
        redisTime += default_timer() - timeNow

        # 분산 딥러닝 알고리즘을 구현할 때에는 이 부분이 필요 없을듯?
        #
        embedding_sock.send(pack('i', len(sub_graphs)))

        # 원소 한 번에 전송 - 2 단계
        value_to_send = np.array(sub_graphs).flatten()
        embedding_sock.send(pack('!' + 'i' * len(value_to_send), *value_to_send))

    datamodelTime = default_timer() - timeNow
    timeNow = default_timer()

    # entity_vector 전송 - GeometricModel load
    # 분산 딥러닝 알고리즘을 구현할 때에는 아래 처럼 id 를 구분해서 보낼 필요 없음
    # 그냥 모든 값을 전송하면 됨
    #


    ########## TODO: INTERFACE ##########
    for id_, vector in zip(entity_ids, entities_initialized):
    
        embedding_sock.send(pack(precision_string * embedding_dim, *vector))

    # transE 에서는 embedding_relation 을 전송
    if train_model == 0:
        
        # relation_vector 전송 - GeometricModel load
        for id_, vector in zip(relation_ids, relations_initialized):
            
            embedding_sock.send(pack(precision_string * embedding_dim, *vector))

        del relations_initialized
    
    # transG 에 추가되는 분기
    elif train_model == 1:
        
        # embedding_clusters 전송 - GeometricModel load    
        for id_, vector in zip(relation_ids, embedding_clusters):

            embedding_sock.send(pack(precision_string * len(vector), *vector))

        # weights_clusters 전송 - GeometricModel load
        for id_, vector in zip(relation_ids, weights_clusters):

            embedding_sock.send(pack(precision_string * len(vector), *vector))

        # size_clusters 전송 - GeometricModel load
        size_clusters = size_clusters.reshape(-1)
        embedding_sock.send(pack('!' + 'i' * len(size_clusters), *size_clusters))

    sockLoadTime = default_timer() - timeNow

    del value_to_send
    del entity_ids
    del entities_initialized

    timeNow = default_timer()

    if cur_iter % 2 == 0:

        # 처리 결과를 받아옴 - GeometricModel save
        # 분산 딥러닝 알고리즘을 구현할 때에는 아래 처럼 id 를 구분해서 보낼 필요 없음
        # 그냥 모든 값을 전송하면 됨
        #


        ########## TODO: INTERFACE ##########
        count_entity = unpack('!i', sockRecv(embedding_sock, 4))[0]
        
        # 순수한 embedding 시간을 측정하기 위해서 여기 위치, cpp 가 send 하면 embedding 이 끝난 것                
        embeddingTime = default_timer() - timeNow
        timeNow = default_timer()

        entity_id_list = unpack('!' + 'i' * count_entity, sockRecv(embedding_sock, count_entity * 4))
        entity_vector_list = unpack(precision_string * count_entity * embedding_dim,
            sockRecv(embedding_sock, precision_byte * embedding_dim * count_entity))
        
        entity_vector_list = np.array(entity_vector_list, dtype=np_dtype).reshape(count_entity, embedding_dim)
        entity_vectors = {f"{entities[id_]}_v": v.tostring() for v, id_ in zip(entity_vector_list, entity_id_list)}

        # transG 에 추가되는 분기
        # transG 의 짝수 이터레이션에선 추가로 전송할 게 없음
        #if train_model == 1:
        #
        #    pass

        sockSaveTime = default_timer() - timeNow
        timeNow = default_timer()

        iter_mset(r, entity_vectors)
        redisTime += default_timer() - timeNow

    else:

        # 처리 결과를 받아옴 - GeometricModel save
        # 분산 딥러닝 알고리즘을 구현할 때에는 아래 처럼 id 를 구분해서 보낼 필요 없음
        # 그냥 모든 값을 전송하면 됨
        #

        ########## TODO: INTERFACE ##########
        # transE 에서는 embedding_relation 을 전송
        if train_model == 0:

            count_relation = unpack('!i', sockRecv(embedding_sock, 4))[0]

            # 순수한 embedding 시간을 측정하기 위해서 여기 위치, cpp 가 send 하면 embedding 이 끝난 것                
            embeddingTime = default_timer() - timeNow
            timeNow = default_timer()

            relation_id_list = unpack('!' + 'i' * count_relation, sockRecv(embedding_sock, count_relation * 4))
            relation_vector_list = unpack(precision_string * count_relation * embedding_dim,
                sockRecv(embedding_sock, precision_byte * embedding_dim * count_relation))
            # relation_vectors 전송
            relation_vector_list = np.array(relation_vector_list, dtype=np_dtype).reshape(count_relation, embedding_dim)
            relation_vectors = {f"{relations[id_]}_v": v.tostring() for v, id_ in
                    zip(relation_vector_list, relation_id_list)}
        
        # transG 에 추가되는 분기
        elif train_model == 1:

            count_relation = unpack('!i', sockRecv(embedding_sock, 4))[0]

            embeddingTime = default_timer() - timeNow # 순수한 embedding 시간을 측정하기 위해서 여기 위치, cpp 가 send 하면 embedding 이 끝난 것                
            timeNow = default_timer()

            relation_id_list = unpack('!' + 'i' * count_relation, sockRecv(embedding_sock, count_relation * 4))
            # embedding_clusters 전송
            cluster_vector_list = unpack(precision_string * count_relation * embedding_dim * 21,
                sockRecv(embedding_sock, 21 * precision_byte * embedding_dim * count_relation))
            cluster_vector_list = np.array(cluster_vector_list, dtype=np_dtype).reshape(count_relation, 21 * embedding_dim)
            cluster_vectors = {f"{relations[id_]}_cv": v.tostring() for v, id_ in
                    zip(cluster_vector_list, relation_id_list)}
            # weights_clusters 전송
            weights_clusters_list = unpack(precision_string * count_relation * 21,
                sockRecv(embedding_sock, precision_byte * 21 * count_relation))
            weights_clusters_list = np.array(weights_clusters_list, dtype=np_dtype).reshape(count_relation, 21)
            weights_clusters = {f"{relations[id_]}_wv": v.tostring() for v, id_ in
                    zip(weights_clusters_list, relation_id_list)}
            # size_clusters 전송
            size_clusters_list = unpack('!' + 'i' * count_relation, sockRecv(embedding_sock, 4 * count_relation))
            size_clusters_list = np.array(size_clusters_list, dtype=np.int32)
            size_clusters = {f"{relations[id_]}_s": v.tostring() for v, id_ in zip(size_clusters_list, relation_id_list)}

        sockSaveTime = default_timer() - timeNow
        timeNow = default_timer()

        # transE
        if train_model == 0:
            
            iter_mset(r, relation_vectors)

        # transG
        elif train_model == 1:
            
            iter_mset(r, cluster_vectors)
            iter_mset(r, weights_clusters)
            iter_mset(r, size_clusters)

        redisTime += default_timer() - timeNow

except Exception as e:

    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

    printt('[error] worker > exception occured in iteration - ' + worker_id)
    printt('[error] worker > ' + str(e))
    printt('[error] worker > exception occured in line ' + str(exc_tb.tb_lineno))
    embedding_sock.close()
    sys.exit(1)

workerTotalTime = default_timer() - workerStart
embedding_sock.send(pack('!i', 1234))
modelRunTime = unpack('d', sockRecv(embedding_sock, 8))[0]
embedding_sock.close()

output_times = dict()
output_times["datamodel_sock"] = datamodelTime
output_times["socket_load"] = sockLoadTime
output_times["embedding"] = embeddingTime
output_times["model_run"] = modelRunTime
output_times["socket_save"] = sockSaveTime
output_times["redis"] = redisTime
output_times["worker_total"] = workerTotalTime
output_times = dumps(output_times, protocol = HIGHEST_PROTOCOL)
r.set("%s_%d" % (worker_id, cur_iter), output_times)

sys.exit(0)
