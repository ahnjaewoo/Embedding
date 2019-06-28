# coding: utf-8
import logging
import os
import pickle
import socket
import sys
from pickle import HIGHEST_PROTOCOL, dumps, load, loads
from struct import pack, unpack
from subprocess import Popen
from time import sleep
from timeit import default_timer

import numpy as np
import redis

from .model.custom_model import TransEWorker, TransGWorker
from .utils import iter_mget, iter_mset, sockRecv

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

########## INTERFACE ##########
if train_model == 0:
    worker = TransEWorker(r, embedding_sock, embedding_dim)
else:
    worker = TransGWorker(r, embedding_sock, embedding_dim)

worker.load_initialized_vectors()

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


    ########## INTERFACE ##########
    worker.get_entities()
    worker.get_relations()

    sockLoadTime = default_timer() - timeNow
    
    timeNow = default_timer()

    if cur_iter % 2 == 0:

        # 처리 결과를 받아옴 - GeometricModel save
        # 분산 딥러닝 알고리즘을 구현할 때에는 아래 처럼 id 를 구분해서 보낼 필요 없음
        # 그냥 모든 값을 전송하면 됨
        #


        ########## INTERFACE ##########
        # 순수한 embedding 시간을 측정하기 위해서 여기 위치, cpp 가 send 하면 embedding 이 끝난 것                
        embeddingTime = default_timer() - timeNow
        timeNow = default_timer()

        worker.get_entities()

        # transG 에 추가되는 분기
        # transG 의 짝수 이터레이션에선 추가로 전송할 게 없음
        #if train_model == 1:
        #
        #    pass

        sockSaveTime = default_timer() - timeNow
        timeNow = default_timer()

        worker.update_entities()
        redisTime += default_timer() - timeNow

    else:

        # 처리 결과를 받아옴 - GeometricModel save
        # 분산 딥러닝 알고리즘을 구현할 때에는 아래 처럼 id 를 구분해서 보낼 필요 없음
        # 그냥 모든 값을 전송하면 됨
        #

        ########## INTERFACE ##########
        # transE 에서는 embedding_relation 을 전송
        embeddingTime = default_timer() - timeNow
        timeNow = default_timer()

        worker.get_relations()        

        sockSaveTime = default_timer() - timeNow
        timeNow = default_timer()

        worker.update_relations()

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
