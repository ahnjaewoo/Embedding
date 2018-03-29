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

print("redis server connection time: %f" % (time() - t_))
logger.warning("redis server connection time: %f" % (time() - t_))

t_ = time()
use_socket = True

if not use_socket:
    if int(cur_iter) % 2 == 0:
        # 이 부분을 socket 으로 DataModel.hpp, Model.hpp 로 전송해줘야 함
        with open("%s/maxmin_%s.txt" % (temp_folder_dir, worker_id), 'w') as f:
            f.write(chunk_data)

    else:
        sub_graphs = pickle.loads(r.get('sub_graph_%s' % worker_id))

        # 이 부분을 socket 으로 DataModel.hpp, Model.hpp 로 전송해줘야 함
        with open("%s/sub_graph_%s.txt" % (temp_folder_dir, worker_id), 'w') as f:

            for (head_id, relation_id, tail_id) in sub_graphs:
                f.write("{} {} {}\n".format(head_id, relation_id, tail_id))

    # GeometricModel.hpp 의 load 에 전송
    # matrix를 text로 빨리 저장하는 법 찾기!
    with open("%s/entity_vectors.txt" % temp_folder_dir, 'w') as f:
        for i, vector in enumerate(entities_initialized):
            f.write(str(entities[i]) + "\t")
            f.write(" ".join([str(v) for v in vector]) + '\n')

    # GeometricModel.hpp 의 load 에 전송
    with open("%s/relation_vectors.txt" % temp_folder_dir, 'w') as f:
        for i, relation in enumerate(relations_initialized):
            f.write(str(relations[i]) + "\t")
            f.write(" ".join([str(v) for v in relation]) + '\n')

    print("file save time: %f" % (time() - t_))
    logger.warning("file save time: %f" % (time() - t_))
    del entities_initialized
    del relations_initialized

# embedding.cpp 와 socket 통신
# worker 가 실행될 때 전달받은 ip 와 port 로 접속
# Embedding.cpp 가 server, 프로세느는 master.py 가 생성
# worker.py 가 client
if use_socket:

    # 첫 iteration 에서눈 Embedding.cpp 의 실행, 소켓 생성을 기다림
    #if cur_iter == 0:
    #    tt.sleep(2)

    embedding_addr = '0.0.0.0'
    # worker_id 를 기반으로 포트를 생성
    embedding_port = 49900 + 5 * int(worker_id.split('_')[1]) + int(cur_iter) % 5
    #embedding_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #embedding_sock.connect((embedding_addr, embedding_port))

    print('port number of ' + worker_id, ' : ' + str(embedding_port) + ' - worker.py')
    logger.warning('port number of ' + worker_id, ' : ' + str(embedding_port) + ' - worker.py')

    while True:
        try:
            embedding_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            embedding_sock.connect((embedding_addr, embedding_port))
            break
        except (TimeoutError, ConnectionRefusedError):
            tt.sleep(1)

    print('socket connected to embedding.cpp in worker.py')
    logger.warning('socket connected to embedding.cpp in worker.py')
    
    # 연산 요청 메시지
    embedding_sock.send(struct.pack('!i', 0))
    # int 임시 땜빵, 매우 큰 문제
    embedding_sock.send(struct.pack('!i', int(worker_id.split('_')[1])))
    embedding_sock.send(struct.pack('!i', int(cur_iter)))            # int
    embedding_sock.send(struct.pack('!i', int(embedding_dim)))       # int
    embedding_sock.send(struct.pack('d', float(learning_rate)))      # double
    embedding_sock.send(struct.pack('d', float(margin)))             # double
    embedding_sock.send(struct.pack('!i', int(data_root_id)))        # int

    # DataModel 생성자 -> GeometricModel load 메소드 -> GeometricModel save 메소드 순서로 통신

    if int(cur_iter) % 2 == 0:
        # entity 전송 - DataModel 생성자
        chunk_anchor, chunk_entity = chunk_data.split('\n')
        chunk_anchor = chunk_anchor.split(' ')
        chunk_entity = chunk_entity.split(' ')

        embedding_sock.send(struct.pack('!i', len(chunk_anchor)))

        for iter_anchor in chunk_anchor:
            embedding_sock.send(struct.pack('!i', int(iter_anchor)))

        embedding_sock.send(struct.pack('!i', len(chunk_entity)))

        for iter_entity in chunk_entity:
            embedding_sock.send(struct.pack('!i', int(iter_entity)))

    else:
        # relation 전송 - DataModel 생성자
        sub_graphs = pickle.loads(r.get('sub_graph_{}'.format(worker_id)))
        embedding_sock.send(struct.pack('!i', len(sub_graphs)))

        for (head_id, relation_id, tail_id) in sub_graphs:
            embedding_sock.send(struct.pack('!i', int(head_id)))
            embedding_sock.send(struct.pack('!i', int(relation_id)))
            embedding_sock.send(struct.pack('!i', int(tail_id)))

    # entity_vector 전송 - GeometricModel load
    for i, vector in enumerate(entities_initialized):
        entity_name = str(entities[i])
        embedding_sock.send(struct.pack('!i', len(entity_name)))
        embedding_sock.send(str.encode(entity_name))    # entity string 자체를 전송







        #embedding_sock.send(struct.pack('!i', entity2id[entity_name])) # entity id 를 int 로 전송






        for v in vector:
            embedding_sock.send(struct.pack('d', float(v)))

    # relation_vector 전송 - GeometricModel load
    for i, relation in enumerate(relations_initialized):
        relation_name = str(relations[i])
        embedding_sock.send(struct.pack('!i', len(relation_name)))
        embedding_sock.send(str.encode(relation_name))  # relation string 자체를 전송






        #embedding_sock.send(struct.pack('!i', relation2id[relation_name])) # relation id 를 int 로 전송






        for v in relation:
            embedding_sock.send(struct.pack('d', float(v)))

    del entities_initialized
    del relations_initialized

    w_id = worker_id.split('_')[1]
    t_ = time()

    if int(cur_iter) % 2 == 0:

        entity_vectors = dict()

        # 처리 결과를 받아옴 - GeometricModel save
        count_entity = struct.unpack('!i', embedding_sock.recv(4))[0]

        for entity_idx in range(count_entity):
            temp_entity_vector = list()
            entity_id_len = struct.unpack('!i', embedding_sock.recv(4))[0]
            entity_id = embedding_sock.recv(entity_id_len).decode()






            #entity_id = struct.unpack('!i', embedding_sock.recv(4))[0]     # entity_id 를 int 로 받음






            for dim_idx in range(int(embedding_dim)):
                temp_entity_vector.append(
                    struct.unpack('d', embedding_sock.recv(8))[0])

            entity_vectors[entity_id + '_v'] = pickle.dumps(
                np.array(temp_entity_vector), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(entity_vectors)

    else:

        relation_vectors = dict()

        # 처리 결과를 받아옴 - GeometricModel save
        count_relation = struct.unpack('!i', embedding_sock.recv(4))[0]

        for relation_idx in range(count_relation):
            temp_relation_vector = list()
            relation_id_len = struct.unpack('!i', embedding_sock.recv(4))[0]
            relation_id = embedding_sock.recv(relation_id_len).decode()






            #relation_id = struct.unpack('!i', embedding_sock.recv(4))[0]   # relation_id 를 int 로 바음







            for dim_idx in range(int(embedding_dim)):
                temp_relation_vector.append(
                    struct.unpack('d', embedding_sock.recv(8))[0])

            relation_vectors[relation_id + '_v'] = pickle.dumps(
                np.array(temp_relation_vector), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(relation_vectors)

    print("redis server connection time: %f" % (time() - t_))
    logger.warning("redis server connection time: %f" % (time() - t_))
    print("{}: {} iteration finished!".format(worker_id, cur_iter))
    logger.warning("{}: {} iteration finished!".format(worker_id, cur_iter))


if not use_socket:
    # 이 부분은 호출 대신 socket 통신으로 대체
    # 여기서 C++ 프로그램 호출
    t_ = time()
    proc = Popen([
        train_code_dir,
        worker_id, cur_iter, embedding_dim, learning_rate, margin, train_iter, data_root_id],
        cwd=preprocess_folder_dir)
    proc.wait()
    print("embedding time: %f" % (time() - t_))
    logger.warning("embedding time: %f" % (time() - t_))

    # 이 부분을 socket 통신으로 대체할 필요가 있음
    # 현재 embeding.cpp 자체에서는 못하고, 그 코드를 타고 들어가야 가능함
    w_id = worker_id.split('_')[1]
    t_ = time()
    if int(cur_iter) % 2 == 0:
        entity_vectors = {}
        with open("%s/entity_vectors_updated_%s.txt" % (temp_folder_dir, w_id), 'r') as f:
            for line in f:
                line = line[:-1].split()
                entity_vectors[line[0] + '_v'] = pickle.dumps(
                    np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(entity_vectors)
    else:
        relation_vectors = {}
        with open("%s/relation_vectors_updated_%s.txt" % (temp_folder_dir, w_id), 'r') as f:
            for line in f:
                line = line[:-1].split()
                relation_vectors[line[0] + '_v'] = pickle.dumps(
                    np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(relation_vectors)

    print("redis server connection time: %f" % (time() - t_))
    logger.warning("redis server connection time: %f" % (time() - t_))
    print("{}: {} iteration finished!".format(worker_id, cur_iter))
    logger.warning("{}: {} iteration finished!".format(worker_id, cur_iter))
