# coding: utf-8
from subprocess import Popen
from time import time
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
root_dir = "/home/rudvlf0413/distributedKGE/Embedding"

# redis에서 embedding vector들 받아오기
t_ = time()

r = redis.StrictRedis(host='163.152.29.73', port=6379, db=0)
entities = pickle.loads(r.get('entities'))
relations = pickle.loads(r.get('relations'))
entity_id = r.mget(entities)
relation_id = r.mget(relations)
entities_initialized = r.mget([entity+'_v' for entity in entities])
relations_initialized = r.mget([relation+'_v' for relation in relations])

entity_id = {entity: int(entity_id[i]) for i, entity in enumerate(entities)}
relation_id = {relation: int(relation_id[i]) for i, relation in enumerate(relations)}

entities_initialized = [pickle.loads(v) for v in entities_initialized]
relations_initialized = [pickle.loads(v) for v in relations_initialized]

print("redis server connection time: %f" % (time()-t_))

t_ = time()

if int(cur_iter) % 2 == 0:



    # 이 부분을 socket 으로 DataModel.hpp, Model.hpp 로 전송해줘야 함




    with open(f"{root_dir}/tmp/maxmin_{worker_id}.txt", 'w') as f:
        
        f.write(chunk_data)

else:
    
    sub_graphs = pickle.loads(r.get(f'sub_graph_{worker_id}'))




    # 이 부분을 socket 으로 DataModel.hpp, Model.hpp 로 전송해줘야 함






    with open(f"{root_dir}/tmp/sub_graph_{worker_id}.txt", 'w') as f:
        
        for (head_id, relation_id, tail_id) in sub_graphs:
        
            f.write(f"{head_id} {relation_id} {tail_id}\n")










# GeometricModel.hpp 의 load 에 전송

# matrix를 text로 빨리 저장하는 법 찾기!
with open("./tmp/entity_vectors.txt", 'w') as f:
    for i, vector in enumerate(entities_initialized):
        f.write(str(entities[i]) + "\t")
        f.write(" ".join([str(v) for v in vector]) + '\n')

# GeometricModel.hpp 의 load 에 전송
with open("./tmp/relation_vectors.txt", 'w') as f:
    for i, relation in enumerate(relations_initialized):
        f.write(str(relations[i]) + "\t")
        f.write(" ".join([str(v) for v in relation]) + '\n')

print("file save time: %f" % (time()-t_))
del entities_initialized
del relations_initialized















use_socket = False

# embedding.cpp 와 socket 통신
# master 에서 embedding.cpp 를 실행해놓고, worker 는 접속만 함
# worker 가 실행될 때 전달받은 ip 와 port 로 접속
# worker 는 client, embedding 은 server
if use_socket:

    # 첫 iteration 에서 embedding.cpp 가 실행되고 소켓을 열기까지 기다림
    if cur_iter == 0:

        tt.sleep(2)

    embedding_addr = '0.0.0.0'  # 이게 맞는 건지 확실치 않음
    embedding_port = 49900 + int(worker_id.split('_')[1]) # worker_id 를 기반으로 포트를 생성
    embedding_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    embedding_sock.connect((embedding_addr, embedding_port))


    embedding_sock.send(struct.pack('!i', 0))                        # 연산 요청 메시지
    embedding_sock.send(struct.pack('!i', int(worker_id.split('_')[1])))           # int 임시 땜빵, 매우 큰 문제
    embedding_sock.send(struct.pack('!i', int(cur_iter)))            # int
    embedding_sock.send(struct.pack('!i', int(embedding_dim)))       # int
    embedding_sock.send(struct.pack('d', float(learning_rate)))      # double   endian 때문에 문제 생길 수 있음
    embedding_sock.send(struct.pack('d', float(margin)))             # double   endian 때문에 문제 생길 수 있음


    # DataModel.hpp 와의 통신 다음에 GeometricModel.hpp 와의 통신이 필요
    # DataModel 생성자 -> GeometricModel load 메소드 -> GeometricModel save 메소드 순서










    # DataModel.hpp 와의 통신 - load 메소드
    # "../tmp/maxmin_worker_"+ to_string(worker_num) + ".txt"
    # "../tmp/sub_graph_worker_"+ to_string(worker_num) + ".txt"
    # 위의 파일 내용을 DataModel.hpp 로 전송해야 함
    # chunk_data 변수의 내용을 전송












    # 임시로 만들어놓음, barrier 같은 기능
    # 아래 부분을 소켓으로 변경하면 이 줄을 제거
    embedding_iter_end = struct.unpack('!i', embedding_sock.recv(4))[0]


    # 이 부분을 socket 통신으로 대체할 필요가 있음
    # 현재 embeding.cpp 자체에서는 못하고, 그 코드를 타고 들어가야 가능함
    w_id = worker_id.split('_')[1]
    t_ = time()
    if int(cur_iter) % 2 == 0:
        entity_vectors = dict()

        # socket 에서 처리 결과를 받아옴
        # GeometricModel.cpp 의 save 에서 처리
        # 이 부분을 활성화하면, 위의 barrier 를 제거해야 함
        """
        count_entity = struct.unpack('!i', embedding_sock.recv(4))[0]


        
        for entity_idx in range(count_entity):

            temp_entity_vector = list()
            entity_id_len = struct.unpack('!i', embedding_sock.recv(4))[0]
            entity_id = struct.unpack('!s', embedding_sock.recv(entity_id_len))[0]

            for dim_idx in range(int(embedding_dim)):

                temp_entity_vector.append(struct.unpack('d', embedding_sock.recv(8))[0])

            entity_vectors[entity_id + '_v'] = pickle.dumps(np.array(temp_entity_vector), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(entity_vectors)
        """

        with open(f"{root_dir}/tmp/entity_vectors_updated_{w_id}.txt", 'r') as f:
            for line in f:
                line = line[:-1].split()
                entity_vectors[line[0] + '_v'] = pickle.dumps(np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(entity_vectors)
   
    else:
        relation_vectors = dict()


        # socket 에서 처리 결과를 받아옴
        # GeometricModel.cpp 의 save 에서 처리
        # 이 부분을 활성화하면, 위의 barrier 를 제거해야 함
        """
        count_relation = struct.unpack('!i', embedding_sock.recv(4))[0]

        for relation_idx in range(count_relation):

            temp_relation_vector = list()
            relation_id_len = struct.unpack('!i', embedding_sock.recv(4))[0]
            relation_id = struct.unpack('!s', embedding_sock.recv(relation_id_len))[0]

            for dim_idx in range(int(embedding_dim)):

                temp_relation_vector.append(struct.unpack('d', embedding_sock.recv(8))[0])

            relation_vectors[relation_id + '_v'] = pickle.dumps(np.array(temp_relation_vector), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(relation_vectors)
        """



        with open(f"{root_dir}/tmp/relation_vectors_updated_{w_id}.txt", 'r') as f:
            for line in f:
                line = line[:-1].split()
                relation_vectors[line[0] + '_v'] = pickle.dumps(np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(relation_vectors)

    print("redis server connection time: %f" % (time()-t_))
    print(f"{worker_id}: {cur_iter} iteration finished!")





if not use_socket:
    # 이 부분은 호출 대신 socket 통신으로 대체
    # 여기서 C++ 프로그램 호출
    t_ = time()
    proc = Popen([
        f"{root_dir}/MultiChannelEmbedding/Embedding.out", 
        worker_id, cur_iter, embedding_dim, learning_rate, margin, train_iter],
        cwd=f'{root_dir}/preprocess/')
    proc.wait()
    print("embedding time: %f" % (time()-t_))


    # 이 부분을 socket 통신으로 대체할 필요가 있음
    # 현재 embeding.cpp 자체에서는 못하고, 그 코드를 타고 들어가야 가능함
    w_id = worker_id.split('_')[1]
    t_ = time()
    if int(cur_iter) % 2 == 0:
        entity_vectors = {}
        with open(f"{root_dir}/tmp/entity_vectors_updated_{w_id}.txt", 'r') as f:
            for line in f:
                line = line[:-1].split()
                entity_vectors[line[0] + '_v'] = pickle.dumps(np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(entity_vectors)
    else:
        relation_vectors = {}
        with open(f"{root_dir}/tmp/relation_vectors_updated_{w_id}.txt", 'r') as f:
            for line in f:
                line = line[:-1].split()
                relation_vectors[line[0] + '_v'] = pickle.dumps(np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(relation_vectors)

    print("redis server connection time: %f" % (time()-t_))
    print(f"{worker_id}: {cur_iter} iteration finished!")
