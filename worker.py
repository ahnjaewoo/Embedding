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
preprocess_folder_dir = "%s/preprocess/" % root_dir
train_code_dir = "%s/MultiChannelEmbedding/Embedding.out" % root_dir
temp_folder_dir = "%s/tmp" % root_dir

redis_ip_address = '163.152.29.73'

t_ = time()
# redis에서 embedding vector들 받아오기
r = redis.StrictRedis(host=redis_ip_address, port=6379, db=0)
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
use_socket = False

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

    print("file save time: %f" % (time()-t_))
    del entities_initialized
    del relations_initialized

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

    # 임시로 만들어놓음, barrier 같은 기능
    # 아래 부분을 소켓으로 변경하면 이 줄을 제거
    # embedding_iter_end = struct.unpack('!i', embedding_sock.recv(4))[0]

    # DataModel.hpp 와의 통신 다음에 GeometricModel.hpp 와의 통신이 필요
    # DataModel 생성자 -> GeometricModel load 메소드 -> GeometricModel save 메소드 순서

    # DataModel.hpp 생성자와 통신
    # "../tmp/maxmin_worker_"+ to_string(worker_num) + ".txt"
    # "../tmp/sub_graph_worker_"+ to_string(worker_num) + ".txt"
    # 위의 파일 내용을 전송해야 함

    if int(cur_iter) % 2 == 0:
        # entity 전송
        chunk_anchor, chunk_entity = chunk_data.split('\n')
        chunk_anchor = list(eval(chunk_anchor))
        chunk_entity = chunk_entity.split(' ')

        embedding_sock.send(struct.pack('!i', len(chunk_anchor)))

        for iter_anchor in chunk_anchor:
            embedding_sock.send(struct.pack('!i', int(iter_anchor)))

        embedding_sock.send(struct.pack('!i', len(chunk_entity)))

        for iter_entity in chunk_entity:
            embedding_sock.send(struct.pack('!i', int(iter_entity)))

    else:
        # relation 전송
        sub_graphs = pickle.loads(r.get('sub_graph_{}'.format(worker_id)))
        embedding_sock.send(struct.pack('!i', len(sub_graphs)))

        for (head_id, relation_id, tail_id) in sub_graphs:
            embedding_sock.send(struct.pack('!i', int(head_id)))
            embedding_sock.send(struct.pack('!i', int(relation_id)))
            embedding_sock.send(struct.pack('!i', int(tail_id)))

    # GeometricModel load 메소드와 통신
    """
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

    print("file save time: %f" % (time()-t_))
    del entities_initialized
    del relations_initialized
    """

    # entity_vector 전송
    for i, vector in enumerate(entities_initialized):
        entity_name = str(entities[i])
        embedding_sock.send(struct.pack('!i', len(entity_name)))
        embedding_sock.send(str.encode(entity_name))    # entity string 자체를 전송

        for v in vector:
            embedding_sock.send(struct.pack('d', float(v)))

    # relation_vector 전송
    for i, relation in enumerate(relations_initialized):
        relation_name = str(relations[i])
        embedding_sock.send(struct.pack('!i', len(relation_name)))
        embedding_sock.send(str.encode(relation_name))  # relation string 자체를 전송

        for v in relation:
            embedding_sock.send(struct.pack('d', float(v)))

    del entities_initialized
    del relations_initialized

    w_id = worker_id.split('_')[1]
    t_ = time()
    
    if int(cur_iter) % 2 == 0:
        entity_vectors = dict()

        # 처리 결과를 받아옴
        # GeometricModel.cpp 의 save 에서 처리 - 작업 완료
        # 이 부분을 활성화하면, 위의 barrier (embedding_iter_end) 를 제거해야 함
        count_entity = struct.unpack('!i', embedding_sock.recv(4))[0]

        for entity_idx in range(count_entity):
            temp_entity_vector = list()
            entity_id_len = struct.unpack('!i', embedding_sock.recv(4))[0]
            entity_id = embedding_sock.recv(entity_id_len).decode()

            for dim_idx in range(int(embedding_dim)):
                temp_entity_vector.append(struct.unpack('d', embedding_sock.recv(8))[0])

            entity_vectors[entity_id + '_v'] = pickle.dumps(np.array(temp_entity_vector), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(entity_vectors)
        """
        with open("%s/entity_vectors_updated_{w_id}.txt" % temp_foler_dir, 'r') as f:
            for line in f:
                line = line[:-1].split()
                entity_vectors[line[0] + '_v'] = pickle.dumps(np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(entity_vectors)
        """

    else:
        relation_vectors = dict()

        # 처리 결과를 받아옴
        # GeometricModel.cpp 의 save 에서 처리 - 작업 완료
        # 이 부분을 활성화하면, 위의 barrier (embedding_iter_end) 를 제거해야 함
        count_relation = struct.unpack('!i', embedding_sock.recv(4))[0]

        for relation_idx in range(count_relation):
            temp_relation_vector = list()
            relation_id_len = struct.unpack('!i', embedding_sock.recv(4))[0]
            relation_id = embedding_sock.recv(relation_id_len).decode()

            for dim_idx in range(int(embedding_dim)):
                temp_relation_vector.append(struct.unpack('d', embedding_sock.recv(8))[0])

            relation_vectors[relation_id + '_v'] = pickle.dumps(np.array(temp_relation_vector), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(relation_vectors)
        
        """
        with open("%s/relation_vectors_updated_%s.txt" % (temp_folder_dir, w_id), 'r') as f:
            for line in f:
                line = line[:-1].split()
                relation_vectors[line[0] + '_v'] = pickle.dumps(np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(relation_vectors)
        """

    print("redis server connection time: %f" % (time()-t_))
    print("{}: {} iteration finished!".format(worker_id, cur_iter))

if not use_socket:
    # 이 부분은 호출 대신 socket 통신으로 대체
    # 여기서 C++ 프로그램 호출
    t_ = time()
    proc = Popen([
        train_code_dir, 
        worker_id, cur_iter, embedding_dim, learning_rate, margin, train_iter],
        cwd=preprocess_folder_dir)
    proc.wait()
    print("embedding time: %f" % (time()-t_))

    # 이 부분을 socket 통신으로 대체할 필요가 있음
    # 현재 embeding.cpp 자체에서는 못하고, 그 코드를 타고 들어가야 가능함
    w_id = worker_id.split('_')[1]
    t_ = time()
    if int(cur_iter) % 2 == 0:
        entity_vectors = {}
        with open("%s/entity_vectors_updated_%s.txt" % (temp_folder_dir, w_id), 'r') as f:
            for line in f:
                line = line[:-1].split()
                entity_vectors[line[0] + '_v'] = pickle.dumps(np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(entity_vectors)
    else:
        relation_vectors = {}
        with open("%s/relation_vectors_updated_%s.txt" % (temp_folder_dir, w_id), 'r') as f:
            for line in f:
                line = line[:-1].split()
                relation_vectors[line[0] + '_v'] = pickle.dumps(np.array(line[1:]), protocol=pickle.HIGHEST_PROTOCOL)
        r.mset(relation_vectors)

    print("redis server connection time: %f" % (time()-t_))
    print("{}: {} iteration finished!".format(worker_id, cur_iter))
