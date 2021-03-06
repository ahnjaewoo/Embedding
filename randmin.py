# coding: utf-8
import networkx as nx
from random import randint
from random import choice
from collections import defaultdict
from struct import pack, unpack
import nxmetis
import os
import sys
import socket
import logging
import random

def sockRecv(sock, length):
    
    data = b''

    while len(data) < length:

        buff = sock.recv(length - len(data))

        if not buff:

            return None

        data = data + buff

    return data

partition_num = int(sys.argv[1])
cur_iter = int(sys.argv[2])
anchor_num = int(sys.argv[3])
anchor_interval = int(sys.argv[4])
root_dir = sys.argv[5]
data_root = sys.argv[6]
debugging = sys.argv[7]
maxmin_port = int(sys.argv[8])

if debugging == 'yes':
    logging.basicConfig(filename='%s/master.log' % root_dir, filemode='w', level=logging.DEBUG)
    logger = logging.getLogger()
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)
    loggerOn = True

    def printt(str_):

        global loggerOn

        print(str_)

        if loggerOn:

            logger.warning(str_ + '\n')

    def handle_exception(exc_type, exc_value, exc_traceback):

        if issubclass(exc_type, KeyboardInterrupt):

            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.error("exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

elif debugging == 'no':
    def printt(str_):
        print(str_)

#printt('[info] maxmin > socket connected (master <-> maxmin)')

# 73
data_files = ('%s/train.txt' % data_root, '%s/dev.txt' % data_root, '%s/test.txt' % data_root)

# 71
#data_files = ('/home/data/dbpedia/1mill/train.ttl', '/home/data/dbpedia/1mill/dev.ttl', '/home/data/dbpedia/1mill/test.ttl')

anchor_dict = dict()
old_anchor = set()

entities = set()
entity_graph = list()
edge_list = list()
entity2id = dict()
connected_entity = defaultdict(set)
anchor = set()
entity_cnt = 0
old_anchor_dict = None

entity_degree = defaultdict(int)
entities_add = entities.add
entity_graph_append = entity_graph.append
edge_list_append = edge_list.append

for file in data_files:

    # 73
    with open(root_dir + file, 'r') as f:

    # 71
    #with open(file, 'r') as f:

        for line in f:

            head, relation, tail = line[:-1].split()
            entities_add(head)
            entities_add(tail)

            if head not in entity2id:

                entity2id[head] = entity_cnt
                entity_cnt += 1

            if tail not in entity2id:

                entity2id[tail] = entity_cnt
                entity_cnt += 1

            entity_degree[entity2id[head]] += 1
            entity_degree[entity2id[tail]] += 1

# 73
with open(root_dir + data_files[0], 'r') as f:

# 71
#with open(data_files[0], 'r') as f:

    for line in f:

        head, relation, tail = line[:-1].split()
        entity_graph_append((head, tail))
        connected_entity[entity2id[head]].add(entity2id[tail])
        connected_entity[entity2id[tail]].add(entity2id[head])

entities_id = {entity2id[v] for v in entities}

for (hd, tl) in entity_graph:

    edge_list_append((entity2id[hd], entity2id[tl]))

# max-min process 실행, socket 연결
# maxmin.cpp 가 server
# master.py 는 client
# master 와 maxmin 은 같은 ip 상에서 작동, 포트는 master 가 임의로 설정
maxmin_addr = '127.0.0.1'
maxmin_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
maxmin_sock.bind((maxmin_addr, maxmin_port))
maxmin_sock.listen(1)

master_sock, master_addr = maxmin_sock.accept()

try:
    while True:

        #printt('[info] maxmin > ready')
        non_anchor_edge_included_vertex = set()

        master_status = unpack('!i', sockRecv(master_sock, 4))[0]

        if master_status == 1:
            # 연결을 끊음
            #printt('[info] maxmin > finish due to disconnect signal')
            maxmin_sock.close()
            sys.exit(0)

        cur_iter = (unpack('!i', sockRecv(master_sock, 4))[0] + 1) // 2
        #printt('[info] maxmin > start')

        if cur_iter == 0:

            anchor_dict = dict()

        else:

            anchor_dict = old_anchor_dict
            anchor = set()
            old_anchor = set()

            for it in range(anchor_interval):

                if it in anchor_dict:
                    
                    old_anchor.update(anchor_dict[it])

        # 한 방에 anchor_num만큼 랜덤으로 뽑아버린다.
        anchor = set(random.sample(entities_id.difference(anchor.union(old_anchor)), anchor_num))

        anchor_dict[cur_iter % anchor_interval] = anchor
        old_anchor_dict = anchor_dict

        # solve the min-cut partition problem of A~, finding A~ and edges
        non_anchor_id = entities_id.difference(anchor)
        non_anchor_edge_list = [(h, t) for h, t in edge_list if h in non_anchor_id and t in non_anchor_id]

        for (h, t) in non_anchor_edge_list:

            non_anchor_edge_included_vertex.add(h)
            non_anchor_edge_included_vertex.add(t)

        # constructing nx.Graph and using metis in order to get min-cut partition
        G = nx.Graph()
        G.add_edges_from(non_anchor_edge_list)

        for node, degree in entity_degree.items():

            if node in G:

                G.node[node]['node_weight'] = degree

        options = nxmetis.MetisOptions(     # objtype=1 => vol
            ptype=-1, objtype=1, ctype=-1, iptype=-1, rtype=-1, ncuts=-1,
            nseps=-1, numbering=-1, niter=cur_iter, seed=-1, minconn=-1, no2hop=-1,
            contig=-1, compress=-1, ccorder=-1, pfactor=-1, ufactor=-1, dbglvl=-1)

        edgecuts, parts = nxmetis.partition(G, nparts=partition_num, node_weight='node_weight')

        # putting residue randomly into non anchor set
        residue = non_anchor_id.difference(non_anchor_edge_included_vertex)

        for v in residue:

            parts[randint(0, partition_num - 1)].append(v)

        # printing the number of entities in each paritions
        printt('[info] maxmin > # of entities in each partitions : [%s]' % " ".join([str(len(p)) for p in parts]))

        # 원소 여러 개를 한 번에 전송
        master_sock.send(pack('!i', len(list(anchor))))
        master_sock.send(pack('!' + 'i' * len(list(anchor)), *list(anchor)))

        for nas in parts:

            master_sock.send(pack('!i', len(nas)))
            master_sock.send(pack('!' + 'i' * len(nas), *nas))

except Exception as e:

    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

    printt('[error] maxmin > ' + str(e))
    printt('[error] maxmin > exception occured in line ' + str(exc_tb.tb_lineno))

finally:

    printt('[info] maxmin > finish')
    master_sock.close()
