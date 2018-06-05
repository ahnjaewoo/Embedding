# coding: utf-8
import networkx as nx
from random import randint
from collections import defaultdict
from time import time
import nxmetis
import sys
import random
import socket
import struct
import logging
import os

partition_num = int(sys.argv[1])
cur_iter = int(sys.argv[2])
anchor_num = int(sys.argv[3])
anchor_interval = int(sys.argv[4])
root_dir = sys.argv[5]
data_root = sys.argv[6]
debugging = sys.argv[7]
temp_folder_dir = "%s/tmp" % root_dir
logging.basicConfig(filename='%s/maxmin.log' % root_dir, filemode='w', level=logging.DEBUG)
logger = logging.getLogger()
handler = logging.StreamHandler(stream=sys.stdout)
logger.addHandler(handler)

if debugging == 'yes':
    logging.basicConfig(filename='%s/master.log' %
                        root_dir, filemode='w', level=logging.DEBUG)
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

elif debugging == 'no':
    def printt(str):
        print(str)


def sockRecv(sock, length):

    data = b''

    while len(data) < length:

        buff = sock.recv(length - len(data))

        if not buff:

            return None

        data = data + buff

    return data

# max-min process 실행, socket 연결
# maxmin.cpp 가 server
# master.py 는 client
t_ = time()
# master 와 maxmin 은 같은 ip 상에서 작동, 포트를 임의로 7847 로 지정
maxmin_addr = '127.0.0.1'
maxmin_port = 7847
maxmin_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
maxmin_sock.bind((maxmin_addr, maxmin_port))
maxmin_sock.listen(1)

master_sock, master_addr = maxmin_sock.accept()

# printt('[info] maxmin > socket connected (master <-> maxmin)')
data_files = ['%s/train.txt' % data_root, '%s/dev.txt' % data_root, '%s/test.txt' % data_root]
output_file = '%s/maxmin_output.txt' % temp_folder_dir
old_anchor_file = '%s/old_anchor.txt' % temp_folder_dir
anchor_dict = dict()
old_anchor = set()

entities = set()
entity_graph = list()
edge_list = list()
entity2id = dict()
connected_entity = defaultdict(set)
anchor = set()
non_anchor_edge_included_vertex = set()
entity_cnt = 0

entity_degree = defaultdict(int)

for file in data_files:

    with open(root_dir + file, 'r') as f:

        for line in f:

            head, relation, tail = line[:-1].split("\t")
            entities.add(head)
            entities.add(tail)

            if head not in entity2id:

                entity2id[head] = entity_cnt
                entity_cnt += 1

            if tail not in entity2id:

                entity2id[tail] = entity_cnt
                entity_cnt += 1
            
            entity_degree[entity2id[head]] += 1
            entity_degree[entity2id[tail]] += 1

with open(root_dir + data_files[0], 'r') as f:

    for line in f:

        head, relation, tail = line[:-1].split("\t")
        entity_graph.append((head, tail))
        connected_entity[entity2id[head]].add(entity2id[tail])
        connected_entity[entity2id[tail]].add(entity2id[head])

entities_id = {entity2id[v] for v in entities}

for (hd, tl) in entity_graph:

    edge_list.append((entity2id[hd], entity2id[tl]))
# printt('[info] maxmin > max-min cut data preprocessing finished (time : {})'.format((time()-t_)))

while True:

    master_status = struct.unpack('!i', sockRecv(master_sock, 4))[0]
    logger.warning(str(master_status)+'\n')
    t_ = time()

    if master_status == 1:
        # 연결을 끊음
        # printt('[info] maxmin > received disconnect signal (master_status = 1)')

        maxmin_sock.close()
        sys.exit(0)

    #partition_num = struct.unpack('!i', sockRecv(master_sock, 4))[0]
    cur_iter = (struct.unpack('!i', sockRecv(master_sock, 4))[0] + 1) // 2
    #anchor_num = struct.unpack('!i', sockRecv(master_sock, 4))[0]
    #anchor_interval = struct.unpack('!i', sockRecv(master_sock, 4))[0]

    if cur_iter == 0:

        anchor_dict = dict()

    else:

        anchor_dict = old_anchor_dict

        for it in range(anchor_interval):

            if it in anchor_dict.keys():

                for a in anchor_dict[it]:

                    old_anchor.add(a)

    for i in range(anchor_num):

        best = None
        best_score = 0

        for vertex in entities_id.difference(anchor.union(old_anchor)):
            # getting degree(v)
            if len(connected_entity[vertex]) <= best_score:

                continue

            score = len(connected_entity[vertex].difference(anchor))

            if score > best_score or (score == best_score and random.choice([True, False])):

                best = vertex
                best_score = score

        if best == None:

            printt('[error] maxmin > no vertex added to anchor')

        else:

            anchor.add(best)

    # printing # of 1st connected entity of anchors
    #tac = defaultdict(bool)
    #for i,v in enumerate(anchor):
    #    for e in connected_entity[v]:
    #        tac[e] = True
    #    temp_cnt = 0
    #    for e in range(entity_cnt):
    #        if tac[e]:
    #            temp_cnt += 1
    #    print("1hop anchor 1 - %d(%d): %d/%d" % (i+1, v, temp_cnt, entity_cnt))
    #print("")
    # printing # of 2nd connected entity of anchors
    #tac = defaultdict(bool)
    #for i,v in enumerate(anchor):
    #    for e in connected_entity[v]:
    #        tac[e] = True
    #        for se in connected_entity[e]:
    #            tac[se] = True
    #    temp_cnt = 0
    #    for e in range(entity_cnt):
    #        if tac[e]:
    #            temp_cnt += 1
    #    print("2hop anchor 1 - %d(%d): %d/%d" % (i+1, v, temp_cnt, entity_cnt))
    #print("")
    # printing # of 3rd connected entity of anchors
    #tac = defaultdict(bool)
    #for i,v in enumerate(anchor):
    #    for e in connected_entity[v]:
    #        tac[e] = True
    #        for se in connected_entity[e]:
    #            tac[se] = True
    #            for te in connected_entity[se]:
    #                tac[te] = True
    #    temp_cnt = 0
    #    for e in range(entity_cnt):
    #        if tac[e]:
    #            temp_cnt += 1
    #    print("3hop anchor 1 - %d(%d): %d/%d" % (i+1, v, temp_cnt, entity_cnt))
    #print("")

    #for i,v in enumerate(anchor):
    #    print('anchor %d(%d): %d' % (i, v, len(connected_entity[v])))

    anchor_dict[cur_iter % anchor_interval] = anchor
    old_anchor_dict = anchor_dict

    # solve the min-cut partition problem of A~, finding A~ and edges
    non_anchor_id = entities_id.difference(anchor)
    non_anchor_edge_list = [(h, t) for (h, t) in edge_list if h in non_anchor_id and t in non_anchor_id]

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

    (edgecuts, parts) = nxmetis.partition(G, nparts=partition_num, node_weight='node_weight')

    # putting residue randomly into non anchor set
    residue = non_anchor_id.difference(non_anchor_edge_included_vertex)
    
    for v in residue:

        parts[randint(0, partition_num - 1)].append(v)

    # printing the number of entities in each paritions
    # printt('[info] maxmin > # of entities in each partitions : [%s]' % " ".join([str(len(p)) for p in parts]))
    master_sock.send(struct.pack('!i', len(list(anchor))))

    for anchor_val in list(anchor):

        master_sock.send(struct.pack('!i', anchor_val))

    for nas in parts:

        master_sock.send(struct.pack('!i', len(nas)))

        for nas_val in nas:

            master_sock.send(struct.pack('!i', nas_val))

    # printt('[info] maxmin > sent anchor and nas to master')          
    # printt('[info] maxmin > max-min cut finished (time : {})'.format((time()-t_)))