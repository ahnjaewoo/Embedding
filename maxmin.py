# coding: utf-8
import networkx as nx
from random import randint
from collections import defaultdict
from time import time
from logging import warning
import nxmetis
import sys
import random
import socket
import struct

use_socket = True
root_dir = sys.argv[5]
temp_folder_dir = "%s/tmp" % root_dir

# max-min process 실행, socket 연결
# maxmin.cpp 가 server
# master.py 는 client
if use_socket:
    t_ = time()
    # master 와 maxmin 은 같은 ip 상에서 작동, 포트를 임의로 7847 로 지정
    maxmin_addr = '127.0.0.1'
    maxmin_port = 7847
    maxmin_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    maxmin_sock.bind((maxmin_addr, maxmin_port))
    maxmin_sock.listen(1)

    master_sock, master_addr = maxmin_sock.accept()

    print("socket between master and maxmin connected - maxmin.py")
    warning("socket between master and maxmin connected - maxmin.py")

    root = 'fb15k'
    data_files = ['/train.txt','/dev.txt', '/test.txt']
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
        with open(root+file, 'r') as f:
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

    with open(root+data_files[0], 'r') as f:
        for line in f:
            head, relation, tail = line[:-1].split("\t")
            entity_graph.append((head, tail))
            connected_entity[entity2id[head]].add(entity2id[tail])
            connected_entity[entity2id[tail]].add(entity2id[head])

    entities_id = {entity2id[v] for v in entities}

    for (hd, tl) in entity_graph:
        edge_list.append((entity2id[hd], entity2id[tl]))
    print("max-min cut data preprocessing finished - max-min preprocessing time: {}".format((time()-t_)))
    warning("max-min cut data preprocessing finished - max-min preprocessing time: {}".format((time()-t_)))

    while True:
        master_status = struct.unpack('!i', master_sock.recv(4))[0]
        t_ = time()

        if master_status == 1:
            # 연결을 끊음
            maxmin_sock.close()
            sys.exit(0)

        partition_num = struct.unpack('!i', master_sock.recv(4))[0]
        cur_iter = (struct.unpack('!i', master_sock.recv(4))[0] + 1) // 2
        anchor_num = struct.unpack('!i', master_sock.recv(4))[0]
        anchor_interval = struct.unpack('!i', master_sock.recv(4))[0]

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
                print("no vertex added to anchor")
                warning("no vertex added to anchor")
            else:
                anchor.add(best)

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
        print('# of entities in each partitions: [%s]' % " ".join([str(len(p)) for p in parts]))
        warning('# of entities in each partitions: [%s]' % " ".join([str(len(p)) for p in parts]))
        master_sock.send(struct.pack('!i', len(list(anchor))))

        for anchor_val in list(anchor):

            master_sock.send(struct.pack('!i', anchor_val))

        for nas in parts:

            master_sock.send(struct.pack('!i', len(nas)))

            for nas_val in nas:

                master_sock.send(struct.pack('!i', nas_val))

        print("max-min cut finished - max-min time: {}".format((time()-t_)))
        warning("max-min cut finished - max-min time: {}".format((time()-t_)))










if not use_socket:
    t_ = time()
    root = 'fb15k'
    data_files = ['/train.txt','/dev.txt', '/test.txt']
    output_file = '%s/maxmin_output.txt' % temp_folder_dir
    old_anchor_file = '%s/old_anchor.txt' % temp_folder_dir
    partition_num = int(sys.argv[1])
    cur_iter = (int(sys.argv[2]) + 1) // 2
    anchor_num = int(sys.argv[3])
    anchor_interval = int(sys.argv[4])
    anchor_dict = {}
    old_anchor = set()

    entities = set()
    entity_graph = []
    edge_list = []
    entity2id = dict()
    connected_entity = defaultdict(set)
    anchor = set()
    non_anchor_edge_included_vertex = set()
    entity_cnt = 0

    entity_degree = defaultdict(int)

    for file in data_files:
        with open(root+file, 'r') as f:
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

    with open(root+data_files[0], 'r') as f:
        for line in f:
            head, relation, tail = line[:-1].split("\t")
            entity_graph.append((head, tail))
            connected_entity[entity2id[head]].add(entity2id[tail])
            connected_entity[entity2id[tail]].add(entity2id[head])

    entities_id = {entity2id[v] for v in entities}

    for (hd, tl) in entity_graph:
        edge_list.append((entity2id[hd], entity2id[tl]))

    # select anchor via max-cut bipartition, read old_anchor.txt if it exists
    try:
        # if current iteration is 0, then initialize old_anchor.txt as empty space
        if cur_iter == 0:
            with open(old_anchor_file, 'w') as fwrite:
                fwrite.write("")

        #read as old anchor set
        with open(old_anchor_file, 'r') as f:
            for it,line in enumerate(f):
                if line not in ['\n', '\r\n']:
                    anchor_dict[it] = set(int(i) for i in line.split(" "))
        for it in range(anchor_interval):
            if it in anchor_dict.keys():
                for a in anchor_dict[it]:
                    old_anchor.add(a)
    except:
        anchor_dict = {}

    # while len(anchor) < anchor_num:
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
            print("no vertex added to anchor")
            warning("no vertex added to anchor")
        else:
            anchor.add(best)
    #writing anchor to old anchor file
    anchor_dict[cur_iter % anchor_interval] = anchor
    with open(old_anchor_file, 'w') as fwrite:
        for it in range(anchor_interval):
            if it in anchor_dict.keys():
                fwrite.write(" ".join([str(i) for i in anchor_dict[it]])+"\n")

    # solve the min-cut partition problem of A~, finding A~ and edges
    non_anchor_id = entities_id.difference(anchor)
    non_anchor_edge_list = [(h, t) for (h, t) in edge_list if h in non_anchor_id and t in non_anchor_id]
    for (h, t) in non_anchor_edge_list:
        non_anchor_edge_included_vertex.add(h)
        non_anchor_edge_included_vertex.add(t) 

    # constructing nx.Graph and using metis in order to get min-cut partition
    G = nx.Graph()
    # G에 있는 각 노드들에 대해 node weight 부여해야!
    # ex)   G.node[0]['node_weight'] = 1
    #       nxmetis.partition(G, nparts=partition_num, node_weight='node_weight')        
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
    print('# of entities in each partitions: [%s]' % " ".join([str(len(p)) for p in parts]))
    warning('# of entities in each partitions: [%s]' % " ".join([str(len(p)) for p in parts]))

    # writing output file
    with open(output_file, "w") as fwrite:
        fwrite.write(" ".join([str(i) for i in anchor])+"\n")
        for nas in parts:
            fwrite.write(" ".join([str(i) for i in nas])+"\n")

    print("max-min cut finished - max-min time: {}".format((time()-t_)))
    warning("max-min cut finished - max-min time: {}".format((time()-t_)))
