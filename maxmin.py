# coding: utf-8
import networkx as nx
from random import randint
from collections import defaultdict
from time import time
import nxmetis
import sys


t_ = time()
root = 'fb15k'
data_files = ['/train.txt','/dev.txt', '/test.txt']
output_file = 'tmp/maxmin_output.txt'
old_anchor_file = 'tmp/old_anchor.txt'
partition_num = int(sys.argv[1])
k = 10

entities = set()
entity_graph = []
edge_list = []
entity2id = dict()
connected_entity = defaultdict(set)
anchor = set()
non_anchor_edge_included_vertex = set()
entity_cnt = 0

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

with open(root+data_files[0], 'r') as f:
    for line in f:
        entity_graph.append((head, tail))
        connected_entity[entity2id[head]].add(entity2id[tail])
        connected_entity[entity2id[tail]].add(entity2id[head])

entities_id = {entity2id[v] for v in entities}

for (hd, tl) in entity_graph:
    edge_list.append((entity2id[hd], entity2id[tl]))

# select anchor via max-cut bipartition, read old_anchor.txt if it exists
try:
    with open(old_anchor_file, 'r') as f:
        for line in f:
            old_anchor = set(int(i) for i in line.split(" "))
except:
    old_anchor = set()

# while len(anchor) < k:
for i in range(k):
    best = None
    best_score = 0
    for vertex in entities_id.difference(anchor.union(old_anchor)):
        # getting degree(v)
        if len(connected_entity[vertex]) <= best_score:
            continue

        score = len(connected_entity[vertex].difference(anchor))
        if score > best_score:
            best = vertex
            best_score = score

    if best == None:
        print("no vertex added to anchor")
    else:
        anchor.add(best)

with open(old_anchor_file, 'w') as fwrite:
    fwrite.write(" ".join([str(i) for i in anchor]))

# solve the min-cut partition problem of A~, finding A~ and edges
non_anchor = entities.difference(anchor)
non_anchor_id = {entity2id[v] for v in non_anchor}
non_anchor_edge_list = [(h, t) for (h, t) in edge_list if h in non_anchor_id and t in non_anchor_id]
for (h, t) in non_anchor_edge_list:
    non_anchor_edge_included_vertex.add(h)
    non_anchor_edge_included_vertex.add(t)

# constructing nx.Graph and using metis in order to get min-cut partition
G = nx.Graph()
G.add_edges_from(non_anchor_edge_list)

options = nxmetis.MetisOptions(     # objtype=1 => vol
    ptype=-1, objtype=1, ctype=-1, iptype=-1, rtype=-1, ncuts=-1,
    nseps=-1, numbering=-1, niter=-1, seed=-1, minconn=-1, no2hop=-1,
    contig=-1, compress=-1, ccorder=-1, pfactor=-1, ufactor=-1, dbglvl=-1)

(edgecuts, parts) = nxmetis.partition(G, nparts=partition_num)

# putting residue randomly into non anchor set
residue = non_anchor_id.difference(non_anchor_edge_included_vertex)
for v in residue:
    parts[randint(0, partition_num - 1)].append(v)

# writing output file
with open(output_file, "w") as fwrite:
    fwrite.write(" ".join([str(i) for i in anchor])+"\n")
    for nas in parts:
        fwrite.write(" ".join([str(i) for i in nas])+"\n")

print("max-min cut finished - max-min time: {}".format((time()-t_)))
