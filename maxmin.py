# coding: utf-8

import networkx as nx
import nxmetis
from random import randint
from collections import defaultdict
from time import time


t_ = time()
data_file = 'fb15k/train.txt'
output_file = 'tmp/maxmin_output.txt'
old_anchor_file = 'tmp/old_anchor.txt'
partition_num = 20
k = 10

entities = set()
entity_graph = []
edge_list = []
connected_entity = defaultdict(set)
anchor = set()
non_anchor_edge_included_vertex = set()

with open(data_file, 'r') as f:
    for line in f:
        head, relation, tail = line[:-1].split("\t")
        entities.add(head)
        entities.add(tail)
        entity_graph.append((head, tail))

        connected_entity[head].add(tail)
        connected_entity[tail].add(head)

entities_list = sorted(entities)
entity2id = {e: i for i, e in enumerate(entities_list)}

for (hd, tl) in entity_graph:
    edge_list.append((entity2id[hd], entity2id[tl]))

# select anchor via max-cut bipartition, read old_anchor.txt if it exists
try:
    with open(old_anchor_file, 'r') as f:
        for line in f:
            old_anchor = set(entities_list[int(i)] for i in line[:-1].split(" "))
except:
    old_anchor = set()

# while len(anchor) < k:
for i in range(k):
    best = None
    best_score = 0
    for vertex in entities.difference(anchor.union(old_anchor)):
        # getting degree(v)
        if len(connected_entity[vertex]) <= best_score:
            continue

        score = len(connected_entity[vertex].difference(anchor))
        if score > best_score:
            best = entity2id[vertex]
            best_score = score

    if best == None:
        print("no vertex added to anchor")
    else:
        print(best)
        anchor.add(best)

with open(old_anchor_file, "w") as fwrite:
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

options = nxmetis.MetisOptions(
    # objtype=1 => vol
    ptype=-1, objtype=1, ctype=-1, iptype=-1, rtype=-1, ncuts=-1,
    nseps=-1, numbering=-1, niter=-1, seed=-1, minconn=-1, no2hop=-1,
    contig=-1, compress=-1, ccorder=-1, pfactor=-1, ufactor=-1, dbglvl=-1)

(edgecuts, parts) = nxmetis.partition(G, nparts=partition_num)

# len(non_anchor) = 14941
# non_anchor_edge_list contained #vertex = 14897
# edgecuts = 14242, len(parts) = 14897

# putting residue randomly into non anchor set
residue = non_anchor_id.difference(non_anchor_edge_included_vertex)
for v in residue:
    parts[randint(0, partition_num - 1)].append(v)

# writing output file
with open(output_file, "w") as fwrite:
    fwrite.write(" ".join([str(i) for i in anchor])+"\n")
    for nas in parts:
        fwrite.write(" ".join([str(i) for i in nas])+"\n")

print("Created anchor & non anchor sets by max-min cut algorithm successfully!")
print(time()-t_)
