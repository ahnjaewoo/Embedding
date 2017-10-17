#coding: utf-8

# max-min 구현하는 부분
# 웬만하면 pypy에서 구현할 수 있도록!

import networkx as nx
import nxmetis
import numpy as np
import pickle
import os
from random import randint

root = 'fb15k/'
tmp = 'tmp/'
data_files = ['train.txt']
output_file = 'maxmin_output.txt'
partition_num = 20
k = 10

# initial graph settings
entities = set()
entities_list = list()
relations = set()
relations_list = list()
entity_graph = []
edge_list = []
connected_entity = dict()
entity2id = dict()
id2entity = dict()
relation2id = dict()
id2relation = dict()

# initial anchor settings
anchor = set()
old_anchor = set()
old_anchor_file = 'old_anchor.txt'
non_anchor = set()
non_anchor_edge_list = []

# initial non-anchor settings
non_anchor = set()
non_anchor_id = set()
non_anchor_edge_list = list()
non_anchor_edge_included_vertex = set()
parts_set = list()
G = None
split_num = int()

# read training files
print("read files")
for file in data_files:
    with open(root+file, 'r') as f:
        for line in f:
            head, relation, tail = line[:-1].split("\t")
            entities.add(head)
            entities.add(tail)
            relations.add(relation)
            entity_graph.append((head, tail))
            
            if head not in connected_entity:
            	connected_entity[head] = set()
            connected_entity[head].add(tail)
            if tail not in connected_entity:
            	connected_entity[tail] = set()
            connected_entity[tail].add(head)

entities_list = sorted(entities)
entity2id = {e: i for i, e in enumerate(entities_list)}
id2entity = {i: e for i, e in enumerate(entities_list)}
relations_list = sorted(relations)
relation2id = {r: i for i, r in enumerate(relations_list)}
id2relation = {i: r for i, r in enumerate(relations_list)}

for (hd,tl) in entity_graph:
	edge_list.append((entity2id[hd],entity2id[tl]))

# selecting anchor via max-cut bipartition
	# read old_anchor.txt if it exists
if os.path.exists(tmp+old_anchor_file):
	with open(tmp+old_anchor_file, 'r') as f:
		for line in f:
			old_anchor = set(line[:-1].split(" "))

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
			best = vertex
			best_score = score
	if best == None:
		print("no vertex added to anchor")
	else:
		anchor.add(best)

	# writing old_anchor_file
fwrite = open(tmp+old_anchor_file, "w")
for v in anchor:
	fwrite.write("%s " % (v))
fwrite.close()

# solving the min-cut partition problem of A~
	# finding A~ and edges 
non_anchor = entities.difference(anchor)
non_anchor_id = {entity2id[v] for v in non_anchor}
non_anchor_edge_list = [(hd,tl) for (hd,tl) in edge_list if hd in non_anchor_id and tl in non_anchor_id]
for (hd,tl) in non_anchor_edge_list:
	non_anchor_edge_included_vertex.add(hd)
	non_anchor_edge_included_vertex.add(tl)

# constructing nx.Graph and using metis in order to get min-cut partition
G = nx.Graph()
split_num = partition_num
if split_num <= 1:
	print("split number error!")
G.add_edges_from(non_anchor_edge_list)

options = nxmetis.MetisOptions()
options.ptype = -1
options.objtype = 1 # vol
options.ctype = -1
options.iptype = -1
options.rtype = -1
options.ncuts = -1
options.nseps = -1
options.numbering = -1
options.niter = -1
options.seed = -1
options.minconn = -1
options.no2hop = -1
options.contig = -1
options.compress = -1
options.ccorder = -1
options.pfactor = -1
options.ufactor = -1
options.dbglvl = -1

(edgecuts, parts) = nxmetis.partition(G, nparts=split_num)

	# len(non_anchor) = 14941
	# non_anchor_edge_list contained #vertex = 14897
	# edgecuts = 14242, len(parts) = 14897

	# 1. putting residue randomly into non anchor set
residue = non_anchor_id.difference(non_anchor_edge_included_vertex)
for v in residue:
	parts[randint(0,split_num - 1)].append(v)
	
	# 2. writing output file
fwrite = open(tmp+output_file, "w")
for v in anchor:
	fwrite.write("%s " % (v))
fwrite.write("\n")
for nas in parts:
	for v in nas:
		fwrite.write("%s " % (id2entity[v]))
	fwrite.write("\n")
fwrite.close()

print("Created anchor & non anchor sets by max-min cut algorithm successfully!")
