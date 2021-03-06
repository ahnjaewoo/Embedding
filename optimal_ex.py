# coding: utf-8
from argparse import ArgumentParser
from collections import defaultdict
import sys


# argument parse
parser = ArgumentParser(description='Distributed Knowledge Graph Embedding')
parser.add_argument('--root_dir', type=str,
                    default="/home/rudvlf0413/distributedKGE/", help='project directory')
args = parser.parse_args()

sys.path.insert(0, args.root_dir)


data_roots = ['/fb15k', '/wn18', '/1mill', '/7.6mill', '/11mill', '/19mill', '/38mill']
workers = range(2, 41)


for data_root in data_roots:
    data_files = ('%s/train.txt' % data_root, '%s/dev.txt' % data_root, '%s/test.txt' % data_root)

    entities = list()
    entities_append = entities.append
    relations = list()
    relations_append = relations.append
    entity2id = dict()
    relation2id = dict()
    entity_cnt = 0
    relations_cnt = 0

    for file in data_files:

        with open(args.root_dir + file, 'r') as f:

            for line in f:

                head, relation, tail = line[:-1].split()

                if head not in entity2id:

                    entities_append(head)
                    entity2id[head] = entity_cnt
                    entity_cnt += 1

                if tail not in entity2id:

                    entities_append(tail)
                    entity2id[tail] = entity_cnt
                    entity_cnt += 1

                if relation not in relation2id:

                    relations_append(relation)
                    relation2id[relation] = relations_cnt
                    relations_cnt += 1

    relation_triples = defaultdict(list)

    with open(args.root_dir + data_files[0], 'r') as f:

        for line in f:

            head, relation, tail = line[:-1].split()
            head, relation, tail = entity2id[head], relation2id[relation], entity2id[tail]
            relation_triples[relation].append((head, tail))

        min_optimal_worker = 100
        for num_worker in workers:

            relation_each_num = [(k, len(v)) for k, v in relation_triples.items()]
            relation_each_num = sorted(relation_each_num, key=lambda x: x[1], reverse=True)
            allocated_relation_worker = [[[], 0] for i in range(num_worker)]

            is_optimal = True
            for i, (relation, num) in enumerate(relation_each_num):

                allocated_relation_worker = sorted(allocated_relation_worker, key=lambda x: x[1])

                diff = allocated_relation_worker[-1][1] - allocated_relation_worker[0][1]
                if i > 0 and num > diff:
                    is_optimal = False

                allocated_relation_worker[0][0].append(relation)
                allocated_relation_worker[0][1] += num

            if is_optimal and min_optimal_worker > num_worker:
                min_optimal_worker = num_worker
                break

        allocated_relation_worker = sorted(allocated_relation_worker, key=lambda x: x[1])
        max_num = allocated_relation_worker[-1][1]
        min_num = allocated_relation_worker[0][1]
        mean = sum([a[1] for a in allocated_relation_worker])/len(allocated_relation_worker)
        print(f"{data_root}, {min_optimal_worker}, {max_num}, {min_num}, {max_num-min_num}, {mean:.4f}")

