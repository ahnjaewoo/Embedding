# coding: utf-8
from multiprocessing import Process
from multiprocessing import Manager
from time import time
import sys
import bisect
from random import randint
sys.setrecursionlimit(10000)


def relation_partition(relation_triple_nums, num_worker, current_best=1e30, init_greedy=True):
    def relation_partition_inner(relation_id, relation_triple_nums, memo, num_worker, worker_relation_num, relation_ids, opt):
        if relation_id == len(relation_triple_nums):
            b1 = max(worker_relation_num)
            if opt['current_best'] > b1:
                opt['current_best'] = b1
            return b1, relation_ids

        else:
            worker_relation_num_tuple_sorted = tuple(sorted(worker_relation_num))
            if (relation_id, tuple(worker_relation_num)) in memo:
                return memo[(relation_id, worker_relation_num_tuple_sorted)], relation_ids
            
            b1 = max(worker_relation_num)
            if b1 > opt['current_best']:
                return 1e30, relation_ids

            else:
                min_val = 1e30
                min_worker = None
                min_relation_ids = None
                processed = set()
                for i in range(num_worker):
                    if worker_relation_num[i] in processed:
                        continue
                    processed.add(worker_relation_num[i])
                    worker_relation_num_ = [k for k in worker_relation_num]
                    worker_relation_num_[i] += relation_triple_nums[relation_id]
                    
                    relation_ids_ = [x[:] for x in relation_ids]
                    relation_ids_[i].append(relation_id)
                    val, relation_ids_ = relation_partition_inner(relation_id+1, relation_triple_nums, memo, num_worker, worker_relation_num_, relation_ids_, opt)
                    if min_val > val:
                        min_val = val
                        min_worker = i
                        min_relation_ids = relation_ids_

                memo[(relation_id, worker_relation_num_tuple_sorted)] = min_val
                return min_val, min_relation_ids

    memo = {}
    worker_relation_num = [0 for _ in range(num_worker)] # [num]
    worker_relation_num[0] += relation_triple_nums[0]
    relation_ids = [[] for _ in range(num_worker)]
    relation_ids[0].append(0)

    if init_greedy:
        greedy_solution = [0 for i in range(num_worker)]
        for num in relation_triple_nums:
            greedy_solution = sorted(greedy_solution)
            greedy_solution[0] += num

        current_best = max(greedy_solution)

    opt = {}
    opt['current_best'] = current_best
    print current_best, greedy_solution
    val, relation_ids = relation_partition_inner(1, relation_triple_nums, memo, num_worker, worker_relation_num, relation_ids, opt)
    return val, relation_ids


relation_triple_nums = sorted([randint(10,1000) for i in range(21)], reverse=True)
t = time()
val, relation_ids = relation_partition(relation_triple_nums, 10, init_greedy=True)
print(time()-t)
print(val, relation_ids)
