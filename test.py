# coding: utf-8
from multiprocessing import Process
from multiprocessing import Manager
from random import randint
import sys
sys.setrecursionlimit(10000)


def relation_partition1(relation_triple_nums, num_worker):
    manager = Manager()
    memo = manager.dict()
    current_best = manager.Value('f', 1e30)
    return_list = manager.list([None for _ in range(num_worker)])

    worker_relation_num = [0 for i in range(num_worker)] # [num]
    relation_num = len(relation_triple_nums)
    relation_id = 0

    procs = []
    solution_list = []
    for i in range(num_worker):
        worker_relation_num_ = worker_relation_num.copy()
        worker_relation_num_[i] += relation_triple_nums[relation_id]
        # worker_relation_num_ = sorted(worker_relation_num_, reverse=True)
        
        solution = manager.list([None for _ in range(relation_num)])
        solution_list.append(solution)
        # recursive, parallel
        procs.append(Process(target=relation_partition2, args=(relation_id+1, relation_triple_nums, memo, current_best, worker_relation_num_, num_worker, return_list, solution, i)))

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()

    min_val = 1e30
    min_worker = None
    for i, val in enumerate(return_list):
        if min_val >= val:
            min_val = val
            min_worker = i

    solution = list(solution_list[min_worker])
    solution[relation_id] = min_worker
    return solution


def relation_partition2(relation_id, relation_triple_nums, memo, current_best, worker_relation_num, num_worker, return_list, solution, procId):
    b1 = max(worker_relation_num)
    worker_relation_num_tuple = tuple(worker_relation_num)

    if relation_id == len(relation_triple_nums):
        if current_best.value > b1:
            current_best.value = b1

        return_list[procId] =  b1
        return b1

    else:
        worker_relation_num_tuple_s = tuple(sorted(worker_relation_num_tuple))
        if (relation_id, worker_relation_num_tuple_s) in memo:
            return_list[procId] = memo[(relation_id, worker_relation_num_tuple_s)]
            return memo[(relation_id, worker_relation_num_tuple_s)]
        
        if b1 >= current_best.value:
            return_list[procId] = 1e30
            return 1e30

        else:
            min_val = 1e30
            min_worker = None
            
            for i in range(num_worker):
                worker_relation_num_ = worker_relation_num.copy()
                worker_relation_num_[i] += relation_triple_nums[relation_id]
                # worker_relation_num_ = worker_relation_num_
                
                # recursive
                val = relation_partition2(relation_id+1, relation_triple_nums, memo, current_best, worker_relation_num_, num_worker, return_list, solution, procId)
                if min_val >= val:
                    min_val = val
                    min_worker = i

            solution[relation_id] = min_worker
            memo[(relation_id, worker_relation_num_tuple_s)] = min_val
            return_list[procId] = min_val
            return min_val


num = 3
relation_triple_nums = [1,2,4]
num_worker = 2
solution = relation_partition1(relation_triple_nums, num_worker)

worker_triple_num = {i: 0 for i in range(num_worker)}
for i, num in enumerate(relation_triple_nums):
    worker_triple_num[solution[i]] += num

for num in worker_triple_num:
    print(num, worker_triple_num[num])
