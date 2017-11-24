# coding: utf-8

from multiprocessing import Process
from multiprocessing import Manager


num_worker = 10
relation_triple_nums = [] # 각 relation이 있는 트리플 갯수


def relation_partition1(relation_id, relation_triple_nums, worker_relation_num, return_list):
    # 바깥쪽 호출하는 함수에서 하고 thread 돌고 나서
    # current best를 가장 작은 워커에 대해서 초기화 (?)
    current_best = 1e30
    manager = Manager()
    memo = manager.dict()

    worker_relation_num = [0 for i in range(num_worker)] # [num]
    relation_num = len(worker_relation_num)

    procs = []
    solution_list = []
    for i in num_worker:
        worker_relation_num_ = worker_relation_num.copy()
        worker_relation_num_[i] += relation_triple_nums[relation_id]
        
        # sorting should be changed
        worker_relation_num_ = sorted(worker_relation_num_, reverse=True)
        solution = manager.list([None for i in range(relation_num)])
        solution_list.append(solution)
        # recursive, parallel
        procs.append(Process(target=relation_partition2, args=(relation_id+1, relation_triple_nums, worker_relation_num_, return_list, solution, i)))

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()

    min_val = 1e30
    min_worker = None
    for i, val in enumerate(return_list):
        if min_val > val:
            min_val = val
            min_worker = i

    solution = list(solution_list[min_worker])
    solution[relation_id] = min_worker

    memo[(relation_id, worker_relation_num)] = min_val
    return solution


def relation_partition2(relation_id, relation_triple_nums, worker_relation_num, return_list, solution, procId):
    b1 = worker_relation_num[0]

    if relation_id == len(relation_triple_nums):
        if current_best > b1:
            current_best = b1

        return_list[procId] =  b1
        return b1

    else:
        if (relation_id, worker_relation_num) in memo:
            return_list[procId] = memo[(relation_id, worker_relation_num)]
            return memo[(relation_id, worker_relation_num)]
        elif b1 > current_best:
            return_list[procId] = 1e30
            return 1e30
        else:
            min_val = 1e30
            min_worker = None
            
            for i in num_worker:
                worker_relation_num_ = worker_relation_num.copy()
                worker_relation_num_[i] += relation_triple_nums[relation_id]
                
                # sorting should be changed
                worker_relation_num_ = sorted(worker_relation_num_, reverse=True)

                # recursive
                val = relation_partition(relation_id+1, relation_triple_nums, worker_relation_num)
                
                if min_val > val:
                    min_val = val
                    min_worker = i

            solution[relation_id] = min_worker
            memo[(relation_id, worker_relation_num)] = min_val
            return_list[procId] = min_val
            return min_val
