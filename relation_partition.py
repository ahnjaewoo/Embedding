# coding: utf-8

from multiprocessing import Process
from multiprocessing import Manager


num_worker = 10
relation_triple_nums = [] # 각 relation이 있는 트리플 갯수
worker_relation_num = [(0, i) for i in range(num_worker)] # [(num, id)]

# 바깥쪽 호출하는 함수에서 하고 thread 돌고 나서
# current best를 가장 작은 워커에 대해서 초기화 (?)
current_best = 1e30
manager = multiprocessing.Manager()
return_list = manager.list([None for i in range(num_worker)])
memo = manager.dict()


def relation_partition1(relation_id, relation_triple_nums, worker_relation_num, return_list):
    b1 = worker_relation_num[0][1]
    relation_clusters = [[] for i in range(num_worker)]

    if relation_id == len(relation_triple_nums):
        if current_best > b1:
            current_best = b1
        return b1, relation_clusters

    else:
        if (relation_id, worker_relation_num) in memo:
            return memo[(relation_id, worker_relation_num)]
        elif b1 > current_best:
            return 1e30, relation_clusters
        else:
            procs = []
            for i in num_worker:
                worker_relation_num_ = worker_relation_num.copy()
                worker_relation_num_[i] += relation_triple_nums[relation_id]
                
                # sorting should be changed
                worker_relation_num_ = sorted(worker_relation_num_, key=lambda x: x[1], reverse=True)

                # recursive, parallel
                procs.append(Process(target=relation_partition2, args=(relation_id+1, relation_triple_nums, worker_relation_num_, return_list, i)))

            for proc in procs:
                proc.start()

            for proc in procs:
                proc.join()

            min_val = 1e30
            min_worker = None
            for i, (val, clusters) in enumerate(return_list):
                if min_val > val:
                    min_val = val
                    min_worker = i
                    relation_clusters = clusters

            worker_relation_num[worker_relation_num[min_worker][1]][0] += relation_triple_nums[relation_id]
            memo[(relation_id, worker_relation_num)] = min_val
            relation_clusters[worker_relation_num[min_worker][1]].append(relation_id)

            return min_val, relation_clusters


def relation_partition2(relation_id, relation_triple_nums, worker_relation_num, return_list, procId):
    b1 = worker_relation_num[0][1]
    relation_clusters = [[] for i in range(num_worker)]

    if relation_id == len(relation_triple_nums):
        if current_best > b1:
            current_best = b1

        return_list[procId] =  (b1, relation_clusters)
        return b1, relation_clusters

    else:
        if (relation_id, worker_relation_num) in memo:
            return_list[procId] = (memo[(relation_id, worker_relation_num)], relation_clusters)
            return memo[(relation_id, worker_relation_num)], relation_clusters
        elif b1 > current_best:
            return_list[procId] = (1e30, relation_clusters)
            return 1e30, relation_clusters
        else:
            min_val = 1e30
            min_worker = None
            
            for i in num_worker:
                worker_relation_num_ = worker_relation_num.copy()
                worker_relation_num_[i] += relation_triple_nums[relation_id]
                
                # sorting should be changed
                worker_relation_num_ = sorted(worker_relation_num_, key=lambda x: x[1], reverse=True)

                # recursive
                val, clusters = relation_partition(relation_id+1, relation_triple_nums, worker_relation_num)
                
                if min_val > val:
                    min_val = val
                    min_worker = i
                    relation_clusters = clusters

            worker_relation_num[worker_relation_num[min_worker][1]][0] += relation_triple_nums[relation_id]
            memo[(relation_id, worker_relation_num)] = min_val
            relation_clusters[worker_relation_num[min_worker][1]].append(relation_id)

            return_list[procId] = (min_val, relation_clusters)
            return min_val, relation_clusters
