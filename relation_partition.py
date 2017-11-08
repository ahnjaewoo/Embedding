# coding: utf-8


num_worker = 10
relation_triple_nums = [] # 각 relation이 있는 트리플 갯수
relation_triples = dict() # 각 relation이 있는 트리플
worker_relation_num = []
partitioned_relations = [[] for _ in range(num_worker)]

current_best = 1e30
memo = dict() # concurrent hashmap으로 바꿔야!

def relation_partition(relation_id, relation_triple_nums, relation_triples, worker_relation_num, partitioned_relations):
    b1 = worker_relation_num[0][1]
    if relation_id == len(relation_triple_nums):
        if current_best > b1:
            current_best = b1
        return b1

    else:
        if (relation_id, worker_relation_num) in memo:
            return memo[(relation_id, worker_relation_num)]
        elif b1 > current_best:
            return 1e30
        else:
            min_val = 1e30
            min_worker = None
            # should be parallelized
            for i in num_worker:
                worker_relation_num_ = worker_relation_num.copy()
                worker_relation_num_[i] += relation_triple_nums[relation_id]
                # sorting should be changed
                worker_relation_num_ = sorted(worker_relation_num_, key=lambda x: x[1], reverse=True)

                # recursive
                val = relation_partition(relation_id+1, relation_triple_nums, relation_triples, worker_relation_num, partitioned_relations)
                
                if min_val > val:
                    min_val = val
                    min_worker = i

            worker_relation_num[min_worker] += relation_triple_nums[relation_id]
            memo[(relation_id, worker_relation_num)] = min_val

            return min_val

# Question) 언제 triple들을 워커에 할당하지?!
