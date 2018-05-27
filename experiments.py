# coding: utf-8

from subprocess import Popen



#FB15K, WN18

#워커 갯수 2,4,6,8

# 워커-마스터 에폭 5-100, 10-50, 20-25
# 오리지널 임베딩도 총 500 에폭으로

# lr : 0.001

# dim: 50, 100

datasets = ['FB15K', 'WN18']
num_workers = [2, 4, 6, 8]
worker_master_epochs = [(5, 100), (10, 50), (20, 25)]
lr = 0.001
ndim = [50, 100]

for dataset in datasets:
    for num_worker in num_workers:
        for 