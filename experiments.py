# coding: utf-8

from subprocess import Popen

#FB15K, WN18
# 워커 갯수 2,4,6,8
# 워커-마스터 에폭 5-100, 10-50, 20-25
# 오리지널 임베딩도 총 500 에폭으로
# lr : 0.001
# dim: 50, 100

datasets = ['/fb15k', '/wn18']
num_workers = [2, 4, 6, 8]
worker_master_epochs = [(5, 100), (10, 50), (20, 25)]
lr = 0.001
ndims = [50, 100]

key_list = ['dataset', 'num_worker', 'train_iter', 'worker_iter', 'ndim',
            'Raw.BestMEANS', 'Raw.BestMRR',
            'Raw.BestHITS', 'Filter.BestMEANS', 'Filter.BestMRR', 'Filter.BestHITS',
            'Accuracy', 'Best', 'train_time']

with open("result.csv", 'w') as result_file:
    result_file.write(", ".join(key_list))
    result_file.write("\n")

    for dataset in datasets:
        for num_worker in num_workers:
            for train_iter, niter in worker_master_epochs:
                for ndim in ndims:
                    process = Popen(['python', 'master.py', '--data_root',
                                     dataset, '--num_worker', str(num_worker),
                                     '--train_iter', str(
                                         train_iter), '--niter', str(niter),
                                     '--ndim', str(ndim), '--lr', str(lr)])
                    process.communicate()

                    print(f"dataset: {dataset}")
                    print(f"num_worker: {num_worker}")
                    print(f"train_iter: {train_iter}")
                    print(f"niter: {niter}")
                    print(f"ndim: {ndim}")
                    result_file.write(f"{dataset}, ")
                    result_file.write(f"{num_worker}, ")
                    result_file.write(f"{train_iter}, {niter}, ")
                    result_file.write(f"{ndim}, ")
                    result_file.write(f"{lr}, ")

                    with open("logs/test_log.txt", 'r') as f:
                        for line in f:
                            line = line[:-1]
                            if line[:3] == '== ':
                                key, value = line[3:].split(" = ")

                                if key == 'train_time':
                                    result_file.write(f"{value}\n")
                                else:
                                    result_file.write(f"{value}, ")
