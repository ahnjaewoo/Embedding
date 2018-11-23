# coding: utf-8
from subprocess import Popen, PIPE

#FB15K, WN18
# 워커 갯수 2,4,6,8
# 워커-마스터 에폭 5-100, 10-50, 20-25
# 오리지널 임베딩도 총 500 에폭으로
# lr : 0.001
# dim: 50, 100

#datasets = ('fb15k', 'wn18')
datasets = ('fb15k',)
#num_workers = (2, 4, 6, 8)
num_workers = (4,)
#master_worker_epochs = ((100, 5), (50, 10), (25, 20))
master_worker_epochs = ((20, 25),)
lr = 0.01
#ndims = (50, 100)
ndims = (100,)
#precisions = (0, 1)
precisions = (0,)
precision_names = ('single', )
#precision_names = ('half',)

key_list = ['dataset', 'num_worker', 'master_epoch', 'worker_iter', 'ndim', 'precision', 'lr',
            'Raw.BestMEANS', 'Raw.BestMRR', 'Raw.BestHITS', 'Filter.BestMEANS',
            'Filter.BestMRR', 'Filter.BestHITS', 'Accuracy', 'preprocessing_time',
            'train_time', 'avg_work_time', 'avg_worker_time', 'avg_maxmin_time',
            'avg_datamodel_sock_time', 'avg_socket_load_time', 'avg_embedding_time',
            'avg_model_run_time', 'avg_socket_save_time', 'avg_redis_time']

with open("result.csv", 'w') as result_file:
    result_file.write(", ".join(key_list))
    result_file.write("\n")

    for dataset in datasets:
        for num_worker in num_workers:
            for master_epoch, worker_iter in master_worker_epochs:
                for ndim in ndims:
                    for precision in precisions:
                        process = Popen(['python', 'master.py', '--data_root',
                                        f'/{dataset}', '--num_worker', str(num_worker),
                                        '--train_iter', str(worker_iter),
                                        '--niter', str(master_epoch), '--ndim', str(ndim),
                                        '--lr', str(lr), '--debugging', 'no',
                                        '--precision', str(precision),
                                        '--redis_ip', 'localhost', '--redis_port', '6379',
                                        '--pypy_dir', '/home/rudvlf0413/pypy2-v6.0.0-linux64/bin/pypy',
                                        '--scheduler_ip', 'localhost:8786'])
                        process.communicate()

                        print(f"dataset: {dataset}")
                        print(f"num_worker: {num_worker}")
                        print(f"train_iter: {master_epoch}")
                        print(f"niter: {worker_iter}")
                        print(f"ndim: {ndim}")
                        print(f"precision: {precision_names[precision]}")
                        print(f"lr: {lr}")
                        result_file.write(f"{dataset}, {num_worker}, {master_epoch}, {worker_iter}, {ndim}, {precision_names[precision]}, {lr}, ")
                        
                        with open("logs/test_log.txt", 'r') as f:
                            for line in f:
                                line = line[:-1]
                                if line[:3] == '== ':
                                    key, value = line[3:].split(" = ")

                                    if key == key_list[-1]:
                                        result_file.write(f"{value}\n")
                                    else:
                                        result_file.write(f"{value}, ")

key_list = ['dataset', 'train_iter', 'ndim', 'lr', 'Raw.BestMEANS', 'Raw.BestMRR',
            'Raw.BestHITS', 'Filter.BestMEANS', 'Filter.BestMRR', 'Filter.BestHITS',
            'Accuracy', 'train_time']

train_iter = 250

print("baseline test")
with open("baseline_result.csv", 'w') as result_file:
    result_file.write(", ".join(key_list))
    result_file.write("\n")

    for dataset_id, dataset in enumerate(datasets):
        for ndim in ndims:
            process = Popen(['./Embedding.out', str(dataset_id), str(ndim), str(lr)],
                            stdout=PIPE, stderr=PIPE, cwd='./baseline/')
            out, _ = process.communicate()
            lines = out.decode('utf-8').split("\n")

            print(f"dataset: {dataset}")
            print(f"train_iter: {train_iter}")
            print(f"ndim: {ndim}")
            print(f"lr: {lr}")
            result_file.write(f"{dataset}, {train_iter}, {ndim}, {lr}, ")

            for line in lines:
                line = line[:-1]
                if line[:3] == '== ':
                    key, value = line[3:].split(" = ")

                    if key == key_list[-1]:
                        result_file.write(f"{value}\n")
                    else:
                        result_file.write(f"{value}, ")
