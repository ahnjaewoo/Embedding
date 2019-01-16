# coding: utf-8
from subprocess import Popen, PIPE

#FB15K, WN18
# 워커 갯수 2,4,6,8
# 워커-마스터 에폭 5-100, 10-50, 20-25
# 오리지널 임베딩도 총 500 에폭으로
# lr : 0.001
# dim: 50, 100

#datasets = ('fb15k',)
datasets = ('dbpedia',)
#num_workers = (2, 4, 6, 8)
num_workers = (4,8,)
#master_worker_epochs = ((100, 5), (50, 10), (25, 20))
#master_worker_epochs = ((25,40),(20,50),(10, 100),(5,200),(5,100))
master_worker_epochs = ((20,50),(10,100),)
lr = 0.0015
#ndims = (50, 100)
ndims = (100,)
#precisions = (0, 1)
precisions = (0,)
precision_names = ('single',)
#precision_names = ('half',)
#train_models = ('transE','transG',)
train_models = ('transG',)
anchor_nums = (10000,)
margin = 3
crp = 0.05
#graph_split_methods = ('degmin', 'randmin',)
graph_split_methods = ('maxmin',)

key_list = ['train_model', 'dataset', 'num_worker', 'master_epoch', 'worker_iter', 'ndim', 'precision', 'lr', 'anchor_num', 'graph_split', 
            'Raw.BestMEANS', 'Raw.BestMRR', 'Raw.BestHITS', 'Filter.BestMEANS',
            'Filter.BestMRR', 'Filter.BestHITS', 'Accuracy', 'preprocessing_time',
            'train_time', 'avg_work_time', 'avg_worker_time', 'avg_maxmin_time',
            'avg_datamodel_sock_time', 'avg_socket_load_time', 'avg_embedding_time',
            'avg_model_run_time', 'avg_socket_save_time', 'avg_redis_time']

with open("result_anchor.csv", 'w') as result_file:
    result_file.write(", ".join(key_list))
    result_file.write("\n")

    for train_model in train_models:
        for dataset in datasets:
            for num_worker in num_workers:
                for master_epoch, worker_iter in master_worker_epochs:
                    for ndim in ndims:
                        for precision in precisions:
                            for anchor_num in anchor_nums:
                                for graph_split_method in graph_split_methods:
                                    process = Popen(['python', 'master.py', '--data_root',
                                                    f'/{dataset}', '--num_worker', str(num_worker),
                                                    '--train_iter', str(worker_iter),
                                                    '--niter', str(master_epoch), '--ndim', str(ndim),
                                                    '--lr', str(lr), '--debugging', 'no',
                                                    '--margin', str(margin),
                                                    '--crp', str(crp),
                                                    '--precision', str(precision),
                                                    '--redis_ip', 'localhost', '--redis_port', '6379',
                                                    '--pypy_dir', '/home/rudvlf0413/pypy2-v6.0.0-linux64/bin/pypy',
                                                    '--scheduler_ip', 'localhost:8786',
                                                    '--anchor_num', str(anchor_num),
                                                    '--graph_split', str(graph_split_method),
                                                    '--train_model', str(train_model)])
                                    process.communicate()

                                    print(f"train_model: {train_model}")
                                    print(f"dataset: {dataset}")
                                    print(f"num_worker: {num_worker}")
                                    print(f"train_iter: {worker_iter}")
                                    print(f"niter: {master_epoch}")
                                    print(f"ndim: {ndim}")
                                    print(f"precision: {precision_names[precision]}")
                                    print(f"lr: {lr}")
                                    print(f"anchor_num: {anchor_num}")
                                    print(f"graph_split: {graph_split_method}")
                                    result_file.write(f"{train_model}, {dataset}, {num_worker}, {master_epoch}, {worker_iter}, {ndim}, {precision_names[precision]}, {lr}, {anchor_num}, {graph_split_method}, ")
                                
                                    with open("logs/test_log.txt", 'r') as f:
                                        for line in f:
                                            line = line[:-1]
                                            if line[:3] == '== ':
                                                key, value = line[3:].split(" = ")

                                                if key == key_list[-1]:
                                                    result_file.write(f"{value}\n")
                                                else:
                                                    result_file.write(f"{value}, ")

key_list = ['train_model', 'dataset', 'train_iter', 'ndim', 'lr', 'Raw.BestMEANS', 'Raw.BestMRR',
            'Raw.BestHITS', 'Filter.BestMEANS', 'Filter.BestMRR', 'Filter.BestHITS',
            'Accuracy', 'train_time']

train_iter = 250
"""
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
"""
 
 
