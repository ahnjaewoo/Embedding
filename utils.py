# coding: utf-8
from port_for import select_random
from subprocess import Popen
import sys
import os
import timeit
import pickle

def sockRecv(sock, length):
    
    data = b''

    while len(data) < length:

        buff = sock.recv(length - len(data))

        if not buff:

            return None

        data += buff

    return data


def data2id(data_root):

    data_root_split = [x.lower() for x in data_root.split('/')]

    if 'fb15k' in data_root_split:

        return 0

    elif 'wn18' in data_root_split:

        return 1

    elif 'dbpedia' in data_root_split:

        return 2

    else:

        print("[error] master > data root mismatch")
        sys.exit(1)


def model2id(train_model):

    if train_model == "transE":

        return 0
    
    elif train_model == "transG":

        return 1

    else:
    
        print("[error] master > train model mismatch")
        sys.exit(1)


def install_libs():

    import os
    from pkgutil import iter_modules

    modules = set([m[1] for m in iter_modules()])

    if 'redis' not in modules:
        os.system("pip install --upgrade redis")
    if 'hiredis' not in modules:
        os.system("pip install --upgrade hiredis")


def work(chunk_data, worker_id, cur_iter, n_dim, lr, margin, train_iter, data_root_id,
         redis_ip, root_dir, debugging, precision, train_model, n_cluster, crp,
         train_code_dir, preprocess_folder_dir, worker_code_dir, unix_socket_path):

    # socket_port = init_port + (cur_iter + 1) * int(worker_id.split('_')[1])
    socket_port = str(select_random())
    log_dir = os.path.join(root_dir, f'logs/embedding_log_{worker_id}_iter_{cur_iter}.txt')

    if cur_iter % 2 == 0:
        with open(f"{root_dir}/chunk_data_{worker_id}.txt", 'wb') as f:
            pickle.dump(chunk_data, f, pickle.HIGHEST_PROTOCOL)
    
    workStart = timeit.default_timer()

    embedding_proc = Popen([train_code_dir, worker_id, str(cur_iter), str(n_dim), str(lr),
                            str(margin), str(train_iter), str(data_root_id), socket_port,
                            log_dir, str(precision), str(train_model), str(n_cluster), str(crp)],
                            cwd=preprocess_folder_dir)


    worker_proc = Popen(["python", worker_code_dir, worker_id, str(cur_iter), str(n_dim),
                         redis_ip, root_dir, socket_port, debugging, str(precision), str(train_model),
                         str(n_cluster), str(crp), unix_socket_path])

    worker_proc.wait()
    worker_return = worker_proc.returncode

    if worker_return > 0:
        # worker.py가 비정상 종료 (embedding이 중간에 비정상이면 worker.py도 비정상)

        if embedding_proc.poll() is None:
            embedding_proc.kill()

        return (False, None)

    else:

        embedding_proc.wait()
        embedding_return = embedding_proc.returncode

        if embedding_return < 0:
            # worker.py는 정상 종료 되었지만 embedding이 비정상 정료
            return (False, None)

        else:

            # 모두 성공적으로 수행
            # worker_return 은 string 형태? byte 형태? 의 pickle 을 가지고 있음
            timeNow = timeit.default_timer()
            return (True, timeNow - workStart)


def iter_mset(redis_client, data_items: dict, chunk_size=1_000_000):
    data_items = list(data_items.items())
    chunk_num = len(data_items) // chunk_size
    
    for i in range(chunk_num):
        sub_data = {k: v for k, v in data_items[i * chunk_size: (i + 1) * chunk_size]}
        redis_client.mset(sub_data)
    
    # 나머지
    sub_data = {k: v for k, v in data_items[chunk_num * chunk_size:]}
    redis_client.mset(sub_data)


def iter_mget(redis_client, data_keys: list, chunk_size=1_000_000):
    results = []
    chunk_num = len(data_keys) // chunk_size

    for i in range(chunk_num):
        sub_keys = data_keys[i * chunk_size: (i + 1) * chunk_size]
        results.extend(redis_client.mget(sub_keys))

    sub_keys = data_keys[chunk_num * chunk_size:]
    results.extend(redis_client.mget(sub_keys))

    return results
