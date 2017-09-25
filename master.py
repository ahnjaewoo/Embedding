# coding: utf-8

from distributed import Client
import redis
import numpy as np
import pickle
import time
import random


# master node에 redis db 실행해야함
# master에서 dask-scheduler 실행
# 각 worker pc에서는 dask-worker <마스터 ip>:8786



def randomSleep(i):
    print(f"{i}th worker start")
    time.sleep(random.randint(1, 5))
    print(f"{i}th worker finished")
    return i


client = Client('163.152.20.66:8786', asynchronous=True)

results = []

# 작업 배정
for i in range(10):
    # worker.py 호출
    results.append(client.submit(randomSleep, i))

print("aa")
# max-min cut 실행


# worker들 작업 끝나면 anchor 등 재분배
for result in results:
    print(result.result())
