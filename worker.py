# coding: utf-8

from subprocess import Popen
import redis
import shlex


def work():
    r = redis.StrictRedis(host='163.152.20.66', port=6379, db=0, password='davian!')

    # redis에서 embedding vector들 받아오기

    vectors = r.mget([f'_{i}' for i in range(a.shape[0])])
    pickle.loads(vectors[i])

    # 파일로 저장, c에서 불러와야!

    # c 프로그램 호출
    command = "bash test.sh"
    args = shlex.split(command)
    proc = Popen(args)
    proc.wait()


    # embedding 학습 끝나면 c에서 업데이트한 것들 불러 오기

    # redis에 해당 학습된 벡터들 업데이트하기
    r.mset({f'_{i}': pickle.dumps(v) for i, v in enumerate(a)})
