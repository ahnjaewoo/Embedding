# coding: utf-8

import socket
import struct
import numpy as np
from time import sleep


def sockRecv(sock, length):

    data = b''

    while len(data) < length:

        buff = sock.recv(length - len(data))

        if not buff:

            return None

        data = data + buff

    return data


a = np.random.random((30, 40))

embedding_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
embedding_sock.connect(('127.0.0.1', 11555))

sleep(10)

test_val = sockRecv(embedding_sock, 8)
print(len(test_val))
test_val = struct.unpack('eeee', test_val)
print(test_val)

embedding_sock.send(struct.pack('e', 0.333))

embedding_sock.close()
