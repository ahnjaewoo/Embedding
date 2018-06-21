#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "umHalf.h"


int main(int argc, char *argv[])
{
        unsigned int len;
        int nSockOpt;
	int embedding_sock, worker_sock;
        struct sockaddr_in embedding_addr;
        struct sockaddr_in worker_addr;
        double send_value = 0.1115;
        int socket_port = 11555;
        half temp;

        embedding_addr.sin_family = AF_INET;
        embedding_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        embedding_addr.sin_port = htons(socket_port);

        if ((embedding_sock = socket(PF_INET, SOCK_STREAM, 0)) < 0)
        {
                printf("[error] embedding > create socket\n");
                return -1;
        }

        if (bind(embedding_sock, (struct sockaddr *)&embedding_addr, sizeof(embedding_addr)) < 0)
        {
                printf("[error] embedding > bind socket\n");
                return -1;
        }

        if (listen(embedding_sock, 1) < 0)
        {
                printf("[error] embedding > listen socket\n");
                return -1;
        }
        len = sizeof(worker_addr);
        if ((worker_sock = accept(embedding_sock, (struct sockaddr *)&worker_addr, &len)) < 0)
        {
                printf("[error] embedding > accept socket\n");
                return -1;
        }
        temp = 0.011555;
        printf("size of %d\n", sizeof(temp));

        half * temp_array = (half *)calloc(5, sizeof(half));
        for (int j = 0; j < 4; j++) {
                temp_array[j] = temp;
        }
        printf("size of %d\n", sizeof(temp_array));

        send(worker_sock, temp_array, 4*sizeof(half), 0);
}
