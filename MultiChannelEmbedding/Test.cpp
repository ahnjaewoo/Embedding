#include "Import.hpp"
#include "DetailedConfig.hpp"
// #include "LatentModel.hpp"
// #include "OrbitModel.hpp"
#include "Task.hpp"
#include <omp.h>
#include <sys/time.h>


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>


void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& data_root_id);

// 400s for each experiment.
int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;

	//first read the txt file and load the model
	//read dimension, LR, margin for parameters
	int dim = 20;
	double alpha = 0.01;
	double training_threshold = 2;
	int worker_num = 0;
	int master_epoch = 0;
	int data_root_id = 0;

	// test.cpp is server
	// worker.py is client
	// IP addr / port are from master.py
	int flag_iter;
	int end_iter;
	unsigned int len;
	int test_sock, worker_sock;
	struct sockaddr_in test_addr;
	struct sockaddr_in worker_addr;

	getParams(argc, argv, dim, alpha, training_threshold, worker_num, master_epoch, data_root_id);

	bzero((char *)&test_addr, sizeof(test_addr));
	test_addr.sin_family = AF_INET;
	test_addr.sin_addr.s_addr = inet_addr("0.0.0.0");
	test_addr.sin_port = htons(7874);

	// to solve bind error
		//struct linger solinger = { 1, 0 };
	//if(setsockopt(test_sock, SOL_SOCKET, SO_LINGER, &solinger, sizeof(struct linger)) == -1){
		
	//	perror("[error] setsockopt(SO_LINGER)\n");
	//	printf("[error] setsocketopt in test.cpp\n");
	//}

	// create socket and check it is valid
	if ((test_sock = socket(PF_INET, SOCK_STREAM, 0)) < 0){

		printf("[error] test.cpp > create socket\n");
		return -1;
	}

	if (bind(test_sock, (struct sockaddr *)&test_addr, sizeof(test_addr)) < 0){

		printf("[error] test.cpp > bind socket\n");
		return -1;
	}

	if (listen(test_sock, 1) < 0){

		printf("[error] test.cpp > listen socket\n");
		return -1;
	}

	len = sizeof(worker_addr);

	if ((worker_sock = accept(test_sock, (struct sockaddr *)&worker_addr, &len)) < 0){

		printf("[error] test.cpp > accept socket\n");
		return -1;
	}

	if (recv(worker_sock, &flag_iter, sizeof(flag_iter), 0) < 0){

		printf("[error] test.cpp > recv flag_iter\n");
		close(worker_sock);
		return -1;
	}

	if (ntohl(flag_iter) == 1){

		printf("[error] test.cpp > recv quit signal (flag_iter = 1), quit");
		close(worker_sock);
		return -1;
	}

	// receive data
	if(recv(worker_sock, &worker_num, sizeof(worker_num), 0) < 0){

		printf("[error] test.cpp > recv worker_num\n");
		close(worker_sock);
		return -1;
	}

	if(recv(worker_sock, &master_epoch, sizeof(master_epoch), 0) < 0){

		printf("[error] test.cpp > recv master_epoch\n");
		close(worker_sock);
		return -1;
	}

	if(recv(worker_sock, &dim, sizeof(dim), 0) < 0){

		printf("[error] test.cpp > recv dim\n");
		close(worker_sock);
		return -1;
	}

	if(recv(worker_sock, &alpha, sizeof(alpha), 0) < 0){

		printf("[error] test.cpp > recv alpha\n");
		close(worker_sock);
		return -1;
	}

	if(recv(worker_sock, &training_threshold, sizeof(training_threshold), 0) < 0){

		printf("[error] test.cpp > recv training_threshold\n");
		close(worker_sock);
		return -1;
	}

	if(recv(worker_sock, &data_root_id, sizeof(data_root_id), 0) < 0){

		printf("[error] test.cpp > recv data_root_id\n");
		close(worker_sock);
		return -1;
	}

	worker_num = ntohl(worker_num);
	master_epoch = ntohl(master_epoch);
	dim = ntohl(dim);
	data_root_id = ntohl(data_root_id);

	// choosing data root by data root id
	if (data_root_id == 0)
	{
		model = new TransE(FB15K, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, worker_sock);
	}
	else if (data_root_id == 1)
	{
		model = new TransE(WN18, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, worker_sock);
	}
	/*
	else if (data_root_id == 2)
	{
		model = new TransE(Dbpedia, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, worker_sock);
	}
	*/
	else
	{
		printf("[error] test.cpp > recv data root id\n");
	}

	//calculating testing time
	struct timeval after, before;
	gettimeofday(&before, NULL);

    printf("[info] test.cpp > test start\n");

	model->test();

	gettimeofday(&after, NULL);
	cout << "[info] test.cpp > testing time :  " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds" << endl;
	delete model;
	close(worker_sock);	


	return 0;
}

void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& data_root_id)
{
	if (argc == 2)
	{
		// very big problem for scaling!!!!!!!!!!!!!!!!!!!!!!!!!!!
		string worker = argv[1];
		worker_num = worker.back() - '0';
	}
	if (argc == 3)
	{
		string worker = argv[1];
                worker_num = worker.back() - '0';
		master_epoch = atoi(argv[2]);
	}
	if (argc == 4)
	{
		string worker = argv[1];
                worker_num = worker.back() - '0';
		master_epoch = atoi(argv[2]);
		dim = atoi(argv[3]);
	}
	if (argc == 5)
	{
		string worker = argv[1];
                worker_num = worker.back() - '0';
		master_epoch = atoi(argv[2]);
		dim = atoi(argv[3]);
		alpha = atof(argv[4]);
	}
	if (argc == 6)
	{
		string worker = argv[1];
                worker_num = worker.back() - '0';
		master_epoch = atoi(argv[2]);
		dim = atoi(argv[3]);
		alpha = atof(argv[4]);
		training_threshold = atof(argv[5]);
	}
	if (argc == 7)
        {
                string worker = argv[1];
                worker_num = worker.back() - '0';
                master_epoch = atoi(argv[2]);
                dim = atoi(argv[3]);
                alpha = atof(argv[4]);
                training_threshold = atof(argv[5]);
		data_root_id = atoi(argv[6]);
        }
}  
