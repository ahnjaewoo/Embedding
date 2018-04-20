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

void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& train_iter, int& data_root_id, int& socket_port);

// 400s for each experiment.
int main(int argc, char* argv[]){
	
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;

	int dim = 20;
	double alpha = 0.01;
	double training_threshold = 2;
	int worker_num = 0;
	int master_epoch = 0;
	int train_iter = 10;
	int data_root_id = 0;
	int socket_port = 0;

	int flag_iter;
	unsigned int len;
	int embedding_sock, worker_sock;
	struct sockaddr_in embedding_addr;
	struct sockaddr_in worker_addr;

	getParams(argc, argv, dim, alpha, training_threshold, worker_num, master_epoch, train_iter, data_root_id, socket_port);

	bzero((char *)&embedding_addr, sizeof(embedding_addr));
	embedding_addr.sin_family = AF_INET;
	embedding_addr.sin_addr.s_addr = inet_addr("0.0.0.0");
	embedding_addr.sin_port = htons(socket_port);

	// to solve bind error
	//struct linger solinger = { 1, 0 };
	//if(setsockopt(embedding_sock, SOL_SOCKET, SO_LINGER, &solinger, sizeof(struct linger)) == -1){
		
	//	perror("[error] setsockopt(SO_LINGER)\n");
	//	printf("[error] setsocketopt in embedding.cpp\n");
	//}

	// embedding.cpp is server
	// worker.py is client
	// IP addr / port are from master.py
	if ((embedding_sock = socket(PF_INET, SOCK_STREAM, 0)) < 0){

		printf("[error] embedding.cpp > create socket - worker_%d\n", worker_num);
		return -1;
	}
	if (bind(embedding_sock, (struct sockaddr *)&embedding_addr, sizeof(embedding_addr)) < 0){

		printf("[error] embedding.cpp > bind socket - worker_%d\n", worker_num);
		return -1;
	}
	if (listen(embedding_sock, 1) < 0){

		printf("[error] embedding.cpp > listen socket - worker_%d\n", worker_num);
		return -1;
	}

	len = sizeof(worker_addr);
	if ((worker_sock = accept(embedding_sock, (struct sockaddr *)&worker_addr, &len)) < 0){

		printf("[error] embedding.cpp > accept socket - worker_%d\n", worker_num);
		return -1;
	}
	else{

		printf("[info] embedding.cpp > accept socket successfully - worker_%d\n", worker_num);
	}

	// choosing data root by data root id
	if (data_root_id == 0){

		model = new TransE(FB15K, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, worker_sock);
	}
	else if (data_root_id == 1){

		model = new TransE(WN18, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, worker_sock);
	}
	//else if (data_root_id == 2){
	//
	//	model = new TransE(Dbpedia, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, worker_sock);
	//}
	else{

		printf("[error] embedding.cpp > wrong data_root_id, recieved : %d\n", data_root_id);
		return -1;
	}

	// calculating training time
	struct timeval after, before;
	gettimeofday(&before, NULL);

	model->run(train_iter);

	gettimeofday(&after, NULL);
	cout << "[info] embedding.cpp > model->run end, training time : " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds" << endl;

	model->save(to_string(worker_num));
	cout << "[info] embedding.cpp > model->save end" << endl;

	delete model;
	sleep(2000);
	close(worker_sock);

	return 0;
}

void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& train_iter, int& data_root_id, int& socket_port){

	if (argc == 2){
		// very big problem for scaling!!!!!!!!!!!!!!!!!!!!!!!!!!!
		string worker = argv[1];
		worker_num = worker.back() - '0';
	}
	if (argc == 3){

		string worker = argv[1];
        worker_num = worker.back() - '0';
		master_epoch = atoi(argv[2]);
	}
	if (argc == 4){

		string worker = argv[1];
        worker_num = worker.back() - '0';
		master_epoch = atoi(argv[2]);
		dim = atoi(argv[3]);
	}
	if (argc == 5){

		string worker = argv[1];
        worker_num = worker.back() - '0';
		master_epoch = atoi(argv[2]);
		dim = atoi(argv[3]);
		alpha = atof(argv[4]);
	}
	if (argc == 6){

		string worker = argv[1];
        worker_num = worker.back() - '0';
		master_epoch = atoi(argv[2]);
		dim = atoi(argv[3]);
		alpha = atof(argv[4]);
		training_threshold = atof(argv[5]);
	}
	if (argc == 7){

		string worker = argv[1];
		worker_num = worker.back() - '0';
		master_epoch = atoi(argv[2]);
		dim = atoi(argv[3]);
		alpha = atof(argv[4]);
		training_threshold = atof(argv[5]);
		train_iter = atoi(argv[6]);
	}
	if (argc == 8){

        string worker = argv[1];
        worker_num = worker.back() - '0';
        master_epoch = atoi(argv[2]);
        dim = atoi(argv[3]);
        alpha = atof(argv[4]);
        training_threshold = atof(argv[5]);
        train_iter = atoi(argv[6]);
		data_root_id = atoi(argv[7]);
    }
	if (argc == 9){

        string worker = argv[1];
        worker_num = worker.back() - '0';
        master_epoch = atoi(argv[2]);
        dim = atoi(argv[3]);
        alpha = atof(argv[4]);
        training_threshold = atof(argv[5]);
        train_iter = atoi(argv[6]);
		data_root_id = atoi(argv[7]);
		socket_port = atoi(argv[8]);
    }	
}