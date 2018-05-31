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

void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& train_iter, int& data_root_id, int& socket_port, string log_dir);

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
	string log_dir;

	unsigned int len;
	int nSockOpt;
	int embedding_sock, worker_sock;
	struct sockaddr_in embedding_addr;
	struct sockaddr_in worker_addr;

	getParams(argc, argv, dim, alpha, training_threshold, worker_num, master_epoch, train_iter, data_root_id, socket_port, log_dir);

	bzero((char *)&embedding_addr, sizeof(embedding_addr));
	embedding_addr.sin_family = AF_INET;
	embedding_addr.sin_addr.s_addr = inet_addr("0.0.0.0");
	embedding_addr.sin_port = htons(socket_port);


	// open log txt file
	FILE * fs_log;
	
	if(master_epoch == 0){

  		fs_log = fopen(log_dir, "w");
	}
	else{

  		fs_log = fopen(log_dir, "w+");
	}

	// embedding.cpp is server
	// worker.py is client
	// IP addr / port are from master.py
	if ((embedding_sock = socket(PF_INET, SOCK_STREAM, 0)) < 0){

		printf("[error] embedding.cpp > create socket - worker_%d\n", worker_num);
		printf("[error] embedding.cpp > return -1\n");
		fprintf(fs_log, "[error] embedding.cpp > create socket - worker_%d\n", worker_num);
		fprintf(fs_log, "[error] embedding.cpp > return -1\n");
		return -1;
	}

	// to solve bind error
	nSockOpt = 1;
	setsockopt(embedding_sock, SOL_SOCKET, SO_REUSEADDR, &nSockOpt, sizeof(nSockOpt));

	if (bind(embedding_sock, (struct sockaddr *)&embedding_addr, sizeof(embedding_addr)) < 0){

		printf("[error] embedding.cpp > bind socket - worker_%d\n", worker_num);
		printf("[error] embedding.cpp > return -1\n");
		fprintf(fs_log, "[error] embedding.cpp > bind socket - worker_%d\n", worker_num);
		fprintf(fs_log, "[error] embedding.cpp > return -1\n");
		return -1;
	}
	
	if (listen(embedding_sock, 1) < 0){

		printf("[error] embedding.cpp > listen socket - worker_%d\n", worker_num);
		printf("[error] embedding.cpp > return -1\n");
		fprintf(fs_log, "[error] embedding.cpp > listen socket - worker_%d\n", worker_num);
		fprintf(fs_log, "[error] embedding.cpp > return -1\n");
		return -1;
	}

	len = sizeof(worker_addr);
	if ((worker_sock = accept(embedding_sock, (struct sockaddr *)&worker_addr, &len)) < 0){

		printf("[error] embedding.cpp > accept socket - worker_%d\n", worker_num);
		printf("[error] embedding.cpp > return -1\n");
		fprintf(fs_log, "[error] embedding.cpp > accept socket - worker_%d\n", worker_num);
		fprintf(fs_log, "[error] embedding.cpp > return -1\n");
		return -1;
	}
	else{

		// printf("[info] embedding.cpp > accept socket successfully - worker_%d\n", worker_num);
		// fprintf(fs_log, "[info] embedding.cpp > accept socket successfully - worker_%d\n", worker_num);
	}

	// choosing data root by data root id
	if (data_root_id == 0){

		model = new TransE(FB15K, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, worker_sock, fs_log);
	}
	else if (data_root_id == 1){

		model = new TransE(WN18, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, worker_sock, fs_log);
	}
	//else if (data_root_id == 2){
	//
	//	model = new TransE(Dbpedia, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, worker_sock, fs_log);
	//}
	else{

		printf("[error] embedding.cpp > wrong data_root_id, recieved : %d\n", data_root_id);
		printf("[error] embedding.cpp > return -1\n");
		fprintf(fs_log, "[error] embedding.cpp > wrong data_root_id, recieved : %d\n", data_root_id);
		fprintf(fs_log, "[error] embedding.cpp > return -1\n");
		return -1;
	}

	// calculating training time
	struct timeval after, before;
	gettimeofday(&before, NULL);

	model->run(train_iter);

	gettimeofday(&after, NULL);
	cout << "[info] embedding.cpp > model->run end, training time : " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds" << endl;
	fprintf(fs_log, "[info] embedding.cpp > testing time : %lf seconds\n", after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0);
	
	model->save(to_string(worker_num), fs_log);
	// cout << "[info] embedding.cpp > model->save end" << endl;
	// fprintf(fs_log, "[info] embedding.cpp > model->save end\n");

	delete model;
	fclose(fs_log);
	close(worker_sock);

	return 0;
}

void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& train_iter, int& data_root_id, int& socket_port, string log_dir){

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
	if (argc == 10){

        string worker = argv[1];
        worker_num = worker.back() - '0';
        master_epoch = atoi(argv[2]);
        dim = atoi(argv[3]);
        alpha = atof(argv[4]);
        training_threshold = atof(argv[5]);
        train_iter = atoi(argv[6]);
		data_root_id = atoi(argv[7]);
		socket_port = atoi(argv[8]);
		log_dir = argv[9];
    }	
}