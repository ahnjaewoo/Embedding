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

void getParams(int argc, char* argv[], int& test_port, int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& data_root_id, string log_dir, int& precision, int& train_model, int& n_cluster, double& crp);

// 400s for each experiment.
int main(int argc, char* argv[]){
	
	srand(time(nullptr));
	omp_set_num_threads(1);

	Model* model = nullptr;

	//first read the txt file and load the model
	//read dimension, LR, margin for parameters
	int dim = 20;
	double alpha = 0.01;
	double training_threshold = 2;
	int worker_num = 0;
	int master_epoch = 0;
	int data_root_id = 0;
	string log_dir;
	int precision;
	int train_model = 0;
	int n_cluster = 10;
	double crp = 0.05;
	
	// test.cpp is server
	// worker.py is client
	// IP addr / port are from master.py
	unsigned int len;
	int nSockOpt;
	int test_sock, master_sock;
	struct sockaddr_in test_addr;
	struct sockaddr_in master_addr;

	int trial;
	int success;
	int test_port;

	getParams(argc, argv, test_port, dim, alpha, training_threshold, worker_num, master_epoch, data_root_id, log_dir, precision, train_model, n_cluster, crp);

	bzero((char *)&test_addr, sizeof(test_addr));
	test_addr.sin_family = AF_INET;
	test_addr.sin_addr.s_addr = inet_addr("0.0.0.0");
	test_addr.sin_port = htons(test_port);


	// open log txt file
	FILE * fs_log;
	fs_log = fopen(argv[7], "w");

	// create socket and check it is valid
	if ((test_sock = socket(PF_INET, SOCK_STREAM, 0)) < 0){

		printf("[error] test.cpp > create socket\n");
		printf("[error] test.cpp > return -1\n");
		fprintf(fs_log, "[error] test.cpp > create socket\n");
		fprintf(fs_log, "[error] test.cpp > return -1\n");
		return -1;
	}

	// to solve bind error
	nSockOpt = 1;
	setsockopt(test_sock, SOL_SOCKET, SO_REUSEADDR, &nSockOpt, sizeof(nSockOpt));
	
	success = 0;
	trial = 0;

	while(trial < 5 && !success){

		if (bind(test_sock, (struct sockaddr *)&test_addr, sizeof(test_addr)) < 0){

			printf("[error] test.cpp > bind socket, retry\n");
			fprintf(fs_log, "[error] test.cpp > bind socket, retry\n");
			trial = trial + 1;
			success = 0;
		}
		else{

			success = 1;
			trial = 0;
			break;
		}
	}

	if(trial >= 5){

		printf("[error] test.cpp > cannot bind socket, terminate");
		printf("[error] test.cpp > return -1\n");
		fprintf(fs_log, "[error] test.cpp > cannot bind socket, terminate");
		fprintf(fs_log, "[error] test.cpp > return -1\n");
		return -1;
	}

	if (listen(test_sock, 1) < 0){

		printf("[error] test.cpp > listen socket\n");
		printf("[error] test.cpp > return -1\n");
		fprintf(fs_log, "[error] test.cpp > listen socket\n");
		fprintf(fs_log, "[error] test.cpp > return -1\n");
		return -1;
	}

	len = sizeof(master_addr);

	if ((master_sock = accept(test_sock, (struct sockaddr *)&master_addr, &len)) < 0){

		printf("[error] test.cpp > accept socket\n");
		printf("[error] test.cpp > return -1\n");
		fprintf(fs_log, "[error] test.cpp > accept socket\n");
		fprintf(fs_log, "[error] test.cpp > return -1\n");
		return -1;
	}
	else{

		printf("[info] test.cpp > accept socket successfully\n");
		fprintf(fs_log, "[info] test.cpp > accept socket successfully\n");
	}


	 // choosing data root by data root id
        if (data_root_id == 0){

			if (train_model == 0) {
				model = new TransE(FB15K, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, master_sock, fs_log, precision);
			} else if (train_model == 1) {
				model = new TransG(FB15K, LinkPredictionTail, report_path, dim, alpha, training_threshold, n_cluster, crp, 10, false, true, true, worker_num, master_epoch, master_sock, fs_log, precision);
			} else {
				printf("[error] embedding > training model mismatch, recieved : %d\n", train_model);
				printf("[error] embedding > return -1\n");
				fprintf(fs_log, "[error] embedding > training model mismatch, recieved : %d\n", train_model);
				fprintf(fs_log, "[error] embedding > return -1\n");
				return -1;
			}
        }
        else if (data_root_id == 1){

                if (train_model == 0) { 
                    model = new TransE(WN18, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, master_sock, fs_log, precision);
                } else if (train_model == 1) {
                    model = new TransG(WN18, LinkPredictionTail, report_path, dim, alpha, training_threshold, n_cluster, crp, 10, false, true, true, worker_num, master_epoch, master_sock, fs_log, precision);
                } else { 
					printf("[error] embedding > training model mismatch, recieved : %d\n", train_model);
					printf("[error] embedding > return -1\n");
					fprintf(fs_log, "[error] embedding > training model mismatch, recieved : %d\n", train_model);
					fprintf(fs_log, "[error] embedding > return -1\n");
					return -1;
                }
        }
	//else if (data_root_id == 2){
	//
	//	model = new TransE(Dbpedia, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, master_sock, fs_log);
	//}
	else{

		printf("[error] test.cpp > wrong data_root_id, recieved : %d\n", data_root_id);
		fprintf(fs_log, "[error] test.cpp > wrong data_root_id, recieved : %d\n", data_root_id);
	}

	//calculating testing time
	struct timeval after, before;
	gettimeofday(&before, NULL);

    printf("[info] test.cpp > test start\n");
    fprintf(fs_log, "[info] test.cpp > test start\n");

	model->test();

	gettimeofday(&after, NULL);
	cout << "[info] test.cpp > testing time :  " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds" << endl;
	fprintf(fs_log, "[info] test.cpp > testing time : %lf seconds\n", after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0);
	
	delete model;
	fclose(fs_log);
	close(master_sock);	

	return 0;
}

void getParams(int argc, char* argv[], int& test_port, int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& data_root_id, string log_dir, int& precision, int& train_model, int& n_cluster, double& crp){

	if (argc == 13){
		string worker = argv[1];
		worker_num = worker.back() - '0';
		master_epoch = atoi(argv[2]);
		dim = atoi(argv[3]);
		alpha = atof(argv[4]);
		training_threshold = atof(argv[5]);
		data_root_id = atoi(argv[6]);
		log_dir = argv[7];
		precision = atoi(argv[8]);
		train_model = atoi(argv[9]);
		n_cluster = atoi(argv[10]);
		crp = atof(argv[11]);
		test_port = atoi(argv[12]);

	} else {
		printf("[error] embedding > parameter number mismatch, recieved: %d\n", argc);
		printf("[error] embedding > return \n");
		return;
	}
}
