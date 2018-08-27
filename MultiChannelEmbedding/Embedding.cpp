#include "Import.hpp"
#include "DetailedConfig.hpp"
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

void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& train_iter, int& data_root_id, int& socket_port, string log_dir, int& precision, int& train_model, int& n_cluster, double& crp);

int main(int argc, char* argv[]){
	
	srand(time(nullptr));
	omp_set_num_threads(1);

	Model* model = nullptr;

	int dim = 20;
	double alpha = 0.01;
	double training_threshold = 1;
	int worker_num = 0;
	int master_epoch = 0;
	int train_iter = 10;
	int data_root_id = 0;
	int socket_port = 0;
	string log_dir;
	int precision;
	int train_model = 0;
	int n_cluster = 10;
	double crp = 0.05;

	unsigned int len;
	int nSockOpt;
	int embedding_sock, worker_sock;
	struct sockaddr_in embedding_addr;
	struct sockaddr_in worker_addr;
	double run_time;

	getParams(argc, argv, dim, alpha, training_threshold, worker_num, master_epoch, train_iter, data_root_id, socket_port, log_dir, precision, train_model, n_cluster, crp);

	bzero((char *)&embedding_addr, sizeof(embedding_addr));
	embedding_addr.sin_family = AF_INET;
	embedding_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
	embedding_addr.sin_port = htons(socket_port);

	// open log txt file
	FILE * fs_log;
	if(master_epoch == 0){

  		fs_log = fopen(argv[9], "w");
	}
	else{

  		fs_log = fopen(argv[9], "w+");
	}

	// embedding.cpp is server
	// worker.py is client
	// IP addr / port are from master.py
	if ((embedding_sock = socket(PF_INET, SOCK_STREAM, 0)) < 0){

		cout << "[error] embedding > create socket - worker_" << worker_num << endl;
		cout << "[error] embedding > return -1\n";
		fprintf(fs_log, "[error] embedding > create socket - worker_%d\n", worker_num);
		fprintf(fs_log, "[error] embedding > return -1\n");
		return -1;
	}

	// to solve bind error
	nSockOpt = 1;
	if (setsockopt(embedding_sock, SOL_SOCKET, SO_REUSEADDR, &nSockOpt, sizeof(nSockOpt)) < 0) {
		cout << "[error] embedding > bind socket - worker_" << worker_num << endl;
		return -1;
	}

	if (bind(embedding_sock, (struct sockaddr *)&embedding_addr, sizeof(embedding_addr)) < 0){

		cout << "[error] embedding > bind socket - worker_" << worker_num << ", " << socket_port << endl;
		cout << "[error] embedding > return -1\n";
		fprintf(fs_log, "[error] embedding > bind socket - worker_%d\n", worker_num);
		fprintf(fs_log, "[error] embedding > return -1\n");
		return -1;
	}
	
	if (listen(embedding_sock, 1) < 0){

		cout << "[error] embedding > listen socket - worker_" << worker_num << endl;
		cout << "[error] embedding > return -1\n";
		fprintf(fs_log, "[error] embedding > listen socket - worker_%d\n", worker_num);
		fprintf(fs_log, "[error] embedding > return -1\n");
		return -1;
	}

	len = sizeof(worker_addr);
	if ((worker_sock = accept(embedding_sock, (struct sockaddr *)&worker_addr, &len)) < 0){

		cout << "[error] embedding > accept socket - worker_" << worker_num << endl;
		cout << "[error] embedding > return -1\n";
		fprintf(fs_log, "[error] embedding > accept socket - worker_%d\n", worker_num);
		fprintf(fs_log, "[error] embedding > return -1\n");
		return -1;
	}

	// choosing data root by data root id
	if (data_root_id == 0){

		if (train_model == 0) {
			model = new TransE(FB15K, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, worker_sock, fs_log, precision);
		} else if (train_model == 1) {
			model = new TransG(FB15K, LinkPredictionTail, report_path, dim, alpha, training_threshold, n_cluster, crp, 10, false, true, true, worker_num, master_epoch, worker_sock, fs_log, precision);
		} else {
			cout << "[error] embedding > training model mismatch, recieved : " << train_model << endl;
	        cout << "[error] embedding > return -1\n";
        	fprintf(fs_log, "[error] embedding > training model mismatch, recieved : %d\n", train_model);
            fprintf(fs_log, "[error] embedding > return -1\n");
	        return -1;
		}
	}
	else if (data_root_id == 1){

		if (train_model == 0) { 
            model = new TransE(WN18, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, worker_sock, fs_log, precision);
        } else if (train_model == 1) {
			model = new TransG(WN18, LinkPredictionTail, report_path, dim, alpha, training_threshold, n_cluster, crp, 10, false, true, true, worker_num, master_epoch, worker_sock, fs_log, precision);
        } else { 
			cout << "[error] embedding > training model mismatch, recieved : " << train_model << endl;
            cout << "[error] embedding > return -1\n";
            fprintf(fs_log, "[error] embedding > training model mismatch, recieved : %d\n", train_model);
            fprintf(fs_log, "[error] embedding > return -1\n");
            return -1;
        }
	}
	//else if (data_root_id == 2){
	//
	//	model = new TransE(Dbpedia, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, worker_sock, fs_log);
	//}
	else{

		cout << "[error] embedding > wrong data_root_id, recieved : " << data_root_id << endl;
		cout << "[error] embedding > return -1\n";
		fprintf(fs_log, "[error] embedding > wrong data_root_id, recieved : %d\n", data_root_id);
		fprintf(fs_log, "[error] embedding > return -1\n");
		return -1;
	}

	// calculating training time
	struct timeval after, before;
	gettimeofday(&before, NULL);

	model->run(train_iter);

	gettimeofday(&after, NULL);
	run_time = after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0;
	//cout << "embedding > model->run end, training time : " << run_time << "seconds" << endl;
	fprintf(fs_log, "embedding > testing time : %lf seconds\n", run_time);
	
	model->save(to_string(worker_num), fs_log);
	//cout << "embedding > model->save end" << endl;
	//fprintf(fs_log, "embedding > model->save end\n");
	send(worker_sock, &run_time, sizeof(run_time), 0);

	delete model;
	fclose(fs_log);
	close(worker_sock);

	return 0;
}

void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& train_iter, int& data_root_id, int& socket_port, string log_dir, int& precision, int& train_model, int& n_cluster, double& crp){
	if (argc == 14) {
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
	    precision = atoi(argv[10]);
		train_model = atoi(argv[11]);
		n_cluster = atoi(argv[12]);
		crp = atof(argv[13]);
	} else {
		printf("[error] embedding > parameter number mismatch, recieved: %d\n", argc);
        printf("[error] embedding > return \n");
		return;
	}
}
