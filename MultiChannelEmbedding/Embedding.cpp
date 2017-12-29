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


void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& is_final);

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
	int is_final = 0;

	/*

	// embedding.cpp is server
	// worker.ph is client
	// IP addr / port are from master.py
	int len;
	struct sockaddr_in embedding_addr;
	struct sockaddr_in worker_addr;
	bzero((char *)&embedding_addr, sizeof(embedding_addr));
	embedding_addr.sin_family = AF_INET;
	embedding_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
	embedding_addr.sin_port = htons(47500);

	// create socket and check it is valid
	if ((embedding_sock = socket(PF_INET, SOCK_STREAM, 0)) < 0){

		return -1;
	}

	if (bind(embedding_sock, (struct sockaddr *)&embedding_addr, sizeof(embedding_addr)) < 0){

		return -1;
	}

	if (listen(embedding_sock, 1) < 0){

		return -1;
	}

	while(1){

		if ((worker_sock = accept(embedding_sock, (struct sockaddr *)&worker_addr, &len)) < 0){

			return -1;
		}

		// get args, to be modified
		msg_len = recv(worker_sock, buff, len, 0);


		model = new TransE(FB15K, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch);

			//calculating training time
		clock_t before, after;
		before = clock();

		model->run(1000);

		after = clock();
		cout << "training training_data time :  " << (double)(after - before) / CLOCKS_PER_SEC << "seconds" << endl;

		//after training, put entities and relations into txt file
		model->save(to_string(worker_num));

		delete model;

		// reconnect to worker.py
		close(worker_sock);
		

		// TODO : model->save using socket communication
	}

	*/

	// Model* model = nullptr;
	getParams(argc, argv, dim, alpha, training_threshold, worker_num, master_epoch, is_final);

	//model = new TransE(FB15K, LinkPredictionTail, report_path, dim, alpha, training_threshold, false);
	model = new TransE(FB15K, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch);


	//calculating training time
	struct timeval after, before;
	gettimeofday(&before, NULL);

	model->run(1000);

	gettimeofday(&after, NULL);
	cout << "training training_data time :  " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds" << endl;

	//after training, put entities and relations into txt file
	model->save(to_string(worker_num));

	if (is_final) model->test();

	delete model;



	//------------ socket while end



	return 0;
}

void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& is_final)
{
	if (argc == 2)
	{
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
		is_final = atoi(argv[6]);
	}
}
 
