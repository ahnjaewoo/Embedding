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


void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& train_iter, int& data_root_id);

// 400s for each experiment.
int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;
	int use_socket = 1;

	//first read the txt file and load the model
	//read dimension, LR, margin for parameters
	int dim = 20;
	double alpha = 0.01;
	double training_threshold = 2;
	int worker_num = 0;
	int master_epoch = 0;
	int train_iter = 10;
	int data_root_id = 0;

	if (use_socket)
	{
		// embedding.cpp is server
		// worker.py is client
		// IP addr / port are from master.py
		int flag_iter;
		int end_iter;
		unsigned int len;
		int embedding_sock[5], worker_sock;
		struct sockaddr_in embedding_addr;
		struct sockaddr_in worker_addr;

		getParams(argc, argv, dim, alpha, training_threshold, worker_num, master_epoch, train_iter, data_root_id);

		bzero((char *)&embedding_addr, sizeof(embedding_addr));
		embedding_addr.sin_family = AF_INET;
		embedding_addr.sin_addr.s_addr = inet_addr("0.0.0.0");
		embedding_addr.sin_port = htons(49900 + worker_num * 5 + master_epoch % 5);

		// to solve bind error
 		//struct linger solinger = { 1, 0 };
		//if(setsockopt(embedding_sock, SOL_SOCKET, SO_LINGER, &solinger, sizeof(struct linger)) == -1){
			
		//	perror("[error] setsockopt(SO_LINGER)\n");
		//	printf("[error] setsocketopt in embedding.cpp\n");
		//}

		while (1){

			if (master_epoch < 5){

				// create socket and check it is valid
				if ((embedding_sock[master_epoch % 5] = socket(PF_INET, SOCK_STREAM, 0)) < 0){

					printf("[error] create socket in embedding.cpp\n");
					return -1;
				}

				if (bind(embedding_sock[master_epoch % 5], (struct sockaddr *)&embedding_addr, sizeof(embedding_addr)) < 0){

					printf("[error] bind socket in embedding.cpp\n");
					return -1;
				}

				if (listen(embedding_sock[master_epoch % 5], 1) < 0){

					printf("[error] listen socket in embedding.cpp\n");
					return -1;
				}
			}

			len = sizeof(worker_addr);

			if ((worker_sock = accept(embedding_sock[master_epoch % 5], (struct sockaddr *)&worker_addr, &len)) < 0){

				printf("[error] accept socket in embedding.cpp\n");
				return -1;
			}

			if (recv(worker_sock, &flag_iter, sizeof(flag_iter), 0) < 0){

				printf("[error] recv flag_iter in embedding.cpp\n");
				close(worker_sock);
				break;
			}

			if (ntohl(flag_iter) == 1){

				printf("[Info] recv flag_iter 1 in embedding.cpp, quit");
				close(worker_sock);
				break;
			}

			// receive data
			if(recv(worker_sock, &worker_num, sizeof(worker_num), 0) < 0){

				printf("[error] recv worker_num in embedding.cpp\n");
				close(worker_sock);
				break;
			}

			if(recv(worker_sock, &master_epoch, sizeof(master_epoch), 0) < 0){

				printf("[error] recv master_epoch in embedding.cpp\n");
				close(worker_sock);
				break;
			}

			if(recv(worker_sock, &dim, sizeof(dim), 0) < 0){

				printf("[error] recv dim in embedding.cpp\n");
				close(worker_sock);
				break;
			}

			if(recv(worker_sock, &alpha, sizeof(alpha), 0) < 0){

				printf("[error] recv alpha in embedding.cpp\n");
				close(worker_sock);
				break;
			}

			if(recv(worker_sock, &training_threshold, sizeof(training_threshold), 0) < 0){

				printf("[error] recv training_threshold in embedding.cpp\n");
				close(worker_sock);
				break;
			}

			if(recv(worker_sock, &data_root_id, sizeof(data_root_id), 0) < 0){

				printf("[error] recv data_root_id in embedding.cpp\n");
				close(worker_sock);
				break;
			}

			worker_num = ntohl(worker_num);
			master_epoch = ntohl(master_epoch) + 1;
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
				printf("[error] recv data root id in embedding.cpp\n");
			}

			// calculating training time
			struct timeval after, before;
			gettimeofday(&before, NULL);

			model->run(train_iter);

			gettimeofday(&after, NULL);
			cout << "model->run end, training training_data time :  " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds" << endl;

			// after training, put entities and relations into txt file
			model->save(to_string(worker_num));
			cout << "model->save end" << endl;

			delete model;
			close(worker_sock);

			embedding_addr.sin_port = htons(49900 + worker_num * 5 + (master_epoch + 1) % 5);
		}
	}
	else 
	{
		// Model* model = nullptr;
		getParams(argc, argv, dim, alpha, training_threshold, worker_num, master_epoch, train_iter, data_root_id);

		if (data_root_id == 0) 
		{
			model = new TransE(FB15K, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, 0);
                }
                else if (data_root_id == 1)
                {
                        model = new TransE(WN18, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, 0);
                }
                /*
                else if (data_root_id == 2)
                {
                        model = new TransE(Dbpedia, LinkPredictionTail, report_path, dim, alpha, training_threshold, true, worker_num, master_epoch, 0);
                }
                */
                else
                {
                        printf("[error] recv data root id in embedding.cpp\n");
                }

		//calculating training time
		struct timeval after, before;
		gettimeofday(&before, NULL);

		model->run(train_iter);

		gettimeofday(&after, NULL);
		cout << "training training_data time :  " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds" << endl;

		//after training, put entities and relations into txt file
		model->save(to_string(worker_num));

		delete model;
	}

	return 0;
}

void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold, int& worker_num, int& master_epoch, int& train_iter, int& data_root_id)
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
		train_iter = atoi(argv[6]);
	}
	if (argc == 8)
        {
                string worker = argv[1];
                worker_num = worker.back() - '0';
                master_epoch = atoi(argv[2]);
                dim = atoi(argv[3]);
                alpha = atof(argv[4]);
                training_threshold = atof(argv[5]);
                train_iter = atoi(argv[6]);
		data_root_id = atoi(argv[7]);
        }	
} 
