#include "Import.hpp"
#include "DetailedConfig.hpp"
// #include "LatentModel.hpp"
// #include "OrbitModel.hpp"
#include "Task.hpp"
#include <omp.h>
#include <sys/time.h>


void getParams(int argc, char* argv[], int& data_root_id, int& dim, double& lr);
int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;
	int dim = 50;
	double lr = 0.001;
	int data_root_id = 0;

	getParams(argc, argv, dim, lr);
	if(data_root_id == 0){
		model = new TransE(FB15K, LinkPredictionTail, report_path, dim, lr, 1);
	} else if(data_root_id == 1) {
		model = new TransE(WN18, LinkPredictionTail, report_path, dim, lr, 1);
	}

	struct timeval after, before;
	gettimeofday(&before, NULL);
	model->run(500);

	gettimeofday(&after, NULL);
	cout << "training training_data time :  " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds" << endl;
	
	model->save("./model.bin");
	model->test();

	return 0;
}

void getParams(int argc, char* argv[], int& data_root_id, int& dim, double& lr){
	if (argc == 2) {
		data_root_id = atoi(argv[1])
	} else if (argc == 3){
		dim = atof(argv[2]);
	} else if(argc == 4){
		lr = atoi(argv[3]);
	}
}