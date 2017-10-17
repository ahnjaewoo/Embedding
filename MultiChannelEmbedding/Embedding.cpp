#include "Import.hpp"
#include "DetailedConfig.hpp"
// #include "LatentModel.hpp"
// #include "OrbitModel.hpp"
#include "Task.hpp"
#include <omp.h>

void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold);

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
	getParams(argc, argv, dim, alpha, training_threshold);

	//model = new TransE(FB15K, LinkPredictionTail, report_path, dim, alpha, training_threshold, false);
	model = new TransE(FB15K, LinkPredictionTail, report_path, dim, alpha, training_threshold, true);

	model->run(1);

	//after training, put entities and relations into txt file
	model->save("");

	model->test();

	delete model;

	return 0;
}

void getParams(int argc, char* argv[], int& dim, double& alpha, double& training_threshold)
{
	if (argc == 2)
	{
		dim = atoi(argv[1]);
	}
	if (argc == 3)
	{
		dim = atoi(argv[1]);
		alpha = atof(argv[2]);	
	}
	if (argc == 4)
	{
		dim = atoi(argv[1]);
		alpha = atof(argv[2]);
		training_threshold = atof(argv[3]);
	}
}
