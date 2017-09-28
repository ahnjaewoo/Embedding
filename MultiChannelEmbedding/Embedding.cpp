#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "LatentModel.hpp"
#include "OrbitModel.hpp"
#include "Task.hpp"
#include <omp.h>

// 400s for each experiment.
int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;

	model = new TransE(FB15K, LinkPredictionTail, report_path, 300, 0.01, 2);

	//first read the txt file
	model->load("output.txt");

	model->run(1);

	//after training, put entities and relations into txt file
	model->save("output.txt");

	model->test();

	delete model;



	return 0;
}