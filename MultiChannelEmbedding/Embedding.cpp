#include "Import.hpp"
#include "DetailedConfig.hpp"
// #include "LatentModel.hpp"
// #include "OrbitModel.hpp"
#include "Task.hpp"
#include <omp.h>

// 400s for each experiment.
int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;

	//first read the txt file and load the model

	//model = new TransE(FB15K, LinkPredictionTail, report_path, 20, 0.01, 2, false);
	model = new TransE(FB15K, LinkPredictionTail, report_path, 20, 0.01, 2, true);

	model->run(1);

	//after training, put entities and relations into txt file
	model->save("");

	model->test();

	delete model;

	return 0;
}
