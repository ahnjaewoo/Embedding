#include "Import.hpp"
#include "DetailedConfig.hpp"
// #include "LatentModel.hpp"
// #include "OrbitModel.hpp"
#include "Task.hpp"
#include <omp.h>
#include <sys/time.h>

// 400s for each experiment.
int main(int argc, char* argv[])
{
	srand(time(nullptr));
	//omp_set_num_threads(6);

	Model* model = nullptr;

	model = new TransE(WN18, LinkPredictionTail, report_path, 50, 0.001, 1);

	struct timeval after, before;
	gettimeofday(&before, NULL);
	model->run(500);

	gettimeofday(&after, NULL);
	cout << "training training_data time :  " << after.tv_sec + after.tv_usec/1000000.0 - before.tv_sec - before.tv_usec/1000000.0 << "seconds" << endl;
	
	model->save("./model.bin");
	model->test();

	return 0;
}
