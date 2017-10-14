#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
//#include "Task.hpp"
#include <omp.h>

int main(int argc, char* argv[])
{
	DataModel* data_model =  new DataModel(FB15K, false);
	cout << "Data Model preprocessing completed!" << endl;
    return 0;
}
