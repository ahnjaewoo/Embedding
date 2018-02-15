#include "Import.hpp"
#include "DetailedConfig.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
//#include "Task.hpp"
#include <omp.h>

int main(int argc, char* argv[])
{
	DataModel* data_model = nullptr;

	if (argc >= 2)
	{
		int data_root_id = atoi(argv[1]);
		if (data_root_id == 0)
		{
			DataModel* data_model =  new DataModel(FB15K, false);
		}
		else if (data_root_id == 1)
		{
			DataModel* data_model =  new DataModel(WN18, false);
		}
		/*
		else if (data_root_id == 2)
		{
			DataModel* data_model =  new DataModel(Dbpedia, false);
		}
		*/
		else
		{
			printf("[error] recv data root id in preprocess.cpp\n");
		}
	}
	else
	{
		DataModel* data_model =  new DataModel(FB15K, false);
	}

	cout << "Data Model preprocessing completed!" << endl;
    return 0;
}
