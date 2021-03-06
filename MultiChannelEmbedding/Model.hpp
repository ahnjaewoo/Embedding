#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
#include <boost/progress.hpp>
#include <fstream>

using namespace std;
using namespace arma;

class Model
{
public:
	const DataModel&	data_model;
	const TaskType		task_type;
	const bool		be_deleted_data_model;
	const bool		is_preprocessed;
	const int		worker_num;
	const int		master_epoch;
	const int		fd;
	FILE * 	fs_log;	
	const int precision;

public:
	ModelLogging&		logging;

public:
	int	epos;

public:
	Model(const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0,
		FILE * fs_log = NULL,
		const int precision = 0)
		:data_model(*(new DataModel(dataset, is_preprocessed, worker_num, master_epoch, fd, fs_log))), task_type(task_type),
		logging(*(new ModelLogging(logging_base_path))),be_deleted_data_model(true),
		is_preprocessed(is_preprocessed), worker_num(worker_num), master_epoch(master_epoch), fd(fd), fs_log(fs_log), precision(precision)
	{
		epos = 0;
		best_triplet_result = 0;
		
		// std::cout << "[info] Model.hpp > Model constructor called" << endl;
		// std::cout << "\t[Dataset]\t" << dataset.name;
		// std::cout << TaskTypeName(task_type);

		// logging.record() << "[info] Model.hpp > Model constructor called";
		// logging.record() << "\t[Dataset]\t" << dataset.name;
		// logging.record() << TaskTypeName(task_type);
	}

	Model(const Dataset& dataset,
		const string& file_zero_shot,
		const TaskType& task_type,
		const string& logging_base_path,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0,
		FILE * fs_log = NULL,
		const int precision = 0)
		:data_model(*(new DataModel(dataset, is_preprocessed, worker_num, master_epoch, fd, fs_log))), task_type(task_type),
		logging(*(new ModelLogging(logging_base_path))), be_deleted_data_model(true),
		is_preprocessed(is_preprocessed), worker_num(worker_num), master_epoch(master_epoch), fd(fd), fs_log(fs_log), precision(precision)
	{
		epos = 0;
		best_triplet_result = 0;
		// std::cout << "[info] Model.hpp > Model constructor called" << endl;

		// logging.record() << "\t[Dataset]\t" << dataset.name;
		// logging.record() << TaskTypeName(task_type);
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet) = 0;
	virtual void train_triplet(const pair<pair<int, int>, int>& triplet) = 0;
	virtual void train_triplet_parts(const pair<pair<int, int>, int>& triplet) = 0;
	virtual void train_triplet_parts_relation(const pair<pair<int, int>, int>& triplet) = 0;

public:
	virtual void train(bool last_time = false)
	{
		++epos;

#pragma omp parallel for
		for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
		{
			train_triplet(*i);
		}
	}

	virtual void train_parts(bool last_time = false)
	{
		++epos;

#pragma omp parallel for
		for (auto i = data_model.data_train_parts.begin(); i != data_model.data_train_parts.end(); ++i)
		{
			train_triplet_parts(*i);
		}
	}

	virtual void train_parts_relation(bool last_time = false)
	{
		++epos;

#pragma omp parallel for
		for (auto i = data_model.data_train_parts.begin(); i != data_model.data_train_parts.end(); ++i)
		{
			train_triplet_parts_relation(*i);
		}
	}

	void run(int total_epos)
	{
		//epoch is an even : entity by anchor
		if (master_epoch % 2 == 0)
		{
			// logging.record() << "\t[Epos]\t" << total_epos;
			// cout << "[info] Model.hpp > train entity at master epoch " << master_epoch << endl;
			--total_epos;
			//boost::progress_display	cons_bar(total_epos);
			while (total_epos-- > 0)
			{
				//++cons_bar;
				train_parts();

				if (task_type == TripletClassification)
					test_triplet_classification();
			}

			train_parts(true);
		}
		//epoch is an odd : relation
		else
		{
			// logging.record() << "\t[Epos]\t" << total_epos;
			// cout << "[info] Model.hpp > train relation at master epoch " << master_epoch << endl;
			--total_epos;
			//boost::progress_display	cons_bar(total_epos);
			while (total_epos-- > 0)
			{
				//++cons_bar;
				//train();
				train_parts_relation();

				if (task_type == TripletClassification)
					test_triplet_classification();
			}
			//train(true);
			train_parts_relation(true);
		}
	}

public:
	double		best_triplet_result;
	double		best_link_mean;
	double		best_link_hitatten;
	double		best_link_fmean;
	double		best_link_fhitatten;

	void reset()
	{
		best_triplet_result = 0;
		best_link_mean = 1e10;
		best_link_hitatten = 0;
		best_link_fmean = 1e10;
		best_link_fhitatten = 0;
	}

	void test(int hit_rank = 10)
	{
		logging.record();

		best_link_mean = 1e10;
		best_link_hitatten = 0;
		best_link_fmean = 1e10;
		best_link_fhitatten = 0;

		if (task_type == LinkPredictionHead || task_type == LinkPredictionTail || task_type == LinkPredictionRelation)
			test_link_prediction(hit_rank);
		if (task_type == LinkPredictionHeadZeroShot || task_type == LinkPredictionTailZeroShot || task_type == LinkPredictionRelationZeroShot)
			test_link_prediction_zeroshot(hit_rank);
		else
			test_triplet_classification();
	}

public:
	void test_triplet_classification()
	{
		double real_hit = 0;
		for (auto r = 0; r < data_model.set_relation.size(); ++r)
		{

			vector<pair<double, bool>>	threshold_dev;
			for (auto i = data_model.data_dev_true.begin(); i != data_model.data_dev_true.end(); ++i)
			{
				if (i->second != r)
					continue;

				threshold_dev.push_back(make_pair(prob_triplets(*i), true));
			}
			for (auto i = data_model.data_dev_false.begin(); i != data_model.data_dev_false.end(); ++i)
			{
				if (i->second != r)
					continue;

				threshold_dev.push_back(make_pair(prob_triplets(*i), false));
			}

			sort(threshold_dev.begin(), threshold_dev.end());

			double threshold;
			double vari_mark = 0;
			int total = 0;
			int hit = 0;
			for (auto i = threshold_dev.begin(); i != threshold_dev.end(); ++i)
			{
				if (i->second == false)
					++hit;
				++total;

				if (vari_mark <= 2 * hit - total + data_model.data_dev_true.size())
				{
					vari_mark = 2 * hit - total + data_model.data_dev_true.size();
					threshold = i->first;
				}
			}

			double lreal_hit = 0;
			double lreal_total = 0;
			for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
			{
				if (i->second != r)
					continue;

				++lreal_total;

				if (prob_triplets(*i) > threshold)

					++real_hit, ++lreal_hit;
			}

			for (auto i = data_model.data_test_false.begin(); i != data_model.data_test_false.end(); ++i)
			{
				if (i->second != r)
					continue;

				++lreal_total;
				if (prob_triplets(*i) <= threshold)
					++real_hit, ++lreal_hit;
			}

			//logging.record()<<data_model.relation_id_to_name.at(r)<<"\t"
			//	<<lreal_hit/lreal_total;
		}

		// printf("[Info] Model.hpp > true = %d\n", data_model.data_test_true.size());
		// printf("[Info] Model.hpp > false = %d\n", data_model.data_test_false.size());
		// printf("[Info] Model.hpp > true + false = %d\n", data_model.data_test_true.size() + data_model.data_test_false.size());
		// printf("[Info] Model.hpp > real_hit = %lf\n", real_hit);
		
		// fprintf(fs_log, "[Info] Model.hpp > Triplet classification\n");
		// fprintf(fs_log, "[Info] Model.hpp > true = %d\n", data_model.data_test_true.size());
		// fprintf(fs_log, "[Info] Model.hpp > false = %d\n", data_model.data_test_false.size());
		// fprintf(fs_log, "[Info] Model.hpp > true + false = %d\n", data_model.data_test_true.size() + data_model.data_test_false.size());
		// fprintf(fs_log, "[Info] Model.hpp > real_hit = %lf\n", real_hit);

		std::cout << epos << "\t Accuracy = "
			<< real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size());
		best_triplet_result = max(
			best_triplet_result,
			real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size()));
		std::cout << ", Best = " << best_triplet_result << endl;
		fprintf(fs_log, "== Accuracy = %lf\n", real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size()));
		//fprintf(fs_log, "== Best = %lf\n", best_triplet_result);

		// logging.record() << epos << "\t Accuracy = "
		// 	<< real_hit / (data_model.data_test_true.size() + data_model.data_test_false.size())
		// 	<< ", Best = " << best_triplet_result;

		std::cout.flush();
	}

	void test_link_prediction(int hit_rank = 10, const int part = 0)
	{
		double mean = 0;
		double hits = 0;
		double fmean = 0;
		double fhits = 0;
		double rmrr = 0;
		double fmrr = 0;
		double total = data_model.data_test_true.size();

		double arr_mean[20] = { 0 };
		double arr_total[5] = { 0 };

		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			++arr_total[data_model.relation_type[i->second]];
		}

		int cnt = 0;

		// boost::progress_display cons_bar(data_model.data_test_true.size() / 100);

#pragma omp parallel for
		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			++cnt;
			// if (cnt % 100 == 0)
			// {
			// 	++cons_bar;
			// }

			pair<pair<int, int>, int> t = *i;
			int frmean = 0;
			int rmean = 0;
			double score_i = prob_triplets(*i);

			if (task_type == LinkPredictionRelation || part == 2)
			{
				for (auto j = 0; j != data_model.set_relation.size(); ++j)
				{

					t.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
						++frmean;
				}
			}
			else
			{
				for (auto j = 0; j != data_model.set_entity.size(); ++j)
				{

					if (task_type == LinkPredictionHead || part == 1)
						t.first.first = j;
					else
						t.first.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++frmean;
						//if (frmean > hit_rank)
						//	break;
					}
				}
			}

#pragma omp critical
			{
				if (frmean < hit_rank)
					++arr_mean[data_model.relation_type[i->second]];

				mean += rmean;
				fmean += frmean;
				rmrr += 1.0 / (rmean + 1);
				fmrr += 1.0 / (frmean + 1);

				if (rmean < hit_rank)
					++hits;
				if (frmean < hit_rank)
					++fhits;
			}
		}

		//std::cout << endl;
		// for (auto i = 1; i <= 4; ++i)
		// {
		// 	std::cout << i << ':' << arr_mean[i] / arr_total[i] << endl;
		// 	logging.record() << i << ':' << arr_mean[i] / arr_total[i];
		// }
		// logging.record();

		best_link_mean = min(best_link_mean, mean / total);
		best_link_hitatten = max(best_link_hitatten, hits / total);
		best_link_fmean = min(best_link_fmean, fmean / total);
		best_link_fhitatten = max(best_link_fhitatten, fhits / total);

		std::cout << "Raw.BestMEANS = " << best_link_mean << endl;
		std::cout << "Raw.BestMRR = " << rmrr / total << endl;
		std::cout << "Raw.BestHITS = " << best_link_hitatten << endl;
		// logging.record() << "Raw.BestMEANS = " << best_link_mean;
		// logging.record() << "Raw.BestMRR = " << rmrr / total;
		// logging.record() << "Raw.BestHITS = " << best_link_hitatten;

		std::cout << "Filter.BestMEANS = " << best_link_fmean << endl;
		std::cout << "Filter.BestMRR= " << fmrr / total << endl;
		std::cout << "Filter.BestHITS = " << best_link_fhitatten << endl;
		// logging.record() << "Filter.BestMEANS = " << best_link_fmean;
		// logging.record() << "Filter.BestMRR= " << fmrr / total;
		// logging.record() << "Filter.BestHITS = " << best_link_fhitatten;

		// fprintf(fs_log, "Link prediction\n");
		fprintf(fs_log, "== Raw.BestMEANS = %lf\n", best_link_mean);
		fprintf(fs_log, "== Raw.BestMRR = %lf\n", rmrr / total);
		fprintf(fs_log, "== Raw.BestHITS = %lf\n", best_link_hitatten);

		fprintf(fs_log, "== Filter.BestMEANS = %lf\n", best_link_fmean);
		fprintf(fs_log, "== Filter.BestMRR = %lf\n", fmrr / total);
		fprintf(fs_log, "== Filter.BestHITS = %lf\n", best_link_fhitatten);

		std::cout.flush();
	}

public:
	void test_link_prediction_zeroshot(int hit_rank = 10, const int part = 0)
	{
		reset();

		double mean = 0;
		double hits = 0;
		double fmean = 0;
		double fhits = 0;
		double total = data_model.data_test_true.size();

		double arr_mean[20] = { 0 };
		double arr_total[5] = { 0 };

		cout << endl;

		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			if (i->first.first >= data_model.zeroshot_pointer
				&& i->first.second >= data_model.zeroshot_pointer)
			{
				++arr_total[3];
			}
			else if (i->first.first < data_model.zeroshot_pointer
				&& i->first.second >= data_model.zeroshot_pointer)
			{
				++arr_total[2];
			}
			else if (i->first.first >= data_model.zeroshot_pointer
				&& i->first.second < data_model.zeroshot_pointer)
			{
				++arr_total[1];
			}
			else
			{
				++arr_total[0];
			}
		}

		cout << "0 holds " << arr_total[0] << endl;
		cout << "1 holds " << arr_total[1] << endl;
		cout << "2 holds " << arr_total[2] << endl;
		cout << "3 holds " << arr_total[3] << endl;

		int cnt = 0;

#pragma omp parallel for
		for (auto i = data_model.data_test_true.begin(); i != data_model.data_test_true.end(); ++i)
		{
			++cnt;
			if (cnt % 100 == 0)
			{
				std::cout << cnt << ',';
				std::cout.flush();
			}

			pair<pair<int, int>, int> t = *i;
			int frmean = 0;
			int rmean = 0;
			double score_i = prob_triplets(*i);

			if (task_type == LinkPredictionRelationZeroShot || part == 2)
			{
				for (auto j = 0; j != data_model.set_relation.size(); ++j)
				{
					t.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
						++frmean;
				}
			}
			else
			{
				for (auto j = 0; j != data_model.set_entity.size(); ++j)
				{
					if (task_type == LinkPredictionHeadZeroShot || part == 1)
						t.first.first = j;
					else
						t.first.second = j;

					if (score_i >= prob_triplets(t))
						continue;

					++rmean;

					if (data_model.check_data_all.find(t) == data_model.check_data_all.end())
					{
						++frmean;
					}
				}
			}

#pragma omp critical
			{
				if (frmean < hit_rank)
				{
					if (i->first.first >= data_model.zeroshot_pointer
						&& i->first.second >= data_model.zeroshot_pointer)
					{
						++arr_mean[3];
					}
					else if (i->first.first < data_model.zeroshot_pointer
						&& i->first.second >= data_model.zeroshot_pointer)
					{
						++arr_mean[2];
					}
					else if (i->first.first >= data_model.zeroshot_pointer
						&& i->first.second < data_model.zeroshot_pointer)
					{
						++arr_mean[1];
					}
					else
					{
						++arr_mean[0];
					}
				}

				mean += rmean;
				fmean += frmean;
				if (rmean < hit_rank)
					++hits;
				if (frmean < hit_rank)
					++fhits;
			}
		}

		std::cout << endl;
		for (auto i = 0; i < 4; ++i)
		{
			std::cout << i << ':' << arr_mean[i] / arr_total[i] << endl;
			logging.record() << i << ':' << arr_mean[i] / arr_total[i];
		}
		logging.record();

		best_link_mean = min(best_link_mean, mean / total);
		best_link_hitatten = max(best_link_hitatten, hits / total);
		best_link_fmean = min(best_link_fmean, fmean / total);
		best_link_fhitatten = max(best_link_fhitatten, fhits / total);

		std::cout << "Raw.BestMEANS = " << best_link_mean << endl;
		std::cout << "Raw.BestHITS = " << best_link_hitatten << endl;
		logging.record() << "Raw.BestMEANS = " << best_link_mean;
		logging.record() << "Raw.BestHITS = " << best_link_hitatten;
		std::cout << "Filter.BestMEANS = " << best_link_fmean << endl;
		std::cout << "Filter.BestHITS = " << best_link_fhitatten << endl;
		logging.record() << "Filter.BestMEANS = " << best_link_fmean;
		logging.record() << "Filter.BestHITS = " << best_link_fhitatten;
	}

	virtual void draw(const string& filename, const int radius, const int id_relation) const
	{
		return;
	}

	virtual void draw(const string& filename, const int radius,
		const int id_head, const int id_relation)
	{
		return;
	}

	virtual void report(const string& filename) const
	{
		return;
	}
public:
	~Model()
	{
		logging.record();
		if (be_deleted_data_model)
		{
			delete &data_model;
			delete &logging;
		}
	}

public:
	int count_entity() const
	{
		return data_model.set_entity.size();
	}

	int count_relation() const
	{
		return data_model.set_relation.size();
	}

	const DataModel& get_data_model() const
	{
		return data_model;
	}

public:
	virtual void save(const string& filename, FILE * fs_log)
	{
		cout << "BAD" << endl;
		return;
	}

	virtual void load(const string& filename, FILE * fs_log)
	{
		cout << "BAD!" << endl;
		return;
	}

	virtual vec entity_representation(int entity_id) const
	{
		cout << "BAD";
		return NULL;
	}

	virtual vec relation_representation(int relation_id) const
	{
		cout << "BAD";
		return NULL;
	}

// distributed KGE interface
public:

	virtual void send_entities() {

		cout << "BAD!" << endl;
		
		return;
	}

	virtual void send_relations() {

		cout << "BAD!" << endl;
		
		return;
	}

	virtual void receive_entities() {

		cout << "BAD!" << endl;
		
		return;
	}

	virtual void receive_relations() {

		cout << "BAD!" << endl;

		return;
	}
};
