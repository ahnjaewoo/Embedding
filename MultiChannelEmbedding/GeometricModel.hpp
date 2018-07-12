#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
#include "DataModel.hpp"
#include "Model.hpp"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fstream>
#include "umHalf.h"

class TransE
	:public Model
{
protected:
	vector<vec>	embedding_entity;
	vector<vec>	embedding_relation;

public:
	const int		dim;
	const double	alpha;
	const double	training_threshold;
	const int	master_epoch;

public:
	TransE(const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0,
		FILE * fs_log = NULL,
		const int precision = 0)
		:Model(dataset, task_type, logging_base_path, is_preprocessed, worker_num, master_epoch, fd, fs_log, precision),
		dim(dim), alpha(alpha), training_threshold(training_threshold), master_epoch(master_epoch)
	{

		// printf("[info] GeometricModel.hpp > TransE constructor called");
		// fprintf(fs_log, "[info] GeometricModel.hpp > TransE constructor called");

		// logging.record() << "\t[Name]\tTransE";
		// logging.record() << "\t[Dimension]\t" << dim;
		// logging.record() << "\t[Learning Rate]\t" << alpha;
		// logging.record() << "\t[Training Threshold]\t" << training_threshold;

		embedding_entity.resize(count_entity());
		embedding_relation.resize(count_relation());

		if (is_preprocessed)
		{
			for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem) {elem = vec(dim); });
			for_each(embedding_relation.begin(), embedding_relation.end(), [=](vec& elem) {elem = vec(dim); });




			// load 함수 부분이 소켓과 연동되어야 함, load 함수 자체에서 처리
			load("", fs_log);
		}
		else
		{
			for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem){elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });
			for_each(embedding_relation.begin(), embedding_relation.end(), [=](vec& elem){elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });
		}

	}

	TransE(const Dataset& dataset,
		const string& file_zero_shot,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0,
		FILE * fs_log = NULL,
		const int precision = 0)
		:Model(dataset, file_zero_shot, task_type, logging_base_path, is_preprocessed, worker_num, master_epoch, fd, fs_log, precision),
		dim(dim), alpha(alpha), training_threshold(training_threshold), master_epoch(master_epoch)
	{

		// printf("[info] GeometricModel.hpp > TransE constructor called");
		// fprintf(fs_log, "[info] GeometricModel.hpp > TransE constructor called");
		
		// logging.record() << "\t[Name]\tTransE";
		// logging.record() << "\t[Dimension]\t" << dim;
		// logging.record() << "\t[Learning Rate]\t" << alpha;
		// logging.record() << "\t[Training Threshold]\t" << training_threshold;

		embedding_entity.resize(count_entity());
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem) {elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });

		embedding_relation.resize(count_relation());
		for_each(embedding_relation.begin(), embedding_relation.end(), [=](vec& elem) {elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });
	}

	virtual void draw(const string& filename, const int radius, const int id_relation) const
	{
		mat	record(radius*6.0 + 10, radius*6.0 + 10);
		record.fill(255);
		for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
		{
			if (i->second == id_relation)
			{
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]),
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1])) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) + 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) + 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) + 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) - 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) - 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) + 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) - 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) - 1) = 0;
			}
		}

		string relation_name = data_model.relation_id_to_name[id_relation];
		record.save(filename + replace_all(relation_name, "/", "_") + ".ppm", pgm_binary);
	}

	virtual void draw(const string& filename, const int radius,
		int id_head, int id_relation)
	{
		mat	record(radius*6.0 + 4, radius*6.0 + 4);
		record.fill(255);
		for (auto i = data_model.rel_tails.at(id_relation).at(id_head).begin();
			i != data_model.rel_tails.at(id_relation).at(id_head).end();
			++i)
		{
			if (rand() % 100 < 80)
				continue;

			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]),
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1])) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 1,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 1) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 1,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 1) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 1,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 1) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 1,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 1) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 2,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 2) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 2,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 2) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 2,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 2) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 2,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 2) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 3,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 3) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 3,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 3) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 3,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 3) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 3,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 3) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 4,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 4) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 4,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 4) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 4,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 4) = 0;
			record(radius * (3.0 + embedding_entity[*i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 4,
				radius *(3.0 + embedding_entity[*i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 4) = 0;
		}

		auto changes = make_pair(make_pair(id_head, id_head), id_relation);
		priority_queue<pair<double, int>>	heap_entity;
		for (auto i = 0; i<count_entity(); ++i)
		{
			changes.first.first = i;
			heap_entity.push(make_pair(prob_triplets(changes), i));
		}

		for (auto j = 0; j<40; ++j)
		{
			int i = heap_entity.top().second;
			int t = 100;
			while (t--)
				heap_entity.pop();

			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]),
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1])) = 1;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 1,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 0) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 0,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 1) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 1,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 0) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 0,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 1) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 2,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 0) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 0,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 2) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 2,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 0) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 0,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 2) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 3,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 0) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 0,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 3) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 3,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 0) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 0,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 3) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 4,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 0) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) + 0,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 4) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 4,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) - 0) = 0;
			record(radius * (3.0 + embedding_entity[i][0] - embedding_entity[id_head][0] - embedding_relation[id_relation][0]) - 0,
				radius *(3.0 + embedding_entity[i][1] - embedding_entity[id_head][1] - embedding_relation[id_relation][1]) + 4) = 0;

		}

		record.save(filename, pgm_binary);
	}

	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		vec error = embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];

		return -sum(abs(error));
	}

	virtual double prob_triplets_test(const pair<pair<int, int>, int>& triplet)
	{
		vec error = embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];


		//cout << "embedding_entity[triplet.first.first] = " << embedding_entity[triplet.first.first] << "\n";
		//cout << "embedding_relation[triplet.second] = " << embedding_relation[triplet.second] << "\n";
		//cout << "embedding_entity[triplet.first.second] = " << embedding_entity[triplet.first.second] << "\n";
		//cout << "-sum(abs(error)) = " << -sum(abs(error)) << "\n";

		return -sum(abs(error));
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		head_f += alpha * sign(head_f + relation_f - tail_f);
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		if (norm_L2(head) > 1.0)
			head = normalise(head);

		if (norm_L2(tail) > 1.0)
			tail = normalise(tail);

		if (norm_L2(relation) > 1.0)
			relation = normalise(relation);

		if (norm_L2(head_f) > 1.0)
			head_f = normalise(head_f);

		if (norm_L2(tail_f) > 1.0)
			tail_f = normalise(tail_f);
	}

	virtual void train_triplet_parts(const pair<pair<int, int>, int>& triplet)
	{
		int head_id = triplet.first.first;
		int tail_id = triplet.first.second;
		int relation_id = triplet.second;

		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet_parts(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		int head_f_id = triplet_f.first.first;
		int tail_f_id = triplet_f.first.second;
		int relation_f_id = triplet_f.second;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		if (data_model.check_anchor.find(head_id) == data_model.check_anchor.end())
		{
			head -= alpha * sign(head + relation - tail);
			if (norm_L2(head) > 1.0) head = normalise(head);
		}
		if (data_model.check_anchor.find(tail_id) == data_model.check_anchor.end())
		{
			tail += alpha * sign(head + relation - tail);
			if (norm_L2(tail) > 1.0) tail = normalise(tail);
		}
		if (data_model.check_anchor.find(head_f_id) == data_model.check_anchor.end())
		{
			head_f += alpha * sign(head_f + relation_f - tail_f);
			if (norm_L2(head_f) > 1.0) head_f = normalise(head_f);
		}
		if (data_model.check_anchor.find(tail_f_id) == data_model.check_anchor.end())
		{
			tail_f -= alpha * sign(head_f + relation_f - tail_f);
			if (norm_L2(tail_f) > 1.0) tail_f = normalise(tail_f);
		}
	}

 virtual void train_triplet_parts_relation(const pair<pair<int, int>, int>& triplet)
 {
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

	  pair<pair<int, int>, int> triplet_f;
	  data_model.sample_false_triplet_relation_parts(triplet, triplet_f);

	  if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

    vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		relation -= alpha * sign(head + relation - tail);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		if (norm_L2(relation) > 1.0)
			relation = normalise(relation);
		if (norm_L2(relation_f) > 1.0)
			relation_f = normalise(relation_f);
	}

	virtual void relation_reg(int i, int j, double factor)
	{
		if (i == j)
			return;

		embedding_relation[i] -= alpha * factor * sign(as_scalar(embedding_relation[i].t()*embedding_relation[j])) * embedding_relation[j];
		embedding_relation[j] -= alpha * factor * sign(as_scalar(embedding_relation[i].t()*embedding_relation[j])) * embedding_relation[i];
	}

	virtual void entity_reg(int i, int j, double factor)
	{
		if (i == j)
			return;

		embedding_entity[i] -= alpha * factor * sign(as_scalar(embedding_entity[i].t()*embedding_entity[j])) * embedding_entity[j];
		embedding_entity[j] -= alpha * factor * sign(as_scalar(embedding_entity[i].t()*embedding_entity[j])) * embedding_entity[i];
	}

public:
	virtual vec entity_representation(int entity_id) const
	{
		return embedding_entity[entity_id];
	}

	virtual vec relation_representation(int relation_id) const
	{
		return embedding_relation[relation_id];
	}

	/* original save function
	virtual void save(const string& filename) override
	{
		// master_epoch이 짝수일 때 (entity embedding ㄱㄱ)
		if (master_epoch % 2 == 0)
		{
			ofstream fout_entity("../tmp/entity_vectors_updated_" + filename + ".txt", ios::binary);
			for (int i = 0; i < count_entity(); i++)
			{
				if (data_model.check_anchor.find(i) == data_model.check_anchor.end()
				&& data_model.check_parts.find(i) != data_model.check_parts.end())
				{
					fout_entity << data_model.entity_id_to_name[i] << '\t';
					for (int j =0; j < dim; j++)
					{
						fout_entity << embedding_entity[i](j) << " ";
					}
					fout_entity << '\n';
				}
			}
			fout_entity.close();
		}
		// master_epoch이 홀수일 때 (relation embedding ㄱㄱ)
		else
		{
			ofstream fout_relation("../tmp/relation_vectors_updated_" + filename + ".txt", ios::binary);
			for (int i = 0; i < count_relation(); i++)
			{
				if (data_model.set_relation_parts.find(i) != data_model.set_relation_parts.end())
				{
					fout_relation << data_model.relation_id_to_name[i] << '\t';
					for (int j = 0; j < dim; j++)
					{
						fout_relation << embedding_relation[i](j) << " ";
					}
					fout_relation << '\n';
				}
			}
			fout_relation.close();
		}
	}
	*/

	virtual void save(const string& filename, FILE * fs_log) override{

		// int string_len;
		int count = 0;
		float value_to_send;

		// master_epoch이 짝수일 때, entity embedding
		if (master_epoch % 2 == 0){

			int checksum = 0;
			int flag = 0;

			while(checksum != 1){

				count = 0;

				for (int i = 0; i < count_entity(); i++){

					if (data_model.check_anchor.find(i) == data_model.check_anchor.end()
					&& data_model.check_parts.find(i) != data_model.check_parts.end()){

						count++;
					}
				}

				count = htonl(count);
				send(fd, &count, sizeof(count), 0);
				count = ntohl(count);
				
				// 원소 한 번에 보냄 (엔티티 한 번에) 를 밑에 사용하면, 아래 for 문을 주석처리

				//for (int i = 0; i < count_entity(); i++){

				//	if (data_model.check_anchor.find(i) == data_model.check_anchor.end()
				//	&& data_model.check_parts.find(i) != data_model.check_parts.end()){
						// entity_id 가 string 으로 주어진 경우
						// entity_id_to_name 을 이용해 string 을 가져와서 보냄
						/*
						string_len = data_model.entity_id_to_name[i].size();
						string_len = htonl(string_len);
						send(fd, &string_len, sizeof(string_len), 0);
						string_len = ntohl(string_len);
						send(fd, data_model.entity_id_to_name[i].c_str() , string_len, 0);
						*/

						// entity_id 가 int 로 주어진 경우
						
				//		i = htonl(i);
				//		send(fd, &i, sizeof(int), 0);
				//		i = ntohl(i);
						
						// 원소 하나씩 보냄
						/*
						for (int j = 0; j < dim; j++){

							value_to_send = embedding_entity[i](j);
							send(fd, &value_to_send, sizeof(value_to_send), 0);
						}
						*/
						//.....................

						// 원소 한 번에 보냄
						/*
						if(precision == 0) {
							float * vector_buff = (float *)calloc(dim + 1, sizeof(float));
							for (int j = 0; j < dim; j++) {

								vector_buff[j] = embedding_entity[i](j);
							}
							send(fd, vector_buff, dim * sizeof(float), 0);

							free(vector_buff);
						}
						else {
							half * vector_buff = (half *)calloc(dim + 1, sizeof(half));
							for (int j = 0; j < dim; j++) {

								vector_buff[j] = (half) embedding_entity[i](j);
							}
							send(fd, vector_buff, dim * sizeof(half), 0);

							free(vector_buff);
						}
						*/
						//.....................

				//	}
				//}


				// 원소 한 번에 보냄 (엔티티 한 번에)

				if (precision == 0) {

					int buff_idx = 0;
					int * idx_buff = (int *)calloc(count + 1, sizeof(int));
					float * vector_buff = (float *)calloc(count * dim + 1, sizeof(float));

					for (int i = 0; i < count_entity(); i++) {

						if (data_model.check_anchor.find(i) == data_model.check_anchor.end()
						&& data_model.check_parts.find(i) != data_model.check_parts.end()){						

							idx_buff[buff_idx] = htonl(i);

							for (int j = 0; j < dim; j++) {

								vector_buff[dim * buff_idx + j] = embedding_entity[i](j);
							}

							buff_idx++;
						}
					}

					send(fd, idx_buff, count * sizeof(int), 0);
					send(fd, vector_buff, count * dim * sizeof(float), 0);

					free(idx_buff);
					free(vector_buff);
				}
				else{

					int buff_idx = 0;
					int * idx_buff = (int *)calloc(count + 1, sizeof(int));
					half * vector_buff = (half *)calloc(count * dim + 1, sizeof(half));

					for (int i = 0; i < count_entity(); i++) {

						if (data_model.check_anchor.find(i) == data_model.check_anchor.end()
						&& data_model.check_parts.find(i) != data_model.check_parts.end()){	

							idx_buff[buff_idx] = htonl(i);

							for (int j = 0; j < dim; j++) {

								vector_buff[dim * buff_idx + j] = (half)embedding_entity[i](j);
							}

							buff_idx++;
						}
					}

					send(fd, idx_buff, count * sizeof(int), 0);
					send(fd, vector_buff, count * dim * sizeof(half), 0);

					free(idx_buff);
					free(vector_buff);
				}

				//.....................

                if (recv(fd, &flag, sizeof(flag), MSG_WAITALL) < 0){

                	printf("[error] GeometricModel.hpp > recv flag (phase 3)\n");
                    printf("[error] GeometricModel.hpp > return -1\n");
                    fprintf(fs_log, "[error] GeometricModel.hpp > recv flag (phase 3)\n");
                    fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
                    close(fd);
                    fclose(fs_log);
                    std::exit(-1);
                }

                flag = ntohl(flag);

                if(flag == 1234){

                	checksum = 1;
                }
                else if(flag == 9876){

                	printf("[error] GeometricModel.hpp > retry phase 3 (entity)\n");
                	fprintf(fs_log, "[error] GeometricModel.hpp > retry phase 3 (entity)\n");
                	checksum = 0;
                }
                else{

                	printf("[error] GeometricModel.hpp > unknown error of phase 3 (entity)\n");
                	printf("[error] GeometricModel.hpp > flag = %d\n", flag);
					//printf("[error] GeometricModel.hpp > recv value = %d\n", recv_val);
                	printf("[error] GeometricModel.hpp > retry phase 3 (entity)\n");
                    fprintf(fs_log, "[error] GeometricModel.hpp > unknown error of phase 3 (entity)\n");
                    fprintf(fs_log, "[error] GeometricModel.hpp > flag = %d\n", flag);
					//fprintf(fs_log, "[error] GeometricModel.hpp > recv value = %d\n", recv_val);
                    fprintf(fs_log, "[error] GeometricModel.hpp > retry phase 3 (entity)\n");
                    checksum = 0;
                }
			}

			// printf("[info] GeometricModel.hpp > phase 3 (entity save) finish\n");
			// fprintf(fs_log, "[info] GeometricModel.hpp > phase 3 (entity save) finish\n");
		}
		else{
			// master_epoch 이 홀수일 때, relation embedding
			int checksum = 0;
			int flag = 0;

			while(checksum != 1){

				for (int i = 0; i < count_relation(); i++){

					if (data_model.set_relation_parts.find(i) != data_model.set_relation_parts.end()){

						count++;
					}
				}

				count = htonl(count);
				send(fd, &count, sizeof(count), 0);
				count = ntohl(count);

				// 원소 한 번에 보냄 (릴레이션 한 번에) 를 밑에 사용하면, 아래 for 문을 주석처리

				//for (int i = 0; i < count_relation(); i++){

				//	if (data_model.set_relation_parts.find(i) != data_model.set_relation_parts.end()){

						// relation_id 가 string 으로 주어진 경우
						// relation_id_to_name 을 이용해 string 을 가져와서 보냄
						/*
						string_len = data_model.relation_id_to_name[i].size();
						string_len = htonl(string_len);
						send(fd, &string_len, sizeof(string_len), 0);
						string_len = ntohl(string_len);
						send(fd, data_model.relation_id_to_name[i].c_str() , string_len, 0);
						*/

						// relation_id 가 int 로 주어진 경우
						
				//		i = htonl(i);
				//		send(fd, &i, sizeof(int), 0);
				//		i = ntohl(i);
						
						// 원소 하나씩 보냄
						/*
						for (int j = 0; j < dim; j++){

							value_to_send = embedding_relation[i](j);
							send(fd, &value_to_send, sizeof(value_to_send), 0);
						}
						*/
						//.....................

						// 원소 한 번에 보냄
						/*
						if(precision == 0) {
							float * vector_buff = (float *)calloc(dim + 1, sizeof(float));
							for (int j = 0; j < dim; j++) {

								vector_buff[j] = embedding_relation[i](j);
							}
							send(fd, vector_buff, dim * sizeof(float), 0);

							free(vector_buff);
						}
						else {
							half * vector_buff = (half *)calloc(dim + 1, sizeof(half));
							for (int j = 0; j < dim; j++) {

								vector_buff[j] = embedding_relation[i](j);
							}
							send(fd, vector_buff, dim * sizeof(half), 0);

							free(vector_buff);
						}
						*/
						//.....................
				//	}
				//}

				// 원소 한 번에 보냄 (릴레이션 한 번에)

				if (precision == 0) {

					int buff_idx = 0;
					int * idx_buff = (int *)calloc(count + 1, sizeof(int));
					float * vector_buff = (float *)calloc(count * dim + 1, sizeof(float));

					for (int i = 0; i < count_relation(); i++) {

						if (data_model.set_relation_parts.find(i) != data_model.set_relation_parts.end()){

							idx_buff[buff_idx] = htonl(i);

							for (int j = 0; j < dim; j++) {

								vector_buff[dim * buff_idx + j] = embedding_relation[i](j);
							}

							buff_idx++;
						}
					}

					send(fd, idx_buff, count * sizeof(int), 0);
					send(fd, vector_buff, count * dim * sizeof(float), 0);

					free(idx_buff);
					free(vector_buff);
				}
				else{

					int buff_idx = 0;
					int * idx_buff = (int *)calloc(count + 1, sizeof(int));
					half * vector_buff = (half *)calloc(count * dim + 1, sizeof(half));

					for (int i = 0; i < count_relation(); i++) {

						if (data_model.set_relation_parts.find(i) != data_model.set_relation_parts.end()){

							idx_buff[buff_idx] = htonl(i);

							for (int j = 0; j < dim; j++) {

								vector_buff[dim * buff_idx + j] = (half)embedding_relation[i](j);
							}

							buff_idx++;
						}
					}
					send(fd, idx_buff, count * sizeof(int), 0);
					send(fd, vector_buff, count * dim * sizeof(half), 0);

					free(idx_buff);
					free(vector_buff);
				}

				//.....................

                if (recv(fd, &flag, sizeof(flag), MSG_WAITALL) < 0){

                	printf("[error] GeometricModel.hpp > recv flag (phase 3)\n");
                    printf("[error] GeometricModel.hpp > return -1\n");
                    fprintf(fs_log, "[error] GeometricModel.hpp > recv flag (phase 3)\n");
                    fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
                    close(fd);
                    fclose(fs_log);
                    std::exit(-1);
                }

                flag = ntohl(flag);

                if(flag == 1234){

                	checksum = 1;
                }
                else if(flag == 9876){

                	printf("[error] GeometricModel.hpp > retry phase 3 (relation)\n");
                	fprintf(fs_log, "[error] GeometricModel.hpp > retry phase 3 (relation)\n");
                	checksum = 0;
                }
                else{

                	printf("[error] GeometricModel.hpp > unknown error of phase 3 (relation)\n");
                	printf("[error] GeometricModel.hpp > flag = %d\n", flag);
                	printf("[error] GeometricModel.hpp > retry phase 3 (relation)\n");
                    fprintf(fs_log, "[error] GeometricModel.hpp > unknown error of phase 3 (relation)\n");
                    fprintf(fs_log, "[error] GeometricModel.hpp > flag = %d\n", flag);
                    fprintf(fs_log, "[error] GeometricModel.hpp > retry phase 3 (relation)\n");
                    checksum = 0;
                }
			}

			// printf("[info] GeometricModel.hpp > phase 3 (relation save) finish\n");
			// fprintf(fs_log, "[info] GeometricModel.hpp > phase 3 (relation save) finish\n");
		}
	}

	virtual void load(const string& filename, FILE * fs_log) override
	{

		// filename 파라미터가 전혀 사용되지 않음을 참고
		// string key;
		int key_length;
		float temp_vector;
		vector<char> temp_buff(256);
  		int success = 0;
		int flag = 0;

		while(!success){

			try{

				int * id_buff = (int *)calloc(count_entity() + 1, sizeof(int));
				
				for (int i = 0; i < count_entity(); i++) {
					
					int entity_id;
	                if (recv(fd, &entity_id, sizeof(int), MSG_WAITALL) < 0){

	                	printf("[error] GeometricModel.hpp > recv entity_id\n");
	                	fprintf(fs_log, "[error] GeometricModel.hpp > recv entity_id\n");
	                    close(fd);
	                    fclose(fs_log);
	                    break;
	                }

					entity_id = ntohl(entity_id);
					id_buff[i] = entity_id; 		// 원소 한 번에 받을 때 사용 (2 단계)

					// 원소 하나씩 받음
					/*

					for (int j = 0; j < dim; j++)
					{
						if (recv(fd, &temp_vector, sizeof(temp_vector), MSG_WAITALL) < 0){

							printf("[error] GeometricModel.hpp > recv temp_vector for loop of dim\n");
							printf("[error] GeometricModel.hpp > return -1\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > recv temp_vector for loop of dim\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
							close(fd);
							fclose(fs_log);
							std::exit(-1);
						}

						embedding_entity[entity_id](j) = temp_vector;						
					}
					*/
					//.....................

                    // 원소 한 번에 받음 - 1 단계
					/*
                    if(precision == 0) {
						float * vector_buff = (float *)calloc(dim + 1, sizeof(float));
						if (recv(fd, vector_buff, dim * sizeof(float), MSG_WAITALL) < 0){

							printf("[error] GeometricModel.hpp > recv vector_buff for loop of dim (entity)\n");
							printf("[error] GeometricModel.hpp > return -1\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > recv vector_buff for loop of dim (entity)\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
							close(fd);
							fclose(fs_log);
							std::exit(-1);
						}

						for (int j = 0; j < dim; j++) {

							embedding_entity[entity_id](j) = vector_buff[j];						
						}

						free(vector_buff);
					}
					else {
						half * vector_buff = (half *)calloc(dim + 1, sizeof(half));
						if (recv(fd, vector_buff, dim * sizeof(half), MSG_WAITALL) < 0){

							printf("[error] GeometricModel.hpp > recv vector_buff for loop of dim (entity)\n");
							printf("[error] GeometricModel.hpp > return -1\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > recv vector_buff for loop of dim (entity)\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
							close(fd);
							fclose(fs_log);
							std::exit(-1);
						}

						for (int j = 0; j < dim; j++) {

							embedding_entity[entity_id](j) = (float) vector_buff[j];						
						}

						free(vector_buff);
					}
					*/
					//.....................
				}

				// 원소 한 번에 받음 - 2 단계 (엔티티 하나씩)
				/*
				for (int i = 0; i < count_entity(); i++) {

                    if(precision == 0) {
						float * vector_buff = (float *)calloc(dim + 1, sizeof(float));
						if (recv(fd, vector_buff, dim * sizeof(float), MSG_WAITALL) < 0){

							printf("[error] GeometricModel.hpp > recv vector_buff for loop of dim (entity)\n");
							printf("[error] GeometricModel.hpp > return -1\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > recv vector_buff for loop of dim (entity)\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
							close(fd);
							fclose(fs_log);
							std::exit(-1);
						}

						for (int j = 0; j < dim; j++) {

							embedding_entity[id_buff[i]](j) = vector_buff[j];						
						}

						free(vector_buff);
					}
					else {
						half * vector_buff = (half *)calloc(dim + 1, sizeof(half));
						if (recv(fd, vector_buff, dim * sizeof(half), MSG_WAITALL) < 0){

							printf("[error] GeometricModel.hpp > recv vector_buff for loop of dim (entity)\n");
							printf("[error] GeometricModel.hpp > return -1\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > recv vector_buff for loop of dim (entity)\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
							close(fd);
							fclose(fs_log);
							std::exit(-1);
						}

						for (int j = 0; j < dim; j++) {

							embedding_entity[id_buff[i]](j) = (float) vector_buff[j];						
						}

						free(vector_buff);
					}
				}
				*/
				//.....................

				// 원소 한 번에 받음 - 2 단계 (엔티티 한 번에)

				if (precision == 0) {

					float * vector_buff = (float *)calloc(count_entity() * dim + 1, sizeof(float));
					if (recv(fd, vector_buff, count_entity() * dim * sizeof(float), MSG_WAITALL) < 0){

						printf("[error] GeometricModel.hpp > recv vector_buff for loop of dim (entity)\n");
						printf("[error] GeometricModel.hpp > return -1\n");
						fprintf(fs_log, "[error] GeometricModel.hpp > recv vector_buff for loop of dim (entity)\n");
						fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
						close(fd);
						fclose(fs_log);
						std::exit(-1);
					}

					for (int i = 0; i < count_entity(); i++) {

						for (int j = 0; j < dim; j++) {

							embedding_entity[id_buff[i]](j) = vector_buff[dim * i + j];
						}
					}

					free(vector_buff);
				}
				else{

					half * vector_buff = (half *)calloc(count_entity() * dim + 1, sizeof(half));
					if (recv(fd, vector_buff, count_entity() * dim * sizeof(half), MSG_WAITALL) < 0){

						printf("[error] GeometricModel.hpp > recv vector_buff for loop of dim (entity)\n");
						printf("[error] GeometricModel.hpp > return -1\n");
						fprintf(fs_log, "[error] GeometricModel.hpp > recv vector_buff for loop of dim (entity)\n");
						fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
						close(fd);
						fclose(fs_log);
						std::exit(-1);
					}

					for (int i = 0; i < count_entity(); i++) {

						for (int j = 0; j < dim; j++) {

							embedding_entity[id_buff[i]](j) = (float) vector_buff[dim * i + j];
						}
					}

					free(vector_buff);
				}

				//.....................

				free(id_buff);

				flag = 1234;
				flag = htonl(flag);
				send(fd, &flag, sizeof(flag), 0);
				success = 1;
			}
        	catch(std::exception& e){

                printf("[error] GeometricModel.hpp > exception occured\n");
                printf("%s\n", e.what());
                fprintf(fs_log, "[error] GeometricModel.hpp > exception occured\n");
                fprintf(fs_log, "%s\n", e.what());
                success = 0;
                flag = 9876;
                flag = htonl(flag);
                send(fd, &flag, sizeof(flag), 0);
        	}
        }
		// printf("[info] GeometricModel.hpp > load entity finish\n");
		// fprintf(fs_log, "[info] GeometricModel.hpp > load entity finish\n");

		success = 0;
		flag = 0;

		while(!success){

			try{

				int * id_buff = (int *)calloc(count_relation() + 1, sizeof(int));

				printf("12\n");

				for (int i = 0; i < count_relation(); i++) {
					
					int relation_id;
	                if (recv(fd, &relation_id, sizeof(int), MSG_WAITALL) < 0){

	                	printf("[error] recv relation_id in GeometricModel.hpp\n");
	                	fprintf(fs_log, "[error] recv relation_id in GeometricModel.hpp\n");
	                    close(fd);
	                    fclose(fs_log);
	                    break;
	                }

					relation_id = ntohl(relation_id);
					id_buff[i] = relation_id; 		// 원소 한 번에 받을 때 사용 (2 단계)
					//.....................
					
					// 원소 하나씩 받음
					/*
					for (int j = 0; j < dim; j++){

		                if (recv(fd, &temp_vector, sizeof(temp_vector), MSG_WAITALL) < 0){

		                	printf("[error] GeometricModel.hpp > recv temp_vector for loop of dim\n");
                            printf("[error] GeometricModel.hpp > return -1\n");
							fprintf(fs_log, "[error] GeometricModel.hpp 932 line\n");
		                	fprintf(fs_log, "[error] GeometricModel.hpp > recv temp_vector for loop of dim\n");
                            fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
                    		close(fd);
                    		fclose(fs_log);
                    		std::exit(-1);
		                }
		                
		                embedding_relation[relation_id](j) = temp_vector;
					}
					*/
					//.....................

                    // 원소 한 번에 받음 - 1 단계
                    /*
					if(precision == 0) {
						float * vector_buff = (float *)calloc(dim + 1, sizeof(float));
						if (recv(fd, vector_buff, dim * sizeof(float), MSG_WAITALL) < 0){

							printf("[error] GeometricModel.hpp > recv vector_buff for loop of dim (relation)\n");
							printf("[error] GeometricModel.hpp > return -1\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > recv vector_buff for loop of dim (relation)\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
							close(fd);
							fclose(fs_log);
							std::exit(-1);
						}

						for (int j = 0; j < dim; j++) {

							embedding_relation[relation_id](j) = vector_buff[j];
						}

						free(vector_buff);
					}
					else {
						half * vector_buff = (half *)calloc(dim + 1, sizeof(half));
						if (recv(fd, vector_buff, dim * sizeof(half), MSG_WAITALL) < 0){

							printf("[error] GeometricModel.hpp > recv vector_buff for loop of dim (relation)\n");
							printf("[error] GeometricModel.hpp > return -1\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > recv vector_buff for loop of dim (relation)\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
							close(fd);
							fclose(fs_log);
							std::exit(-1);
						}

						for (int j = 0; j < dim; j++) {

							embedding_relation[relation_id](j) = (float) vector_buff[j];
						}

						free(vector_buff);
					}
					*/
                    //.....................
				}

				// 원소 한 번에 받음 - 2 단계
				/*
				for (int i = 0; i < count_relation(); i++) {

					if(precision == 0) {
						float * vector_buff = (float *)calloc(dim + 1, sizeof(float));
						if (recv(fd, vector_buff, dim * sizeof(float), MSG_WAITALL) < 0){

							printf("[error] GeometricModel.hpp > recv vector_buff for loop of dim (relation)\n");
							printf("[error] GeometricModel.hpp > return -1\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > recv vector_buff for loop of dim (relation)\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
							close(fd);
							fclose(fs_log);
							std::exit(-1);
						}

						for (int j = 0; j < dim; j++) {

							embedding_relation[id_buff[i]](j) = vector_buff[j];
						}

						free(vector_buff);
					}
					else {
						half * vector_buff = (half *)calloc(dim + 1, sizeof(half));
						if (recv(fd, vector_buff, dim * sizeof(half), MSG_WAITALL) < 0){

							printf("[error] GeometricModel.hpp > recv vector_buff for loop of dim (relation)\n");
							printf("[error] GeometricModel.hpp > return -1\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > recv vector_buff for loop of dim (relation)\n");
							fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
							close(fd);
							fclose(fs_log);
							std::exit(-1);
						}

						for (int j = 0; j < dim; j++) {

							embedding_relation[id_buff[i]](j) = (float) vector_buff[j];
						}

						free(vector_buff);
					}
				}
				*/
				//.....................

				// 원소 한 번에 받음 - 2 단계 (릴레이션 한 번에)

				printf("34\n");

				if (precision == 0) {

					float * vector_buff = (float *)calloc(count_relation() * dim + 1, sizeof(float));
					if (recv(fd, vector_buff, count_relation() * dim * sizeof(float), MSG_WAITALL) < 0){

						printf("[error] GeometricModel.hpp > recv vector_buff for loop of dim (relation)\n");
						printf("[error] GeometricModel.hpp > return -1\n");
						fprintf(fs_log, "[error] GeometricModel.hpp > recv vector_buff for loop of dim (relation)\n");
						fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
						close(fd);
						fclose(fs_log);
						std::exit(-1);
					}

					printf("56\n");

					for (int i = 0; i < count_relation(); i++) {

						for (int j = 0; j < dim; j++) {

							embedding_relation[id_buff[i]](j) = vector_buff[dim * i + j];
						}
					}

					printf("78\n");

					free(vector_buff);
				}
				else{

					half * vector_buff = (half *)calloc(count_relation() * dim + 1, sizeof(half));
					if (recv(fd, vector_buff, count_relation() * dim * sizeof(half), MSG_WAITALL) < 0){

						printf("[error] GeometricModel.hpp > recv vector_buff for loop of dim (relation)\n");
						printf("[error] GeometricModel.hpp > return -1\n");
						fprintf(fs_log, "[error] GeometricModel.hpp > recv vector_buff for loop of dim (relation)\n");
						fprintf(fs_log, "[error] GeometricModel.hpp > return -1\n");
						close(fd);
						fclose(fs_log);
						std::exit(-1);
					}

					for (int i = 0; i < count_relation(); i++) {

						for (int j = 0; j < dim; j++) {

							embedding_relation[id_buff[i]](j) = (float) vector_buff[dim * i + j];
						}
					}

					free(vector_buff);
				}

				//.....................

				free(id_buff);


                flag = 1234;
                flag = htonl(flag);
                send(fd, &flag, sizeof(flag), 0);
                success = 1;
			}
			catch(std::exception& e){

				printf("[error] GeometricModel.hpp > exception occured\n");
				printf("%s\n", e.what());
				fprintf(fs_log, "[error] GeometricModel.hpp > exception occured\n");
				fprintf(fs_log, "%s\n", e.what());
				success = 0;
				flag = 9876;
				flag = htonl(flag);
				send(fd, &flag, sizeof(flag), 0);
			}
		}
		// printf("[info] GeometricModel.hpp > load relation finish\n");
		// fprintf(fs_log, "[info] GeometricModel.hpp > load relation finish\n");
		
		// printf("[info] GeometricModel.hpp > load function finish\n");
		// fprintf(fs_log, "[info] GeometricModel.hpp > load function finish\n");
	}

	/*
	// 원본 load 함수
	virtual void load(const string& filename) override
	{

		ifstream fin_entity("../tmp/entity_vectors.txt", ios::binary);
		ifstream fin_relation("../tmp/relation_vectors.txt", ios::binary);

		string key;


		for (int i = 0; i < count_entity(); i++)
		{
			fin_entity >> key;
			if (data_model.entity_name_to_id.find(key) == data_model.entity_name_to_id.end())
			{
				cout << "entity key does not exist! entity number : " << i << endl;
				return;
			}
			int entity_id = data_model.entity_name_to_id.at(key);

			for (int j = 0; j < dim; j++)
			{
				fin_entity >> embedding_entity[entity_id](j);
			}
		}

		for (int i = 0; i < count_relation(); i++)
		{
			fin_relation >> key;
			if (data_model.relation_name_to_id.find(key) == data_model.relation_name_to_id.end())
			{
				cout << "relation key does not exist!" << endl;
				return;
			}
			int relation_id = data_model.relation_name_to_id.at(key);

			for (int j = 0; j < dim; j++)
			{
				fin_relation >> embedding_relation[relation_id](j);
			}
		}

		fin_entity.close();
		fin_relation.close();
	}
	*/
};

class TransE_ESS
	:public TransE
{
protected:
	const double ESS_factor;

public:
	TransE_ESS(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		double ESS_factor,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0)
		:TransE(dataset, task_type, logging_base_path, dim, alpha, training_threshold, is_preprocessed, worker_num, master_epoch, fd),
		ESS_factor(ESS_factor)
	{
		logging.record() << "\t[Name]\tTransE_ESS";
		logging.record() << "\t[ESS Factor]\t" << ESS_factor;
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		TransE::train_triplet(triplet);
		relation_reg(triplet.second, rand() % count_relation(), ESS_factor);
		entity_reg(triplet.first.first, rand() % count_entity(), ESS_factor);
		entity_reg(triplet.first.second, rand() % count_entity(), ESS_factor);
	}
};

class TransH
	:public TransE
{
protected:
	vector<vec>	embedding_weights;

public:
	TransH(const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0)
		:TransE(dataset, task_type, logging_base_path,
		dim, alpha, training_threshold, is_preprocessed, worker_num, master_epoch, fd)
	{
		logging.record() << "\t[Name]\tTransH";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;

		embedding_weights.resize(count_relation());
		for_each(embedding_weights.begin(), embedding_weights.end(), [=](vec& elem){elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });
	}

	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		vec error = embedding_entity[triplet.first.first]
			- as_scalar(embedding_weights[triplet.second].t() * embedding_entity[triplet.first.first]) * embedding_weights[triplet.second]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second]
			+ as_scalar(embedding_weights[triplet.second].t() * embedding_entity[triplet.first.second]) * embedding_weights[triplet.second];

		return -sum(abs(error));
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		vec& weight = embedding_weights[triplet.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		vec factor_true = sign(head - as_scalar(weight.t()*head)*weight + relation - tail + as_scalar(weight.t()*tail)*weight);
		vec factor_false = sign(head_f - as_scalar(weight.t()*head_f)*weight + relation - tail_f + as_scalar(weight.t()*tail_f)*weight);

		head -= alpha * (eye(dim, dim) - weight * weight.t()) * factor_true;
		tail += alpha * (eye(dim, dim) - weight * weight.t()) * factor_true;
		relation -= alpha * factor_true;
		head_f += alpha * (eye(dim, dim) - weight * weight.t()) * factor_false;
		tail_f -= alpha * (eye(dim, dim) - weight * weight.t()) * factor_false;
		relation_f += alpha * factor_false;
		weight += 2 * alpha * (factor_true * as_scalar(head.t()*weight) - factor_false * as_scalar(tail.t()*weight));

		if (norm_L2(head) > 1.0)
			head = normalise(head);

		if (norm_L2(tail) > 1.0)
			tail = normalise(tail);

		if (norm_L2(relation) > 1.0)
			relation = normalise(relation);

		if (norm_L2(head_f) > 1.0)
			head_f = normalise(head_f);

		if (norm_L2(tail_f) > 1.0)
			tail_f = normalise(tail_f);

		if (norm_L2(weight) > 1.0)
			weight = normalise(weight);
	}

};
class TransA
	:public TransE
{
protected:
	vector<mat>	mat_r;

public:
	TransA(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0)
		:TransE(dataset, task_type, logging_base_path, dim, alpha, training_threshold, is_preprocessed, worker_num, master_epoch, fd)
	{
		logging.record() << "\t[Name]\tTransA";
		mat_r.resize(count_relation());
		for_each(mat_r.begin(), mat_r.end(), [&](mat& m){ m = eye(dim, dim); });
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		vec error = embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];

		return -as_scalar(abs(error).t()*mat_r[triplet.second] * abs(error));
	}

	virtual double training_prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		vec error = embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];

		return -sum(abs(error));
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (training_prob_triplets(triplet) - training_prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		head_f += alpha * sign(head_f + relation_f - tail_f);
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
		relation_f = normalise(relation_f);
	}

	virtual void train(bool last_time)
	{
		TransE::train(alpha);

		if (last_time || task_type == TripletClassification)
		{
			for_each(mat_r.begin(), mat_r.end(), [&](mat& m){ m = eye(dim, dim); });
			for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
			{
				auto triplet = *i;

				vec& head = embedding_entity[triplet.first.first];
				vec& tail = embedding_entity[triplet.first.second];
				vec& relation = embedding_relation[triplet.second];

				pair<pair<int, int>, int> triplet_f;
				data_model.sample_false_triplet(triplet, triplet_f);

				vec& head_f = embedding_entity[triplet_f.first.first];
				vec& tail_f = embedding_entity[triplet_f.first.second];
				vec& relation_f = embedding_relation[triplet_f.second];

				mat_r[triplet.second] -= abs(head + relation - tail) * abs(head + relation - tail).t()
					- abs(head_f + relation_f - tail_f) * abs(head_f + relation_f - tail_f).t();
			}
			for_each(mat_r.begin(), mat_r.end(), [=](mat& elem){elem = normalise(elem); });
		}
	}

	virtual void report(const string& filename) const
	{
		if (task_type == TransA_ReportWeightes)
		{
			for (auto i = mat_r.begin(); i != mat_r.end(); ++i)
			{
				cout << data_model.relation_id_to_name[i - mat_r.begin()] << ":";

				vector<double> weights;
				double total = 0;
				mat mat_l, mat_u;
				lu(mat_l, mat_u, *i);
				for (auto i = 0; i<dim; ++i)
				{
					weights.push_back(mat_u(i, i));
					total += mat_u(i, i);
				}
				sort(weights.begin(), weights.end());
				cout << weights.back() << ",";
				cout << weights[dim / 2] << ",";
				cout << total;
				cout << endl;
			}
		}
	}
};

class TransA_PSD
	:public TransE
{
protected:
	vector<mat>	mat_r;

public:
	TransA_PSD(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0)
		:TransE(dataset, task_type, logging_base_path, dim, alpha, training_threshold, is_preprocessed, worker_num, master_epoch, fd)
	{
		logging.record() << "\t[Name]\tTransA";
		mat_r.resize(count_relation());
		for_each(mat_r.begin(), mat_r.end(), [&](mat& m){ m = eye(dim, dim); });
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		vec error = embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];

		return -as_scalar((error).t()*mat_r[triplet.second] * mat_r[triplet.second].t()*(error));
	}

	virtual double training_prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		vec error = embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];

		return -sum(abs(error));
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		mat& mat_rel = mat_r[triplet.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (training_prob_triplets(triplet) - training_prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		head_f += alpha * sign(head_f + relation_f - tail_f);
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);
	}

	virtual void train(bool last_time)
	{
		TransE::train(alpha);

		if (last_time || task_type == TripletClassification)
		{
			for_each(mat_r.begin(), mat_r.end(), [=](mat& elem){elem = eye(dim, dim); });
			//for(auto i=0; i<10; ++i)
			{
				for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
				{
					auto triplet = *i;

					vec& head = embedding_entity[triplet.first.first];
					vec& tail = embedding_entity[triplet.first.second];
					vec& relation = embedding_relation[triplet.second];

					pair<pair<int, int>, int> triplet_f;
					data_model.sample_false_triplet(triplet, triplet_f);

					vec& head_f = embedding_entity[triplet_f.first.first];
					vec& tail_f = embedding_entity[triplet_f.first.second];
					vec& relation_f = embedding_relation[triplet_f.second];

					mat_r[triplet.second] +=
						0.001 * (head + relation - tail) * (head + relation - tail).t();
					-0.001 * (head_f + relation_f - tail_f) * (head_f + relation_f - tail_f).t();
				}
				for_each(mat_r.begin(), mat_r.end(), [=](mat& elem){elem = elem.i(); });
			}
		}
	}

	virtual void report(const string& filename) const
	{
		if (task_type == TransA_ReportWeightes)
		{
			for (auto i = mat_r.begin(); i != mat_r.end(); ++i)
			{
				cout << data_model.relation_id_to_name[i - mat_r.begin()] << ":";

				vector<double> weights;
				double total = 0;
				mat mat_l, mat_u;
				lu(mat_l, mat_u, *i);
				for (auto i = 0; i<dim; ++i)
				{
					weights.push_back(mat_u(i, i));
					total += mat_u(i, i);
				}
				sort(weights.begin(), weights.end());
				cout << weights.back() << ",";
				cout << weights[dim / 2] << ",";
				cout << total;
				cout << endl;
			}
		}
	}
};

class TransA_PSD_ESS
	:public TransE
{
protected:
	vector<mat>	mat_r;
	const double		ESS_factor;

public:
	TransA_PSD_ESS(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		double ESS_factor,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0)
		:TransE(dataset, task_type, logging_base_path, dim, alpha, training_threshold, is_preprocessed, worker_num, master_epoch, fd), ESS_factor(ESS_factor)
	{
		logging.record() << "\t[Name]\tTransA";
		mat_r.resize(count_relation());
		for_each(mat_r.begin(), mat_r.end(), [&](mat& m){ m = eye(dim, dim); });
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		vec error = embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];

		return -as_scalar((error).t()*mat_r[triplet.second] * mat_r[triplet.second].t()*(error));
	}

	virtual double training_prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		vec error = embedding_entity[triplet.first.first]
			+ embedding_relation[triplet.second]
			- embedding_entity[triplet.first.second];

		return -sum(abs(error));
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];
		mat& mat_rel = mat_r[triplet.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (training_prob_triplets(triplet) - training_prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		head -= alpha * sign(head + relation - tail);
		tail += alpha * sign(head + relation - tail);
		relation -= alpha * sign(head + relation - tail);
		head_f += alpha * sign(head_f + relation_f - tail_f);
		tail_f -= alpha * sign(head_f + relation_f - tail_f);
		relation_f += alpha * sign(head_f + relation_f - tail_f);

		head = normalise(head);
		tail = normalise(tail);
		relation = normalise(relation);
		head_f = normalise(head_f);
		tail_f = normalise(tail_f);

		relation_reg(triplet.second, rand() % count_relation(), ESS_factor);
		entity_reg(triplet.first.first, triplet.first.second, ESS_factor);
		entity_reg(triplet.first.first, rand() % count_entity(), ESS_factor);
		entity_reg(triplet.first.second, rand() % count_entity(), ESS_factor);
	}

	virtual void train(bool last_time)
	{
		TransE::train(alpha);

		if (last_time || task_type == TripletClassification)
		{
			for_each(mat_r.begin(), mat_r.end(), [=](mat& elem){elem = eye(dim, dim); });
			//for(auto i=0; i<10; ++i)
			{
				for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
				{
					auto triplet = *i;

					vec& head = embedding_entity[triplet.first.first];
					vec& tail = embedding_entity[triplet.first.second];
					vec& relation = embedding_relation[triplet.second];

					pair<pair<int, int>, int> triplet_f;
					data_model.sample_false_triplet(triplet, triplet_f);

					vec& head_f = embedding_entity[triplet_f.first.first];
					vec& tail_f = embedding_entity[triplet_f.first.second];
					vec& relation_f = embedding_relation[triplet_f.second];

					mat_r[triplet.second] +=
						0.001 * (head + relation - tail) * (head + relation - tail).t();
					-0.001 * (head_f + relation_f - tail_f) * (head_f + relation_f - tail_f).t();
				}
				for_each(mat_r.begin(), mat_r.end(), [=](mat& elem){elem = elem.i(); });
			}
		}
	}

	virtual void report(const string& filename) const
	{
		if (task_type == TransA_ReportWeightes)
		{
			for (auto i = mat_r.begin(); i != mat_r.end(); ++i)
			{
				cout << data_model.relation_id_to_name[i - mat_r.begin()] << ":";

				vector<double> weights;
				double total = 0;
				mat mat_l, mat_u;
				lu(mat_l, mat_u, *i);
				for (auto i = 0; i<dim; ++i)
				{
					weights.push_back(mat_u(i, i));
					total += mat_u(i, i);
				}
				sort(weights.begin(), weights.end());
				cout << weights.back() << ",";
				cout << weights[dim / 2] << ",";
				cout << total;
				cout << endl;
			}
		}
	}
};

class TransA_ESS
	:public TransA
{
protected:
	const double ESS_factor;

public:
	TransA_ESS(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		double ESS_factor,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0)
		:TransA(dataset, task_type, logging_base_path, dim, alpha, training_threshold, is_preprocessed, worker_num, master_epoch, fd),
		ESS_factor(ESS_factor)
	{
		logging.record() << "\t[Name]\tTransA_ESS";
		logging.record() << "\t[ESS Factor]\t" << ESS_factor;
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		TransA::train_triplet(triplet);
		entity_reg(triplet.first.first, rand() % count_entity(), ESS_factor);
		entity_reg(triplet.first.second, rand() % count_entity(), ESS_factor);
	}
};

class TransM
	:public Model
{
protected:
	vector<vec>				embedding_entity;
	vector<vector<vec>>		embedding_clusters;
	vector<vec>				weights_clusters;

protected:
	const int				n_cluster;
	const double			alpha;
	const double			sparse_factor;
	const bool				single_or_total;
	const double			training_threshold;
	const int			dim;
	const bool				be_weight_normalized;

public:
	TransM(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		int n_cluster,
		double sparse_factor,
		bool sot = true,
		bool be_weight_normalized = false,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0)
		:Model(dataset, task_type, logging_base_path, is_preprocessed, worker_num, master_epoch, fd), dim(dim), alpha(alpha),
		training_threshold(training_threshold), n_cluster(n_cluster), sparse_factor(sparse_factor),
		single_or_total(sot), be_weight_normalized(be_weight_normalized)
	{
		logging.record() << "\t[Name]\tTransM";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Cluster Counts]\t" << n_cluster;
		logging.record() << "\t[Sparsity Factor]\t" << sparse_factor;

		if (be_weight_normalized)
			logging.record() << "\t[Weight Normalized]\tTrue";
		else
			logging.record() << "\t[Weight Normalized]\tFalse";

		if (sot)
			logging.record() << "\t[Single or Total]\tTrue";
		else
			logging.record() << "\t[Single or Total]\tFalse";

		embedding_entity.resize(count_entity());
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem){elem = randu(dim, 1); });

		embedding_clusters.resize(count_relation());
		for (auto &elem_vec : embedding_clusters)
		{
			elem_vec.resize(n_cluster);
			for_each(elem_vec.begin(), elem_vec.end(), [=](vec& elem){elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });
		}

		weights_clusters.resize(count_relation());
		for (auto & elem_vec : weights_clusters)
		{
			elem_vec.resize(n_cluster);
			for (auto & elem : elem_vec)
			{
				elem = 1.0 / n_cluster;
			}
		}
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		if (single_or_total == false)
			return training_prob_triplets(triplet);

		double	mixed_prob = 1e-8;
		for (int c = 0; c<n_cluster; ++c)
		{
			vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
				- embedding_entity[triplet.first.second];
			mixed_prob = max(mixed_prob, fabs(weights_clusters[triplet.second][c]) * exp(-sum(abs(error_c))));
		}

		return mixed_prob;
	}

	virtual double training_prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		double	mixed_prob = 1e-8;
		for (int c = 0; c<n_cluster; ++c)
		{
			vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
				- embedding_entity[triplet.first.second];
			mixed_prob += fabs(weights_clusters[triplet.second][c]) * exp(-sum(abs(error_c)));
		}

		return mixed_prob;
	}

	virtual void draw(const string& filename, const int radius, const int id_relation) const
	{
		mat	record(radius*6.0 + 10, radius*6.0 + 10);
		record.fill(255);
		for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
		{
			if (i->second == id_relation)
			{
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]),
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1])) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) + 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) + 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) + 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) - 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) - 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) + 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) - 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) - 1) = 0;
			}
		}

		string relation_name = data_model.relation_id_to_name[id_relation];
		record.save(filename + replace_all(relation_name, "/", "_") + ".ppm", pgm_binary);
	}

	virtual void train_cluster_once(
		const pair<pair<int, int>, int>& triplet,
		const pair<pair<int, int>, int>& triplet_f,
		int cluster, double prob_true, double prob_false, double factor)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_clusters[triplet.second][cluster];
		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_clusters[triplet_f.second][cluster];

		double prob_local_true = exp(-sum(abs(head + relation - tail)));
		double prob_local_false = exp(-sum(abs(head_f + relation_f - tail_f)));

		weights_clusters[triplet.second][cluster] +=
			alpha / prob_true * prob_local_true * sign(weights_clusters[triplet.second][cluster]);
		weights_clusters[triplet_f.second][cluster] -=
			alpha / prob_false * prob_local_false  * sign(weights_clusters[triplet_f.second][cluster]);

		weights_clusters[triplet.second][cluster] -=
			alpha * sign(weights_clusters[triplet.second][cluster]);

		head -= alpha * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		tail += alpha * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		relation -= alpha * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		head_f += alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false * fabs(weights_clusters[triplet.second][cluster]);
		tail_f -= alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false  * fabs(weights_clusters[triplet.second][cluster]);
		relation_f += alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false * fabs(weights_clusters[triplet.second][cluster]);

		if (norm_L2(relation) > 1.0)
			relation = normalise(relation);

		if (norm_L2(relation_f) > 1.0)
			relation_f = normalise(relation_f);
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		double prob_true = (training_prob_triplets(triplet));
		double prob_false = (training_prob_triplets(triplet_f));

		if (prob_true / prob_false > training_threshold)
			return;

		for (int c = 0; c<n_cluster; ++c)
		{
			train_cluster_once(triplet, triplet_f, c, prob_true, prob_false, alpha);
		}

		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];

		if (norm_L2(head) > 1.0)
			head = normalise(head);

		if (norm_L2(tail) > 1.0)
			tail = normalise(tail);

		if (norm_L2(head_f) > 1.0)
			head_f = normalise(head_f);

		if (norm_L2(tail_f) > 1.0)
			tail_f = normalise(tail_f);

		if (be_weight_normalized)
			weights_clusters[triplet.second] = normalise(weights_clusters[triplet.second]);
	}

public:
	virtual void report(const string& filename) const
	{
		if (task_type == TransM_ReportClusterNumber)
		{
			vector<int>	count_cluster(n_cluster);
			double		total_number = 0;
			for (auto i = weights_clusters.begin(); i != weights_clusters.end(); ++i)
			{
				int n = 0;
				for_each(i->begin(), i->end(), [&](double w) {if (fabs(w)>0.005) ++n; });
				cout << data_model.relation_id_to_name[i - weights_clusters.begin()] << ":" << n << endl;
				++count_cluster[n];
				total_number += n;
			}
			copy(count_cluster.begin(), count_cluster.end(), std::ostream_iterator<int>(cout, "\n"));
			cout << total_number / count_relation() << endl;
			cout << total_number << endl;
			return;
		}
		else if (task_type == TransM_ReportDetailedClusterLabel)
		{
			vector<bitset<32>>	counts_component(count_relation());
			ofstream fout(filename.c_str());
			for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
			{
				int pos_cluster = 0;
				double	mixed_prob = 1e-8;
				for (int c = 0; c<n_cluster; ++c)
				{
					vec error_c = embedding_entity[i->first.first]
						+ embedding_clusters[i->second][c]
						- embedding_entity[i->first.second];
					if (mixed_prob < exp(-sum(abs(error_c))))
					{
						pos_cluster = c;
						mixed_prob = exp(-sum(abs(error_c)));
					}
				}

				counts_component[i->second][pos_cluster] = 1;
				fout << data_model.entity_id_to_name[i->first.first] << '\t';
				fout << data_model.relation_id_to_name[i->second] << "==" << pos_cluster << '\t';
				fout << data_model.entity_id_to_name[i->first.second] << '\t';
				fout << endl;
			}
			fout.close();

			for (auto i = 0; i != counts_component.size(); ++i)
			{
				fout << data_model.relation_id_to_name[i] << ":";
				fout << counts_component[i].count() << endl;
			}
		}
	}
};

class TransE_SW
	:public TransE
{
protected:
	const double wake_factor;

public:
	TransE_SW(const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		double wake_factor,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0)
		:wake_factor(wake_factor),
		TransE(dataset, task_type, logging_base_path, dim, alpha, training_threshold, is_preprocessed, worker_num, master_epoch, fd)
	{
		logging.record() << "\t[Name]\tTransE SleepWake";
		logging.record() << "\t[Wake Factor]\t" << wake_factor;
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet) override
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_relation[triplet.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		if (prob_triplets(triplet) - prob_triplets(triplet_f) > training_threshold)
			return;

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_relation[triplet_f.second];

		vec grad = head + relation - tail;
		for_each(grad.begin(), grad.end(), [&](double& elem) {if (abs(elem) < wake_factor) elem = 0; });
		grad = sign(grad);

		head -= alpha * grad;
		tail += alpha * grad;
		relation -= alpha * grad;

		vec grad_f = head_f + relation_f - tail_f;
		for_each(grad_f.begin(), grad_f.end(), [&](double& elem) {if (abs(elem) < wake_factor) elem = 0; });
		grad_f = sign(grad_f);

		head_f += alpha * grad_f;
		tail_f -= alpha * grad_f;
		relation_f += alpha * grad_f;

		if (norm_L2(head) > 1.0)
			head = normalise(head);

		if (norm_L2(tail) > 1.0)
			tail = normalise(tail);

		if (norm_L2(relation) > 1.0)
			relation = normalise(relation);

		if (norm_L2(head_f) > 1.0)
			head_f = normalise(head_f);

		if (norm_L2(tail_f) > 1.0)
			tail_f = normalise(tail_f);
	}
};

class TransG
	:public Model
{
protected:
	vector<vec>				embedding_entity;
	vector<vector<vec>>		embedding_clusters;
	vector<vec>				weights_clusters;
	vector<int>		size_clusters;

protected:
	const int				n_cluster;
	const double			alpha;
	const bool				single_or_total;
	const double			training_threshold;
	const int			dim;
	const bool				be_weight_normalized;
	const int			step_before;
	const double			normalizor;

protected:
	double			CRP_factor;

public:
	TransG(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		int n_cluster,
		double CRP_factor,
		int step_before = 10,
		bool sot = false,
		bool be_weight_normalized = true,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0)
		:Model(dataset, task_type, logging_base_path, is_preprocessed, worker_num, master_epoch, fd), dim(dim), alpha(alpha),
		training_threshold(training_threshold), n_cluster(n_cluster), CRP_factor(CRP_factor),
		single_or_total(sot), be_weight_normalized(be_weight_normalized), step_before(step_before),
		normalizor(1.0 / pow(3.1415, dim / 2))
	{
		// logging.record() << "\t[Name]\tTransM";
		// logging.record() << "\t[Dimension]\t" << dim;
		// logging.record() << "\t[Learning Rate]\t" << alpha;
		// logging.record() << "\t[Training Threshold]\t" << training_threshold;
		// logging.record() << "\t[Cluster Counts]\t" << n_cluster;
		// logging.record() << "\t[CRP Factor]\t" << CRP_factor;
		//
		// if (be_weight_normalized)
		// 	logging.record() << "\t[Weight Normalized]\tTrue";
		// else
		// 	logging.record() << "\t[Weight Normalized]\tFalse";
		//
		// if (sot)
		// 	logging.record() << "\t[Single or Total]\tTrue";
		// else
		// 	logging.record() << "\t[Single or Total]\tFalse";

		embedding_entity.resize(count_entity());
		embedding_clusters.resize(count_relation());
		weights_clusters.resize(count_relation());
		size_clusters.resize(count_relation(), n_cluster);

		if (is_preprocessed) {
			for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem) {elem = vec(dim); });
			for (auto &elem_vec : embedding_clusters) {
				elem_vec.resize(30);
				for_each(elem_vec.begin(), elem_vec.end(), [=](vec& elem) {elem = vec(dim); });
			}
			for (auto & elem_vec : weights_clusters)
			{
				elem_vec.resize(21);
				elem_vec.fill(0.0);
			}

			// load 함수 부분이 소켓과 연동되어야 함, load 함수 자체에서 처리
			load("", fs_log);
		} else {
			for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem){elem = randu(dim, 1); });

			embedding_clusters.resize(count_relation());
			for (auto &elem_vec : embedding_clusters)
			{
				elem_vec.resize(30);
				for_each(elem_vec.begin(), elem_vec.end(), [=](vec& elem){elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });
			}

			weights_clusters.resize(count_relation());
			for (auto & elem_vec : weights_clusters)
			{
				elem_vec.resize(21);
				elem_vec.fill(0.0);
				for (auto i = 0; i<n_cluster; ++i)
				{
					elem_vec[i] = 1.0 / n_cluster;
				}
			}

			size_clusters.resize(count_relation(), n_cluster);
			this->CRP_factor = CRP_factor / data_model.data_train.size() * count_relation();
		}
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		if (single_or_total == false)
			return training_prob_triplets(triplet);

		double	mixed_prob = 1e-100;
		for (int c = 0; c<size_clusters[triplet.second]; ++c)
		{
			vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
				- embedding_entity[triplet.first.second];
			mixed_prob = max(mixed_prob, fabs(weights_clusters[triplet.second][c])
				* exp(-sum(abs(error_c))));
		}

		return mixed_prob;
	}

	virtual double training_prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		double	mixed_prob = 1e-100;
		for (int c = 0; c<size_clusters[triplet.second]; ++c)
		{
			vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
				- embedding_entity[triplet.first.second];
			mixed_prob += fabs(weights_clusters[triplet.second][c]) * exp(-sum(abs(error_c)));
		}

		return mixed_prob;
	}

	virtual void draw(const string& filename, const int radius, const int id_relation) const
	{
		mat	record(radius*6.0 + 10, radius*6.0 + 10);
		record.fill(255);
		for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
		{
			if (i->second == id_relation)
			{
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]),
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1])) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) + 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) + 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) + 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) - 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) - 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) + 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) - 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) - 1) = 0;
			}
		}

		string relation_name = data_model.relation_id_to_name[id_relation];
		record.save(filename + replace_all(relation_name, "/", "_") + ".ppm", pgm_binary);
	}

	virtual void save(const string& filename, FILE * fs_log) override
	{
		ofstream fout(filename, ios::binary);
		storage_vmat<double>::save(embedding_entity, fout);
		storage_vmat<double>::save(weights_clusters, fout);
		fout.close();
	}

	virtual void load(const string& filename, FILE * fs_log) override
	{
		ifstream fin(filename, ios::binary);
		storage_vmat<double>::load(embedding_entity, fin);
		storage_vmat<double>::load(weights_clusters, fin);
		fin.close();
	}

	virtual void train_cluster_once(
		const pair<pair<int, int>, int>& triplet,
		const pair<pair<int, int>, int>& triplet_f,
		int cluster, double prob_true, double prob_false, double factor)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_clusters[triplet.second][cluster];
		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_clusters[triplet_f.second][cluster];

		double prob_local_true = exp(-sum(abs(head + relation - tail)));
		double prob_local_false = exp(-sum(abs(head_f + relation_f - tail_f)));

		weights_clusters[triplet.second][cluster] +=
			factor / prob_true * prob_local_true * sign(weights_clusters[triplet.second][cluster]);
		weights_clusters[triplet_f.second][cluster] -=
			factor / prob_false * prob_local_false  * sign(weights_clusters[triplet_f.second][cluster]);

		head -= factor * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		tail += factor * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		relation -= factor * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]);
		head_f += factor * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false * fabs(weights_clusters[triplet.second][cluster]);
		tail_f -= factor * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false  * fabs(weights_clusters[triplet.second][cluster]);
		relation_f += factor * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false * fabs(weights_clusters[triplet.second][cluster]);

		if (norm_L2(relation) > 1.0)
			relation = normalise(relation);

		if (norm_L2(relation_f) > 1.0)
			relation_f = normalise(relation_f);
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];

		if (!head.is_finite())
			cout << "d";

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		double prob_true = training_prob_triplets(triplet);
		double prob_false = training_prob_triplets(triplet_f);

		if (prob_true / prob_false > exp(training_threshold))
			return;

		for (int c = 0; c<size_clusters[triplet.second]; ++c)
		{
			train_cluster_once(triplet, triplet_f, c, prob_true, prob_false, alpha);
		}

		double prob_new_component = CRP_factor * exp(-sum(abs(head - tail)));

		if (randu() < prob_new_component / (prob_new_component + prob_true)
			&& size_clusters[triplet.second] < 20
			&& epos >= step_before)
		{
#pragma omp critical
			{
				//cout<<"A";
				weights_clusters[triplet.second][size_clusters[triplet.second]] = CRP_factor;
				embedding_clusters[triplet.second][size_clusters[triplet.second]] = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); //tail - head;
				++size_clusters[triplet.second];
			}
		}

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];

		if (norm_L2(head) > 1.0)
			head = normalise(head);

		if (norm_L2(tail) > 1.0)
			tail = normalise(tail);

		if (norm_L2(head_f) > 1.0)
			head_f = normalise(head_f);

		if (norm_L2(tail_f) > 1.0)
			tail_f = normalise(tail_f);

		if (be_weight_normalized)
			weights_clusters[triplet.second] = normalise(weights_clusters[triplet.second]);
	}

public:
	virtual void report(const string& filename) const
	{
		if (task_type == TransM_ReportClusterNumber)
		{
			for (auto i = 0; i<count_relation(); ++i)
			{
				cout << data_model.relation_id_to_name[i] << ':';
				cout << size_clusters[i] << endl;
			}
			return;
		}
		else if (task_type == TransM_ReportDetailedClusterLabel)
		{
			vector<bitset<32>>	counts_component(count_relation());
			ofstream fout(filename.c_str());
			for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
			{
				int pos_cluster = 0;
				double	mixed_prob = 1e-8;
				for (int c = 0; c<n_cluster; ++c)
				{
					vec error_c = embedding_entity[i->first.first]
						+ embedding_clusters[i->second][c]
						- embedding_entity[i->first.second];
					if (mixed_prob < exp(-sum(abs(error_c))))
					{
						pos_cluster = c;
						mixed_prob = exp(-sum(abs(error_c)));
					}
				}

				counts_component[i->second][pos_cluster] = 1;
				fout << data_model.entity_id_to_name[i->first.first] << '\t';
				fout << data_model.relation_id_to_name[i->second] << "==" << pos_cluster << '\t';
				fout << data_model.entity_id_to_name[i->first.second] << '\t';
				fout << endl;
			}
			fout.close();

			for (auto i = 0; i != counts_component.size(); ++i)
			{
				fout << data_model.relation_id_to_name[i] << ":";
				fout << counts_component[i].count() << endl;
			}
		}
	}
};

class TransG_Hiracherical
	:public Model
{
protected:
	vector<vec>				embedding_entity;
	vector<vector<vec>>		embedding_clusters;
	vector<vec>				weights_clusters;
	vector<int>				size_clusters;
	vec						variance;

protected:
	const int				n_cluster;
	const double			alpha;
	const bool				single_or_total;
	const double			training_threshold;
	const int				dim;
	const bool				be_weight_normalized;
	const int				step_before;
	const double			normalizor;
	const double			variance_bound;

protected:
	double					CRP_factor;

public:
	TransG_Hiracherical(
		const Dataset& dataset,
		const TaskType& task_type,
		const string& logging_base_path,
		int dim,
		double alpha,
		double training_threshold,
		int n_cluster,
		double CRP_factor,
		int step_before = 10,
		double variance_bound = 0.01,
		bool sot = false,
		bool be_weight_normalized = true,
		const bool is_preprocessed = false,
		const int worker_num = 0,
		const int master_epoch = 0,
		const int fd = 0)
		:Model(dataset, task_type, logging_base_path, is_preprocessed, worker_num, master_epoch, fd), dim(dim), alpha(alpha),
		training_threshold(training_threshold), n_cluster(n_cluster), CRP_factor(CRP_factor),
		single_or_total(sot), be_weight_normalized(be_weight_normalized), step_before(step_before),
		normalizor(1.0 / pow(3.1415, dim / 2)), variance_bound(variance_bound)
	{
		logging.record() << "\t[Name]\tTransM";
		logging.record() << "\t[Dimension]\t" << dim;
		logging.record() << "\t[Learning Rate]\t" << alpha;
		logging.record() << "\t[Training Threshold]\t" << training_threshold;
		logging.record() << "\t[Cluster Counts]\t" << n_cluster;
		logging.record() << "\t[CRP Factor]\t" << CRP_factor;

		if (be_weight_normalized)
			logging.record() << "\t[Weight Normalized]\tTrue";
		else
			logging.record() << "\t[Weight Normalized]\tFalse";

		if (sot)
			logging.record() << "\t[Single or Total]\tTrue";
		else
			logging.record() << "\t[Single or Total]\tFalse";

		embedding_entity.resize(count_entity());
		for_each(embedding_entity.begin(), embedding_entity.end(), [=](vec& elem){elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });

		embedding_clusters.resize(count_relation());
		for (auto &elem_vec : embedding_clusters)
		{
			elem_vec.resize(30);
			for_each(elem_vec.begin(), elem_vec.end(), [=](vec& elem){elem = (2 * randu(dim, 1) - 1)*sqrt(6.0 / dim); });
		}

		weights_clusters.resize(count_relation());
		for (auto & elem_vec : weights_clusters)
		{
			elem_vec.resize(30);
			elem_vec.fill(0.0);
			for (auto i = 0; i<n_cluster; ++i)
			{
				elem_vec[i] = 1.0 / n_cluster;
			}
		}

		size_clusters.resize(count_relation(), n_cluster);
		this->CRP_factor = CRP_factor / data_model.data_train.size() * count_relation();

		variance.resize(count_entity());
		variance.fill(0.0);
	}

public:
	virtual double prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		if (single_or_total == false)
			return training_prob_triplets(triplet);

		double total_variance = variance[triplet.first.first] * variance[triplet.first.first]
			+ variance[triplet.first.second] * variance[triplet.first.second] + 1;

		double	mixed_prob = 1e-100;
		for (int c = 0; c<size_clusters[triplet.second]; ++c)
		{
			vec error_c = (embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
				- embedding_entity[triplet.first.second]);
			mixed_prob = max(mixed_prob, fabs(weights_clusters[triplet.second][c])
				* exp(-sum(abs(error_c)) / total_variance));
		}

		return mixed_prob;
	}

	virtual double training_prob_triplets(const pair<pair<int, int>, int>& triplet)
	{
		double total_variance = variance[triplet.first.first] * variance[triplet.first.first]
			+ variance[triplet.first.second] * variance[triplet.first.second] + 1;

		double	mixed_prob = 1e-100;
		for (int c = 0; c<size_clusters[triplet.second]; ++c)
		{
			vec error_c = embedding_entity[triplet.first.first] + embedding_clusters[triplet.second][c]
				- embedding_entity[triplet.first.second];
			mixed_prob += fabs(weights_clusters[triplet.second][c])
				* exp(-sum(abs(error_c)) / total_variance);
		}

		return mixed_prob;
	}

	virtual void draw(const string& filename, const int radius, const int id_relation) const
	{
		mat	record(radius*6.0 + 10, radius*6.0 + 10);
		record.fill(255);
		for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
		{
			if (i->second == id_relation)
			{
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]),
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1])) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) + 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) + 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) + 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) - 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) - 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) + 1) = 0;
				record(radius * (3.0 + embedding_entity[i->first.second][0] - embedding_entity[i->first.first][0]) - 1,
					radius *(3.0 + embedding_entity[i->first.second][1] - embedding_entity[i->first.first][1]) - 1) = 0;
			}
		}

		string relation_name = data_model.relation_id_to_name[id_relation];
		record.save(filename + replace_all(relation_name, "/", "_") + ".ppm", pgm_binary);
	}

	virtual void train_cluster_once(
		const pair<pair<int, int>, int>& triplet,
		const pair<pair<int, int>, int>& triplet_f,
		int cluster, double prob_true, double prob_false, double factor)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];
		vec& relation = embedding_clusters[triplet.second][cluster];
		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];
		vec& relation_f = embedding_clusters[triplet_f.second][cluster];

		double total_variance = variance[triplet.first.first] * variance[triplet.first.first]
			+ variance[triplet.first.second] * variance[triplet.first.second] + 1;
		double total_variance_f = variance[triplet_f.first.first] * variance[triplet_f.first.first]
			+ variance[triplet_f.first.second] * variance[triplet_f.first.second] + 1;
		double prob_local_true = exp(-sum(abs(head + relation - tail)) / total_variance);
		double prob_local_false = exp(-sum(abs(head_f + relation_f - tail_f)) / total_variance_f);

		const double thres = -variance_bound;

		variance[triplet.first.first] += alpha * 2 * fabs(weights_clusters[triplet.second][cluster])
			* prob_local_true / prob_true * sum(abs(head + relation - tail))
			/ total_variance / total_variance * variance[triplet.first.first];
		variance[triplet.first.first] = max(-thres, min(thres, variance[triplet.first.first]));

		variance[triplet.first.second] += alpha * 2 * fabs(weights_clusters[triplet.second][cluster])
			* prob_local_true / prob_true * sum(abs(head + relation - tail))
			/ total_variance / total_variance * variance[triplet.first.second];
		variance[triplet.first.second] = max(thres, min(thres, variance[triplet.first.second]));

		variance[triplet_f.first.first] -= alpha * 2 * fabs(weights_clusters[triplet_f.second][cluster])
			* prob_local_false / prob_false * sum(abs(head_f + relation_f - tail_f))
			/ total_variance_f / total_variance_f * variance[triplet_f.first.first];
		variance[triplet_f.first.first] = max(thres, min(thres, variance[triplet_f.first.first]));

		variance[triplet_f.first.second] -= alpha * 2 * fabs(weights_clusters[triplet_f.second][cluster])
			* prob_local_false / prob_false * sum(abs(head_f + relation_f - tail_f))
			/ total_variance_f / total_variance_f * variance[triplet_f.first.second];
		variance[triplet_f.first.second] = max(thres, min(thres, variance[triplet_f.first.second]));

		weights_clusters[triplet.second][cluster] +=
			alpha / prob_true * prob_local_true * sign(weights_clusters[triplet.second][cluster]);
		weights_clusters[triplet_f.second][cluster] -=
			alpha / prob_false * prob_local_false  * sign(weights_clusters[triplet_f.second][cluster]);

		head -= alpha * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]) / total_variance;
		tail += alpha * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]) / total_variance;
		relation -= alpha * sign(head + relation - tail)
			* prob_local_true / prob_true * fabs(weights_clusters[triplet.second][cluster]) / total_variance;
		head_f += alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false * fabs(weights_clusters[triplet.second][cluster]) / total_variance_f;
		tail_f -= alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false  * fabs(weights_clusters[triplet.second][cluster]) / total_variance_f;
		relation_f += alpha * sign(head_f + relation_f - tail_f)
			* prob_local_false / prob_false * fabs(weights_clusters[triplet.second][cluster]) / total_variance_f;

		if (norm_L2(relation) > 1.0)
			relation = normalise(relation);

		if (norm_L2(relation_f) > 1.0)
			relation_f = normalise(relation_f);
	}

	virtual void train_triplet(const pair<pair<int, int>, int>& triplet)
	{
		vec& head = embedding_entity[triplet.first.first];
		vec& tail = embedding_entity[triplet.first.second];

		pair<pair<int, int>, int> triplet_f;
		data_model.sample_false_triplet(triplet, triplet_f);

		double prob_true = training_prob_triplets(triplet);
		double prob_false = training_prob_triplets(triplet_f);

		if (prob_true / prob_false > exp(training_threshold))
			return;

		for (int c = 0; c<size_clusters[triplet.second]; ++c)
		{
			train_cluster_once(triplet, triplet_f, c, prob_true, prob_false, alpha);
		}

		double prob_new_component = CRP_factor * exp(-sum(abs(head - tail)))
			/ (variance[triplet.first.first] + variance[triplet.first.second] + 1);

		if (randu() < prob_new_component / (prob_new_component + prob_true)
			&& size_clusters[triplet.second] <= 20
			&& epos >= step_before)
		{
#pragma omp critical
			{
				//cout<<"A";
				weights_clusters[triplet.second][size_clusters[triplet.second]] = CRP_factor;
				embedding_clusters[triplet.second][size_clusters[triplet.second]] = tail - head;//(2*randu(dim,1)-1)*sqrt(6.0/dim);
				++size_clusters[triplet.second];
			}
		}

		vec& head_f = embedding_entity[triplet_f.first.first];
		vec& tail_f = embedding_entity[triplet_f.first.second];

		if (norm_L2(head) > 1.0)
			head = normalise(head);

		if (norm_L2(tail) > 1.0)
			tail = normalise(tail);

		if (norm_L2(head_f) > 1.0)
			head_f = normalise(head_f);

		if (norm_L2(tail_f) > 1.0)
			tail_f = normalise(tail_f);

		if (be_weight_normalized)
			weights_clusters[triplet.second] = normalise(weights_clusters[triplet.second]);
	}

public:
	virtual void report(const string& filename) const
	{
		if (task_type == TransM_ReportClusterNumber)
		{
			for (auto i = 0; i<count_relation(); ++i)
			{
				cout << data_model.relation_id_to_name[i] << ':';
				cout << size_clusters[i] << endl;
			}
			return;
		}
		else if (task_type == TransM_ReportDetailedClusterLabel)
		{
			vector<bitset<32>>	counts_component(count_relation());
			ofstream fout(filename.c_str());
			for (auto i = data_model.data_train.begin(); i != data_model.data_train.end(); ++i)
			{
				int pos_cluster = 0;
				double	mixed_prob = 1e-8;
				for (int c = 0; c<n_cluster; ++c)
				{
					vec error_c = embedding_entity[i->first.first]
						+ embedding_clusters[i->second][c]
						- embedding_entity[i->first.second];
					if (mixed_prob < exp(-sum(abs(error_c))))
					{
						pos_cluster = c;
						mixed_prob = exp(-sum(abs(error_c)));
					}
				}

				counts_component[i->second][pos_cluster] = 1;
				fout << data_model.entity_id_to_name[i->first.first] << '\t';
				fout << data_model.relation_id_to_name[i->second] << "==" << pos_cluster << '\t';
				fout << data_model.entity_id_to_name[i->first.second] << '\t';
				fout << endl;
			}
			fout.close();

			for (auto i = 0; i != counts_component.size(); ++i)
			{
				fout << data_model.relation_id_to_name[i] << ":";
				fout << counts_component[i].count() << endl;
			}
		}
	}
};
