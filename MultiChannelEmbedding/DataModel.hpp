#pragma once
#include "Import.hpp"
#include "ModelConfig.hpp"
//#include <boost/archive/xml_oarchive.hpp>
//#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/set.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <sstream>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

class DataModel
{
public:
    set<pair<pair<int, int>, int> >     check_data_train;
    set<pair<pair<int, int>, int> >     check_data_all;

public:
    vector<pair<pair<int, int>, int> >  data_train;
    vector<pair<pair<int, int>, int> >  data_train_parts;
    vector<pair<pair<int, int>, int> >  data_dev_true;
    vector<pair<pair<int, int>, int> >  data_dev_false;
    vector<pair<pair<int, int>, int> >  data_test_true;
    vector<pair<pair<int, int>, int> >  data_test_false;

public:
    set<int>            set_tail;
    set<int>            set_head;
    set<int>            set_entity_parts;
    vector<int>         vector_entity_parts;
    set<int>            set_relation_parts;
    vector<int>         vector_relation_parts;
    set<string>         set_entity;
    set<string>         set_relation;

public:
    vector<set<int>>    set_relation_tail;
    vector<set<int>>    set_relation_head;

public:
    vector<int> relation_type;

public:
    vector<string>      entity_id_to_name;
    vector<string>      relation_id_to_name;
    map<string, int>    entity_name_to_id;
    map<string, int>    relation_name_to_id;

public:
    vector<double>      prob_head;
    vector<double>      prob_tail;
    vector<double>      relation_tph;
    vector<double>      relation_hpt;
    map<string, int>    count_entity;
    map<int, bool>      check_anchor;
    map<int, bool> 	check_parts;

public:
    map<int, map<int, int> >    tails;
    map<int, map<int, int> >    heads;

public:
    map<int, map<int, vector<int> > >     rel_heads;
    map<int, map<int, vector<int> > >     rel_tails;
    map<pair<int, int>, int>             rel_finder;

public:
    int zeroshot_pointer;

public:
    DataModel(const Dataset& dataset, const bool is_preprocessed, const int worker_num, const int master_epoch, const int fd) {
    	if (is_preprocessed)
    	{
    		ifstream input("../tmp/data_model.bin", ios_base::binary);
        boost::archive::binary_iarchive ia(input);
        ia >> entity_name_to_id;
        ia >> entity_id_to_name;
        ia >> relation_id_to_name;
        ia >> relation_name_to_id;
        ia >> data_train;
        ia >> data_dev_true;
        ia >> data_dev_false;
        ia >> data_test_true;
        ia >> data_test_false;
        ia >> check_data_train;
        ia >> check_data_all;
        ia >> set_entity;
        ia >> set_relation;
        ia >> count_entity;
        ia >> rel_heads;
        ia >> rel_tails;
        ia >> rel_finder;
        ia >> relation_tph;
        ia >> relation_hpt;
        ia >> set_relation_head;
        ia >> set_relation_tail;
        ia >> prob_head;
        ia >> prob_tail;
        ia >> tails;
        ia >> heads;
        ia >> relation_type;
	      input.close();
    	}
    	else
    	{
    		ofstream output("../tmp/data_model.bin", ios::binary);
	        boost::archive::binary_oarchive oa(output);

	        load_training(dataset.base_dir + dataset.training);

	        relation_hpt.resize(set_relation.size());
	        relation_tph.resize(set_relation.size());
	        for (auto i = 0; i != set_relation.size(); ++i)
	        {
	            double sum = 0;
	            double total = 0;
	            for (auto ds = rel_heads[i].begin(); ds != rel_heads[i].end(); ++ds)
	            {
	                ++sum;
	                total += ds->second.size();
	            }
	            relation_tph[i] = total / sum;
	        }
	        for (auto i = 0; i != set_relation.size(); ++i)
	        {
	            double sum = 0;
	            double total = 0;
	            for (auto ds = rel_tails[i].begin(); ds != rel_tails[i].end(); ++ds)
	            {
	                ++sum;
	                total += ds->second.size();
	            }
	            relation_hpt[i] = total / sum;
	        }

	        zeroshot_pointer = set_entity.size();
	        load_testing(dataset.base_dir + dataset.developing, data_dev_true, data_dev_false, dataset.self_false_sampling);
	        load_testing(dataset.base_dir + dataset.testing, data_test_true, data_test_false, dataset.self_false_sampling);


	        set_relation_head.resize(set_entity.size());
	        set_relation_tail.resize(set_relation.size());
	        prob_head.resize(set_entity.size());
	        prob_tail.resize(set_entity.size());
	        for (auto i = data_train.begin(); i != data_train.end(); ++i)
	        {
	            ++prob_head[i->first.first];
	            ++prob_tail[i->first.second];

	            ++tails[i->second][i->first.first];
	            ++heads[i->second][i->first.second];

	            set_relation_head[i->second].insert(i->first.first);
	            set_relation_tail[i->second].insert(i->first.second);
        	}

#pragma omp parallel for
#pragma ivdep
	        for (auto elem = prob_head.begin(); elem != prob_head.end(); ++elem)
	        {
	            *elem /= data_train.size();
	        }

#pragma omp parallel for
#pragma ivdep
	        for (auto elem = prob_tail.begin(); elem != prob_tail.end(); ++elem)
	        {
	            *elem /= data_train.size();
	        }

	        double threshold = 1.5;
	        relation_type.resize(set_relation.size());

	        for (auto i = 0; i<set_relation.size(); ++i)
	        {
	            if (relation_tph[i]<threshold && relation_hpt[i]<threshold)
	            {
	                relation_type[i] = 1;
	            }
	            else if (relation_hpt[i] <threshold && relation_tph[i] >= threshold)
	            {
	                relation_type[i] = 2;
	            }
	            else if (relation_hpt[i] >= threshold && relation_tph[i] < threshold)
	            {
	                relation_type[i] = 3;
	            }
	            else
	            {
	                relation_type[i] = 4;
	            }
	        }

                 oa << entity_name_to_id;
                 oa << entity_id_to_name;
                 oa << relation_id_to_name;
                 oa << relation_name_to_id;
                 oa << data_train;
                 oa << data_dev_true;
                 oa << data_dev_false;
                 oa << data_test_true;
	         oa << data_test_false;
                 oa << check_data_train;
                 oa << check_data_all;
                 oa << set_entity;
                 oa << set_relation;
                 oa << count_entity;
                 oa << rel_heads;
                 oa << rel_tails;
                 oa << rel_finder;
                 oa << relation_tph;
                 oa << relation_hpt;
                 oa << set_relation_head;
                 oa << set_relation_tail;
                 oa << prob_head;
                 oa << prob_tail;
                 oa << tails;
                 oa << heads;
		 oa << relation_type;


	        output.close();
    	}

        if (master_epoch % 2 == 0) {
            
            // entity
            if (fd == 0) {

                // 파일로 가져옴
                ifstream input("../tmp/Embedding/maxmin_worker_"+ to_string(worker_num) + ".txt");
                string str;
                vector<string> anchor;

                getline(input, str);
                anchor = split(str, ' ');

                for (string e : anchor)
                {
                    set_entity_parts.insert(stoi(e));
                    check_anchor[stoi(e)] = true;
                    check_parts[stoi(e)] = true;
                }

                while (!input.eof())
                {
                    input >> str;
                    set_entity_parts.insert(stoi(str));
                    check_parts[stoi(str)] = true;
                }

                for (auto i = data_train.begin(); i != data_train.end(); ++i) {
                
                    int head = (*i).first.first;
                    int tail = (*i).first.second;
                    
                    if (check_parts.find(head) != check_parts.end() && check_parts.find(tail) != check_parts.end()){
                    
                        data_train_parts.push_back(*i);
                    }
                }         
            }
            else {

                // 소켓으로 가져옴
                int anchor_num;
                int entity_num;
                int temp_value;
                int success = 0;
                int flag = 0;

                // try catch 에서 exception 발생시 이 구간을 반복
                while(!success){

                    try{

                        if (recv(fd, &anchor_num, sizeof(anchor_num), 0) < 0){

                            printf("[error] DataModel.hpp > recv anchor_num\n");
                            close(fd);
                            return;
                        }

                        for (int idx = 0; idx < ntohl(anchor_num); idx++) {

                            if (recv(fd, &temp_value, sizeof(temp_value), 0) < 0){

                                printf("[error] DataModel.hpp > recv temp_value for anchor\n");
                                close(fd);
                                return;
                            }

                            temp_value = ntohl(temp_value);
                            set_entity_parts.insert(temp_value);
                            check_anchor[temp_value] = true;
                            check_parts[temp_value] = true;
                        }

                        if (recv(fd, &entity_num, sizeof(entity_num), 0) < 0){

                            printf("[error] DataModel.hpp > recv entity_num\n");
                            close(fd);
                            return;
                        }

                        for (int idx = 0; idx < ntohl(entity_num); idx++) {

                            if (recv(fd, &temp_value, sizeof(temp_value), 0) < 0){

                                printf("[error] DataModel.hpp >  recv temp_value for entity\n");
                                close(fd);
                                return;
                            }

                            temp_value = ntohl(temp_value);
                            set_entity_parts.insert(temp_value);
                            check_parts[temp_value] = true;
                        }

                        for (auto i = data_train.begin(); i != data_train.end(); ++i) {
                        
                            int head = (*i).first.first;
                            int tail = (*i).first.second;
                            
                            if (check_parts.find(head) != check_parts.end() && check_parts.find(tail) != check_parts.end()){
                            
                                data_train_parts.push_back(*i);
                            }
                        }   

                        flag = 1234;
                        flag = htonl(flag);
                        send(fd, &flag, sizeof(flag), 0);
                        success = 1;
                    }
                    catch(std::exception& e){

                        printf("[error] DataModel.hpp >  entity : exception occured\n");
                        printf("%s\n", e.what());
                        success = 0;
                        flag = 9876;
                        flag = htonl(flag);
                        send(fd, &flag, sizeof(flag), 0);
                    }
                }
            }

            cout << "[info] DataModel.hpp > entity preprocesing let's get it!" << endl;   
        }
        else {
            //relation
            if (fd == 0) {

                // 파일로 가져옴
                ifstream input("../tmp/sub_graph_worker_"+ to_string(worker_num) + ".txt");
                string str;
                pair<pair<int,int>, int> tmp;

                while (!input.eof())
                {
                    string head, tail, relation;
                    input >> head >> relation >> tail;
                    if (head == "" && relation == "" && tail == "") break;

                    set_entity_parts.insert(stoi(head));
                    set_entity_parts.insert(stoi(tail));
                    set_relation_parts.insert(stoi(relation));
                    tmp.first.first = stoi(head);
                    tmp.second = stoi(relation);
                    tmp.first.second = stoi(tail);
                    data_train_parts.push_back(tmp);
                }
            }
            else {

                // 소켓으로 가져옴
                int triplet_num = 0;
                int temp_value_head;
                int temp_value_relation;
                int temp_value_tail;
                int success = 0;
                int flag = 0;

                while(!success){

                    pair<pair<int,int>, int> tmp;

                    try{

                        if (recv(fd, &triplet_num, sizeof(triplet_num), 0) < 0){

                            cout << "[error] DataModel.hpp > recv triplet_num" << endl;
                            close(fd);
                            return;
                        }

                        for (int idx = 0; idx < ntohl(triplet_num); idx++) {

                            if (recv(fd, &temp_value_head, sizeof(temp_value_head), 0) < 0){

                                cout << "[error] DataModel.hpp > recv temp_value_head" << endl;
                                close(fd);
                                break;
                            }

                            if (recv(fd, &temp_value_relation, sizeof(temp_value_relation), 0) < 0){

                                cout <<"[error] DataModel.hpp > recv temp_value_relation" << endl;
                                close(fd);
                                break;
                            }

                            if (recv(fd, &temp_value_tail, sizeof(temp_value_tail), 0) < 0){

                                cout << "[error] DataModel.hpp > recv temp_value_tail" << endl;
                                close(fd);
                                break;
                            }

                            temp_value_head = ntohl(temp_value_head);
                            temp_value_relation = ntohl(temp_value_relation);
                            temp_value_tail = ntohl(temp_value_tail);

                            set_entity_parts.insert(temp_value_head);
                            set_entity_parts.insert(temp_value_tail);
                            set_relation_parts.insert(temp_value_relation);
                            tmp.first.first = temp_value_head;
                            tmp.second = temp_value_relation;
                            tmp.first.second = temp_value_tail;
                            data_train_parts.push_back(tmp);
                        }

                        flag = 1234;
                        flag = htonl(flag);
                        send(fd, &flag, sizeof(flag), 0);
                        success = 1;
                    }
                    catch(std::exception& e){

                        printf("[error] DataModel.hpp > relation : exception occured\n");
                        printf("%s\n", e.what());
                        success = 0;
                        flag = 9876;
                        flag = htonl(flag);
                        send(fd, &flag, sizeof(flag), 0);
                    }
                }

            }
            cout << "[info] DataModel.hpp > relation preprocessing let's get it!" << endl;
        }
        vector_entity_parts.assign(set_entity_parts.begin(), set_entity_parts.end());
        vector_relation_parts.assign(set_relation_parts.begin(), set_relation_parts.end());
        cout << "[info] DataModel.hpp > # of triples in worker" << worker_num << ": " << data_train_parts.size() << "/" << data_train.size() << endl;
	cout << "[info] DataModel.hpp > # of test triples: " << data_test_true.size() << endl;
    }

    DataModel(const Dataset& dataset, const string& file_zero_shot, const bool is_preprocessed, const int worker_num, const int master_epoch, const int fd)
    {
        load_training(dataset.base_dir + dataset.training);

        relation_hpt.resize(set_relation.size());
        relation_tph.resize(set_relation.size());
        for (auto i = 0; i != set_relation.size(); ++i)
        {
            double sum = 0;
            double total = 0;
            for (auto ds = rel_heads[i].begin(); ds != rel_heads[i].end(); ++ds)
            {
                ++sum;
                total += ds->second.size();
            }
            relation_tph[i] = total / sum;
        }
        for (auto i = 0; i != set_relation.size(); ++i)
        {
            double sum = 0;
            double total = 0;
            for (auto ds = rel_tails[i].begin(); ds != rel_tails[i].end(); ++ds)
            {
                ++sum;
                total += ds->second.size();
            }
            relation_hpt[i] = total / sum;
        }

        zeroshot_pointer = set_entity.size();
        load_testing(dataset.base_dir + dataset.developing, data_dev_true, data_dev_false, dataset.self_false_sampling);
        load_testing(dataset.base_dir + dataset.testing, data_dev_true, data_dev_false, dataset.self_false_sampling);
        load_testing(file_zero_shot, data_test_true, data_test_false, dataset.self_false_sampling);

        set_relation_head.resize(set_entity.size());
        set_relation_tail.resize(set_relation.size());
        prob_head.resize(set_entity.size());
        prob_tail.resize(set_entity.size());
        for (auto i = data_train.begin(); i != data_train.end(); ++i)
        {
            ++prob_head[i->first.first];
            ++prob_tail[i->first.second];

            ++tails[i->second][i->first.first];
            ++heads[i->second][i->first.second];

            set_relation_head[i->second].insert(i->first.first);
            set_relation_tail[i->second].insert(i->first.second);
        }

        for (auto & elem : prob_head)
        {
            elem /= data_train.size();
        }

        for (auto & elem : prob_tail)
        {
            elem /= data_train.size();
        }

        double threshold = 1.5;
        relation_type.resize(set_relation.size());
        for (auto i = 0; i < set_relation.size(); ++i)
        {
            if (relation_tph[i] < threshold && relation_hpt[i] < threshold)
            {
                relation_type[i] = 1;
            }
            else if (relation_hpt[i] < threshold && relation_tph[i] >= threshold)
            {
                relation_type[i] = 2;
            }
            else if (relation_hpt[i] >= threshold && relation_tph[i] < threshold)
            {
                relation_type[i] = 3;
            }
            else
            {
                relation_type[i] = 4;
            }
        }
    }


    void load_training(const string& filename)
    {
        fstream fin(filename.c_str());

        while (!fin.eof())
        {
            string head, tail, relation;
            fin >> head >> relation >> tail;

            if (head == "" && relation == "" && tail == "") break;

            if (entity_name_to_id.find(head) == entity_name_to_id.end())
            {
                entity_name_to_id.insert(make_pair(head, entity_name_to_id.size()));
                entity_id_to_name.push_back(head);
            }

            if (entity_name_to_id.find(tail) == entity_name_to_id.end())
            {
                entity_name_to_id.insert(make_pair(tail, entity_name_to_id.size()));
                entity_id_to_name.push_back(tail);
            }

            if (relation_name_to_id.find(relation) == relation_name_to_id.end())
            {
                relation_name_to_id.insert(make_pair(relation, relation_name_to_id.size()));
                relation_id_to_name.push_back(relation);
            }

            data_train.push_back(make_pair(
                make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
                relation_name_to_id[relation]));

            check_data_train.insert(make_pair(
                make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
                relation_name_to_id[relation]));
            check_data_all.insert(make_pair(
                make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
                relation_name_to_id[relation]));

            set_entity.insert(head);
            set_entity.insert(tail);
            set_relation.insert(relation);

            ++count_entity[head];
            ++count_entity[tail];

            rel_heads[relation_name_to_id[relation]][entity_name_to_id[head]]
                .push_back(entity_name_to_id[tail]);
            rel_tails[relation_name_to_id[relation]][entity_name_to_id[tail]]
                .push_back(entity_name_to_id[head]);
            rel_finder[make_pair(entity_name_to_id[head], entity_name_to_id[tail])]
                = relation_name_to_id[relation];
        }

        fin.close();
    }

    void load_testing(
        const string& filename,
        vector<pair<pair<int, int>, int>>& vin_true,
        vector<pair<pair<int, int>, int>>& vin_false,
        bool self_sampling = false)
    {
        fstream fin(filename.c_str());
        if (self_sampling == false)
        {
            while (!fin.eof())
            {
                string head, tail, relation;
                int flag_true;

                fin >> head >> relation >> tail;
                fin >> flag_true;

                if (head == "" && relation == "" && tail == "") break;

                if (entity_name_to_id.find(head) == entity_name_to_id.end())
                {
                    entity_name_to_id.insert(make_pair(head, entity_name_to_id.size()));
                    entity_id_to_name.push_back(head);
                }

                if (entity_name_to_id.find(tail) == entity_name_to_id.end())
                {
                    entity_name_to_id.insert(make_pair(tail, entity_name_to_id.size()));
                    entity_id_to_name.push_back(tail);
                }

                if (relation_name_to_id.find(relation) == relation_name_to_id.end())
                {
                    relation_name_to_id.insert(make_pair(relation, relation_name_to_id.size()));
                    relation_id_to_name.push_back(relation);
                }

                set_entity.insert(head);
                set_entity.insert(tail);
                set_relation.insert(relation);

                if (flag_true == 1)
                    vin_true.push_back(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
                    relation_name_to_id[relation]));
                else
                    vin_false.push_back(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
                    relation_name_to_id[relation]));

                check_data_all.insert(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
                    relation_name_to_id[relation]));
            }
        }
        else
        {
            while (!fin.eof())
            {
                string head, tail, relation;
                pair<pair<int, int>, int>   sample_false;

                fin >> head >> relation >> tail;

                if (head == "" && relation == "" && tail == "") break;

                if (entity_name_to_id.find(head) == entity_name_to_id.end())
                {
                    entity_name_to_id.insert(make_pair(head, entity_name_to_id.size()));
                    entity_id_to_name.push_back(head);
                }

                if (entity_name_to_id.find(tail) == entity_name_to_id.end())
                {
                    entity_name_to_id.insert(make_pair(tail, entity_name_to_id.size()));
                    entity_id_to_name.push_back(tail);
                }

                if (relation_name_to_id.find(relation) == relation_name_to_id.end())
                {
                    relation_name_to_id.insert(make_pair(relation, relation_name_to_id.size()));
                    relation_id_to_name.push_back(relation);
                }

                set_entity.insert(head);
                set_entity.insert(tail);
                set_relation.insert(relation);

                sample_false_triplet(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
                    relation_name_to_id[relation]), sample_false);

                vin_true.push_back(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
                    relation_name_to_id[relation]));
                vin_false.push_back(sample_false);

                check_data_all.insert(make_pair(make_pair(entity_name_to_id[head], entity_name_to_id[tail]),
                    relation_name_to_id[relation]));
            }
        }

        fin.close();
    }

    void sample_false_triplet(
        const pair<pair<int, int>, int>& origin,
        pair<pair<int, int>, int>& triplet) const
    {

        double prob = relation_hpt[origin.second] / (relation_hpt[origin.second] + relation_tph[origin.second]);

        triplet = origin;
        while (true)
        {
            if (rand() % 1000 < 1000 * prob)
            {
                triplet.first.second = rand() % set_entity.size();
            }
            else
            {
                triplet.first.first = rand() % set_entity.size();
            }

            if (check_data_train.find(triplet) == check_data_train.end())
                return;
        }
    }

    void sample_false_triplet_parts(
        const pair<pair<int, int>, int>& origin,
        pair<pair<int, int>, int>& triplet) const
    {

        double prob = relation_hpt[origin.second] / (relation_hpt[origin.second] + relation_tph[origin.second]);

        triplet = origin;
        while (true)
        {
            if (rand() % 1000 < 1000 * prob)
            {
                triplet.first.second = vector_entity_parts[rand() % vector_entity_parts.size()];
            }
            else
            {
                triplet.first.first = vector_entity_parts[rand() % vector_entity_parts.size()];
            }

            if (check_data_train.find(triplet) == check_data_train.end())
                return;
        }
    }

    void sample_false_triplet_relation(
        const pair<pair<int, int>, int>& origin,
        pair<pair<int, int>, int>& triplet) const
    {

        double prob = relation_hpt[origin.second] / (relation_hpt[origin.second] + relation_tph[origin.second]);

        triplet = origin;
        while (true)
        {
	    int rand_num = rand();
            if (rand_num % 100 < 50)
                triplet.second = rand() % set_relation.size();
            else if (rand_num % 1000 < 1000 * prob)
            {
                triplet.first.second = rand() % set_entity.size();
            }
            else
            {
                triplet.first.first = rand() % set_entity.size();
            }

            if (check_data_train.find(triplet) == check_data_train.end())
                return;
        }
    }

    void sample_false_triplet_relation_parts(
        const pair<pair<int, int>, int>& origin,
        pair<pair<int, int>, int>& triplet) const
    {

        double prob = relation_hpt[origin.second] / (relation_hpt[origin.second] + relation_tph[origin.second]);

        triplet = origin;
        while (true)
        {
	    int rand_num = rand();
            if (rand_num % 100 < 50)
                triplet.second = vector_relation_parts[rand() % vector_relation_parts.size()];
            else if (rand_num % 1000 < 1000 * prob)
            {
                triplet.first.second = vector_entity_parts[rand() % vector_entity_parts.size()];
            }
            else
            {
                triplet.first.first = vector_entity_parts[rand() % vector_entity_parts.size()];
            }

            if (check_data_train.find(triplet) == check_data_train.end())
                return;
        }
    }

    vector<string> split(const string &s, char delim)
  	{
  		stringstream ss(s);
  		string item;
  		vector<string> tokens;
  		while (getline(ss, item, delim)) {
  			tokens.push_back(item);
  		}
  		return tokens;
  	}
};
 
 
