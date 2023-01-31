#ifndef _KMEAN_HPP
#define _KMEAN_HPP

#include "common.hpp"
#include <unordered_set>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <map>
#include "handle_data.hpp"

typedef struct cluster{

    std::vector<double> *centroid;
    std::vector<data *> *data_points;
    std::map<int, int> label_count;
    int all_classes;
     
    cluster(data *start){
        centroid = new std::vector<double>;
        data_points = new std::vector<data *>;
        for(auto val : *(start->get_features())){
            centroid->push_back(val);
        }
        data_points->push_back(start);
        label_count[start->get_enum_label()] = 1;
        all_classes = start->get_enum_label();
    }

    void add_data_point(data *new_data){
        int size = data_points->size();
        data_points->push_back(new_data);
        for(int i = 0; i<centroid->size()-1; i++){
            double value = centroid->at(i);
            value *= size;
            value += new_data->get_features()->at(i);
            value /= (double)data_points->size();
            centroid->at(i) = value;
        }
        if(label_count.find(new_data->get_enum_label()) == label_count.end()){
            label_count[new_data->get_enum_label()] = 1;
        }
        else{
            label_count[new_data->get_enum_label()]++;
        }
        set_class();

    }

    void set_class(){
        int max;
        int freq = 0;
        for(auto val : label_count){
            if(val.second > freq){
                max = val.first;
                freq = val.second;
            }
        }
    }

    
} cluster_t;


class kmeans: public common_data
{
    int num_clusters;
    std::vector<cluster_t *> *clusters;
    std::unordered_set<int> *used_data_points;

    public:
    kmeans(int k);
    void init_clusters();
    void init_clusters_class();
    void train();
    double distance(std::vector<double> *, data *);
    double accuracy();
    double test_accuracy();
};

#endif