#ifndef _DATA_H
#define _DATA_H

#include <vector>
#include "stdint.h"
#include "stdio.h"

class data
{

    std::vector<uint8_t> * features;
    std::vector<double> * normalized_features;
    std::vector<int> * class_vector;
    uint8_t labels;
    int map_data;
    double distance;

    public:
    data();
    ~data();
    void set_features(std::vector<uint8_t> * );
    void set_normalized_features(std::vector<double> * );
    void set_class_vector(int counts);
    void add_feature(uint8_t);
    void add_feature(double);
    void set_labels(uint8_t );
    void set_distance(double);
    void set_map_data(int);
    void print_normalized_features();

    int get_feature_vector_size();
    uint8_t get_labels();
    uint8_t get_enum_label();

    std::vector<uint8_t> * get_features();
    std::vector<double> * get_normalized_features();
    std::vector<int>  get_class_vector();
    double get_distance();
};

#endif