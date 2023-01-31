#ifndef _handle_data_H
#define _handle_data_H

#include <fstream>
#include "stdint.h"
#include "data.hpp"
#include <vector>
#include <string>
#include <map>
#include <unordered_set>


class handle_data
{

    std::vector<data *>  *data_vector;
    std::vector<data *>  *train_data_vector;
    std::vector<data *>  *test_data_vector;
    std::vector<data *> *validation_data_vector;

    int class_size;
    int feature_size;
    std::map<uint8_t, int> class_map;


    const double train_data_ratio = 0.75;
    const double test_data_ratio = 0.2;
    const double validation_data_ratio = 0.05;


    public:

    handle_data();
    ~handle_data();

    void read_data(std::string file_name);

    void read_labels(std::string file_name);

    void split_data();

    void count_class();

    void normalize();

    uint32_t convert_to_little_endian(const unsigned char* bytes);

    int get_class_counts();

    std::vector<data *> *get_train_data_vector();

    std::vector<data *> *get_test_data_vector();

    std::vector<data *> *get_validation_data_vector();


    


};

#endif