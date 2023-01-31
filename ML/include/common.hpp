#ifndef __COMMON_HPP
#define __COMMON_HPP

#include "data.hpp"
#include <vector>

class common_data{

    protected:
    std::vector<data *>  *train_data_vector;
    std::vector<data *>  *test_data_vector;
    std::vector<data *> *validation_data_vector;

    public:

    void set_train_data_vector(std::vector<data *> *list);
    void set_test_data_vector(std::vector<data *> *list);
    void set_validation_data_vector(std::vector<data *> *list);

};

#endif