#include "common.hpp"

void common_data::set_train_data_vector(std::vector<data *> *list){
    train_data_vector = list;
}
void common_data::set_test_data_vector(std::vector<data *> *list){
    test_data_vector = list;
}
void common_data::set_validation_data_vector(std::vector<data *> *list){
    validation_data_vector = list;
}