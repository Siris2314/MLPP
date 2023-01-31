#include "data.hpp"

data::data(){
    features = new std::vector<uint8_t>;
}
data::~data(){
    
}
void data::set_features(std::vector<uint8_t> *list){
    features = list;
}
void data::add_feature(uint8_t num){
    features->push_back(num);
}
void data::set_labels(uint8_t num){
    labels = num;
}
void data::set_map_data(int num){
    map_data = num;
}

void data::set_distance(double num){
    distance = num;
}

int data::get_feature_vector_size(){
    return features->size();
}
uint8_t data::get_labels(){
    return labels;
}
uint8_t data::get_enum_label(){
    return map_data;
}
std::vector<uint8_t> * data::get_features(){
    return features;
}
double data::get_distance(){
    return distance;
}