#include "../include/knn.hpp"
#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "../../include/handle_data.hpp"


knn::knn(int num){
    k = num;
}
knn::knn(){
    //leave empty
}
knn::~knn(){
    //leave empty
}

void knn::find_neighbors(data *query){
    neighbors = new std::vector<data *>;
    double min = std::numeric_limits<double>::max();
    double prev_min = min;
    int counter = 0;

    for(int i = 0; i<k;i++){
        if(i == 0){
            for(int j=0; j<train_data_vector->size(); j++){
                double dist = distance(query, train_data_vector->at(j));
                train_data_vector->at(j)->set_distance(dist);
                if(dist < min){
                    min = dist;
                    counter = j;
                }
            }
            neighbors->push_back(train_data_vector->at(counter));
            prev_min = min;
            min = std::numeric_limits<double>::max();
        }
        else{
            for(int j = 0; j<train_data_vector->size(); j++){
                double dist = train_data_vector->at(j)->get_distance();
                if(dist > prev_min && dist < min){
                    min = dist;
                    counter = j;
                }
            }
            neighbors->push_back(train_data_vector->at(counter));
            prev_min = min;
            min = std::numeric_limits<double>::max();
        }
    }
}

void knn::set_k(int num){
    k = num;
}

int knn::predict(){
    std::map<uint8_t, int> frequency;
    for(int i = 0; i<neighbors->size(); i++){
        if(frequency.find(neighbors->at(i)->get_enum_label()) == frequency.end()){
            frequency[neighbors->at(i)->get_enum_label()] = 1;
        }
        else{
            frequency[neighbors->at(i)->get_enum_label()]++;
        }
    }

    int final = 0;
    int max = 0;

    for(auto kv: frequency){
        if(kv.second > max){
            max = kv.second;
            final = kv.first;
        }
    }
    neighbors->clear();
    return final;
}

//Euclidean distance
double knn::distance(data *query, data *input){

    double distance = 0.0;
    if(query->get_feature_vector_size() != input->get_feature_vector_size()){
        printf("Error: Feature vector size mismatch");
        exit(1);
    }
#ifdef EUCLID
    for(unsigned i = 0; i<query->get_feature_vector_size(); i++){
        distance += pow(query->get_features()->at(i) - input->get_features()->at(i), 2);
    }
    distance = sqrt(distance);
    return distance;
#elif defined MANHATTAN
    for(unsigned i = 0; i<query->get_feature_vector_size(); i++){
        distance += abs(query->get_features()->at(i) - input->get_features()->at(i));
    }
    // printf("Manhattan distance: %.3f\n", distance);
     return distance;
    //Manhattan distance implementation
#endif
}
double knn::accuracy(){
    double curr_acuuracy = 0;
    int counter = 0;
    int data_index = 0;
    for(data *d: *validation_data_vector){
        find_neighbors(d);
        int prediction = predict();
        printf("%d -> %d \n", prediction, d->get_enum_label());
        if(prediction == d->get_enum_label()){
            counter++;
        }
        data_index++;
        printf("Accuracy: %.3f %%\n", (double)counter*100.0/(double)data_index);
    }
    curr_acuuracy = (double)counter*100.0/(double)validation_data_vector->size();
    printf("validation accuracy for k = %d : %.3f %%\n", k,curr_acuuracy);
    return curr_acuuracy;

}
double knn::test_accuracy(){
    double curr_acuuracy = 0;
    int counter = 0;
    for(data *d: *test_data_vector){
        find_neighbors(d);
        int prediction = predict();
        if(prediction == d->get_enum_label()){
            counter++;
        }
    }
    curr_acuuracy = (double)counter*100.0/(double)test_data_vector->size();
    printf("test accuracy for k = %d : %.3f %%\n",k ,curr_acuuracy);
    return curr_acuuracy;
}

int main(){
    handle_data *hd = new handle_data();
    hd->read_data("/Users/arihanttripathi/Documents/C++ML/dataset/MNIST/train-images.idx3-ubyte");
    hd->read_labels("/Users/arihanttripathi/Documents/C++ML/dataset/MNIST/train-labels.idx1-ubyte");
    hd->split_data();
    hd->count_class();
    knn *k = new knn();
    k->set_train_data_vector(hd->get_train_data_vector());
    k->set_test_data_vector(hd->get_test_data_vector());
    k->set_validation_data_vector(hd->get_validation_data_vector());
    double perf = 0;
    double best_perf = 0;
    int best_k = 1;

    for(int i = 1; i<=4; i++){
       if(i==1){
              k->set_k(i);
              perf = k->accuracy();
              best_perf = perf;
         }
         else{
              k->set_k(i);
              perf = k->accuracy();
              if(perf > best_perf){
                best_perf = perf;
                best_k = i;
              }
       }
    }
    k->set_k(best_k);
    k->test_accuracy();
}