#ifndef _KNN_H
#define _KNN_H

#include "common.hpp"

class knn : public common_data
{
    int k;
    std::vector<data *> *neighbors;


    public:
    knn(int);
    knn();
    ~knn();

    void find_neighbors(data *query);
    void set_k(int num);

    int predict();

    double distance(data *query, data *input);
    double accuracy();
    double test_accuracy();

};
#endif