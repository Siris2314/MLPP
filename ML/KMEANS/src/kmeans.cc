#include "../include/kmeans.hpp"


kmeans::kmeans(int k){
    num_clusters = k;
    clusters = new std::vector<cluster_t *>;
    used_data_points = new std::unordered_set<int>;
}
void kmeans::init_clusters(){

    for(int i = 0; i<num_clusters; i++){
        int index = rand() % train_data_vector->size();
        while(used_data_points->find(index) != used_data_points->end()){
            index = rand() % train_data_vector->size();
        }
        clusters->push_back(new cluster_t(train_data_vector->at(index)));
        used_data_points->insert(index);
    }
}
void kmeans::init_clusters_class(){
    std::unordered_set<int> used_classes;
    for(int i = 0; i<train_data_vector->size(); i++){
       if(used_classes.find(train_data_vector->at(i)->get_enum_label()) == used_classes.end()){
           clusters->push_back(new cluster_t(train_data_vector->at(i)));
           used_classes.insert(train_data_vector->at(i)->get_enum_label());
           used_data_points->insert(i);
       }
    }
}
void kmeans::train(){
    int index = 0;
    while(used_data_points->size() < train_data_vector->size()){
        while(used_data_points->find(index) != used_data_points->end()){
            index++;
        }
        double min = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for(int j = 0; j<clusters->size(); j++){
            double dist = distance(clusters->at(j)->centroid, train_data_vector->at(index));
            if(dist < min){
                min = dist;
                best_cluster = j;
            }
        }
        clusters->at(best_cluster)->add_data_point(train_data_vector->at(index));
        used_data_points->insert(index);
       
    }
}
double kmeans::distance(std::vector<double> *centroid, data *point){
    double dist = 0.0;
    for(int i = 0; i<centroid->size(); i++){
        dist += pow(centroid->at(i) - point->get_features()->at(i), 2);
    }
    return sqrt(dist);
}
double kmeans::accuracy(){
    double num_correct = 0.0;
    for(auto point: *validation_data_vector){
        double min = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for(int j = 0; j<clusters->size(); j++){
            double dist = distance(clusters->at(j)->centroid, point);
            if(dist < min){
                min = dist;
                best_cluster = j;
            }
        }
        if(clusters->at(best_cluster)->all_classes == point->get_enum_label()){
            num_correct++;
        }
    }
    return (num_correct / (double)validation_data_vector->size()) * 100.0;
}
double kmeans::test_accuracy(){
    double num_correct = 0.0;
    for(auto point: *test_data_vector){
        double min = std::numeric_limits<double>::max();
        int best_cluster = 0;
        for(int j = 0; j<clusters->size(); j++){
            double dist = distance(clusters->at(j)->centroid, point);
            if(dist < min){
                min = dist;
                best_cluster = j;
            }
        }
        if(clusters->at(best_cluster)->all_classes == point->get_enum_label()){
            num_correct++;
        }
    }
    return (num_correct / (double)test_data_vector->size()) * 100.0;

}

int main(){
    handle_data *hd = new handle_data();
    hd->read_data("/Users/arihanttripathi/Documents/C++ML/dataset/MNIST/train-images.idx3-ubyte");
    hd->read_labels("/Users/arihanttripathi/Documents/C++ML/dataset/MNIST/train-labels.idx1-ubyte");
    hd->split_data();
    hd->count_class();
    double perf = 0.0;
    double test_perf = 0.0;
    int best_k = 1;
    for(int k = hd->get_class_counts(); k<hd->get_train_data_vector()->size(); k++){
        kmeans *km = new kmeans(k);
        km->set_train_data_vector(hd->get_train_data_vector());
        km->set_test_data_vector(hd->get_test_data_vector());
        km->set_validation_data_vector(hd->get_validation_data_vector());
        km->init_clusters();
        km->train();
        perf = km->accuracy();
        printf("Current K: %d, Accuracy: %.2f\n", k, perf);
        if(perf > test_perf){
            test_perf= perf;
            best_k = k;
        }
        
    }

        kmeans *km = new kmeans(best_k);
        km->set_train_data_vector(hd->get_train_data_vector());
        km->set_test_data_vector(hd->get_test_data_vector());
        km->set_validation_data_vector(hd->get_validation_data_vector());
        km->init_clusters();
        perf = km->test_accuracy();
        printf("Current K: %d, Accuracy: %.2f\n", best_k, perf);    
    
}