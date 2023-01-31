#include "handle_data.hpp"

handle_data::handle_data(){
    data_vector = new std::vector<data *>;
    train_data_vector = new std::vector<data *>;
    test_data_vector = new std::vector<data *>;
    validation_data_vector = new std::vector<data *>;
    
}
handle_data::~handle_data(){

}

void handle_data::read_data(std::string file_name){
    //4 because MAGIC NUMBER, NUMBER OF IMAGES, NUMBER OF ROWS, NUMBER OF COLUMNS
    uint32_t header[4];
    unsigned char bytes[4];
    FILE *fp = fopen(file_name.c_str(), "r");

    if(fp){
        for(int i=0; i<4; i++){
            if(fread(bytes, sizeof(bytes), 1, fp)){
                header[i] = convert_to_little_endian(bytes);
            }
        }
        printf("Got file header.\n");
        int image_size = header[2] * header[3];
        for(int i = 0; i < header[1]; i++){
            data *d = new data();
            uint8_t element[1];
            for(int j = 0; j < image_size; j++){
                if(fread(element, sizeof(element), 1, fp)){
                    d->add_feature(element[0]);
                }
                else{
                    printf("Error reading file.\n");
                    exit(1);
                }
            }
            data_vector->push_back(d);
        }
        printf("Got all %lu features.\n", data_vector->size());
    }
    else{
        printf("Yes\n");
        printf("Error opening file.\n");
        exit(1);
    }
}
void handle_data::read_labels(std::string file_name){

    //2 because MAGIC NUMBER, NUMBER OF LABELS
    uint32_t header[2];
    unsigned char btyes[4];
    FILE *fp = fopen(file_name.c_str(), "r");

    if(fp){
        for(int i=0; i<2; i++){
            if(fread(btyes, sizeof(btyes), 1, fp)){
                header[i] = convert_to_little_endian(btyes);
            }
        }
        printf("Got Label file header.\n");
        for(int i = 0; i < header[1]; i++){
            uint8_t element[1];
            if(fread(element, sizeof(element), 1, fp)){
                data_vector->at(i)->set_labels(element[0]);
            }
            else{
                printf("Error reading file.\n");
                exit(1);
            }
        }
        printf("Got all %lu labels.\n", data_vector->size());
    }
    else{
        printf("Error opening file.\n");
        exit(1);
    }
}
void handle_data::split_data(){
    std::unordered_set<int> used_indexes;
    int train_data_size = data_vector->size() * train_data_ratio;
    int test_data_size = data_vector->size() * test_data_ratio;
    int validation_data_size = data_vector->size() * validation_data_ratio;

    //train data
    int count = 0;
    while(count < train_data_size){
        int index = rand() % data_vector->size();
        if(used_indexes.find(index) == used_indexes.end()){
            train_data_vector->push_back(data_vector->at(index));
            used_indexes.insert(index);
            count++;
        }
    }

    //test data
    count = 0;
    while(count <test_data_size){
        int index = rand() % data_vector->size();
        if(used_indexes.find(index) == used_indexes.end()){
            test_data_vector->push_back(data_vector->at(index));
            used_indexes.insert(index);
            count++;
        }
    }

    //validation data
    count = 0;
    while(count <validation_data_size){
        int index = rand() % data_vector->size();
        if(used_indexes.find(index) == used_indexes.end()){
            validation_data_vector->push_back(data_vector->at(index));
            used_indexes.insert(index);
            count++;
        }
    }
    
    printf("Train size: %lu\n", train_data_vector->size());
    printf("Test size: %lu\n", test_data_vector->size());
    printf("Validation size: %lu\n", validation_data_vector->size());




}

void handle_data::normalize(){
    std::vector<double> mins, maxs;

    data *d = data_vector->at(0);

    for(auto val : *d->get_features()){
        mins.push_back(val);
        maxs.push_back(val);
    }

    for(int i = 1; i<data_vector->size();i++){
        d = data_vector->at(i);
        for(int j = 0; j<d->get_features()->size(); j++){
            double val = (double)d->get_features()->at(j);
            if(val < mins.at(j)){
                mins[j] = val;
            }
            if(val > maxs.at(j)){
                maxs[j] = val;
            }
        }
    }

    for(int i = 0; i<data_vector->size(); i++){
        data_vector->at(i)->set_normalized_features(new std::vector<double>());
        data_vector->at(i)->set_class_vector(class_size);
        for(int j = 0; j<data_vector->at(i)->get_feature_vector_size(); j++){
            if(maxs[j] - mins[j] == 0){
                data_vector->at(i)->add_feature(0.0);
            }
            else{
                data_vector->at(i)->add_feature((double)(data_vector->at(i)->get_features()->at(j) - mins[j]) / (maxs[j] - mins[j]));
            }
        }
    }


}
void handle_data::count_class(){
    int count = 0;
    for(unsigned i = 0; i<data_vector->size(); i++){
        if(class_map.find(data_vector->at(i)->get_labels()) == class_map.end()){
            class_map[data_vector->at(i)->get_labels()] = count;
            data_vector->at(i)->set_map_data(count);
            count++;
        }
    }
    class_size = count;
    for(data *data : *data_vector){
        data->set_class_vector(class_size);
    }
    printf("Class size: %d\n", class_size);
}

uint32_t handle_data::convert_to_little_endian(const unsigned char* bytes){
    return (uint32_t)((bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | (bytes[3]));
}

int handle_data::get_class_counts(){
    return class_size;
}

std::vector<data *> * handle_data::get_train_data_vector(){
    return train_data_vector;
}

std::vector<data *> * handle_data::get_test_data_vector(){
    return test_data_vector;
}

std::vector<data *> * handle_data::get_validation_data_vector(){
    return validation_data_vector;
}

// int main(){
//     handle_data *hd = new handle_data();
//     hd->read_data("/Users/arihanttripathi/Documents/C++ML/train-images.idx3-ubyte");
//     hd->read_labels("/Users/arihanttripathi/Documents/C++ML/train-labels.idx1-ubyte");
//     hd->split_data();
//     hd->count_class();
// }