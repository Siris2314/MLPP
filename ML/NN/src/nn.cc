#include "nn.hpp"
#include "layer.hpp"
#include "../../include/handle_data.hpp"
#include <numeric>

Network::Network(std::vector<int> spec, int input_size, int num, double lr){

    for(int i = 0; i<spec.size(); i++){
        if(i == 0){
            layers.push_back(new Layer(input_size, spec.at(i)));
        }
        else{
            layers.push_back(new Layer(layers.at(i-1)->neurons.size(), spec.at(i)));
        }
        layers.push_back(new Layer(layers.at(layers.size()-1)->neurons.size(), num));
        this->lR = lr;
    }

}

Network::~Network(){}


double Network::activationFunction(std::vector<double> weights, std::vector<double> input){

    double activation = weights.back();

    for(int i = 0; i<weights.size()-1; i++){
        activation += weights[i] * input[i];
    }

    return activation;

}


double Network::transfer(double activation){

    return 1.0 / (1.0 + exp(-activation));

}

double Network::transferDerivative(double output){

    return output * (1.0 - output);

}

std::vector<double> Network::feedForward(data *data){
    std::vector<double> inputs = *data->get_normalized_features();
    for(int i = 0; i<layers.size(); i++){
        Layer *layer = layers.at(i);
        std::vector<double> newInputs;
        for(Neuron *n: layer->neurons){
            double activation = this->activationFunction(n->weights, inputs);
            n->final = this->transfer(activation);
            newInputs.push_back(n->final);
        }
        inputs = newInputs;
    }
    return inputs;
}

void Network::backPropagate(data *data){
    for(int i = layers.size()-1; i>=0; i++){
        Layer *layer = layers.at(i);
        std::vector<double> errors;
        if(i != layers.size() - 1){
            for(int j = 0; j<layer->neurons.size(); j++){
                double error = 0.0;
                for(Neuron *n: layers.at(i+1)->neurons){
                    error += n->weights.at(j) * n->delta;
                }
                errors.push_back(error);
            }
        }
        else{
            for(int j = 0; j<layer->neurons.size(); j++){
                Neuron *n = layer->neurons.at(j);
                errors.push_back((double)data->get_class_vector().at(j) - n->final);
            }
        }

        for(int j = 0; j<layer->neurons.size(); j++){
            Neuron *n = layer->neurons.at(j);
            n->delta = errors.at(j) * this->transferDerivative(n->final);
        }
    }
    
}

void Network::updateWeights(data *data){
    std::vector<double> inputs = *data->get_normalized_features();
    for(int i = 0; i<layers.size(); i++){
        if(i!=0){
            for(Neuron *n: layers.at(i-1)->neurons){
                inputs.push_back(n->final);
            }
        }
        for(Neuron *n:layers.at(i)->neurons){
            for(int j = 0; j<inputs.size(); j++){
                n->weights.at(j) += this->lR * n->delta * inputs.at(j);
            }
            n->weights.back() += this->lR * n->delta;
        }
        inputs.clear();
    }
}

int Network::predict(data *data){
    std::vector<double> outputs = feedForward(data);
    return std::distance(outputs.begin(), std::max_element(outputs.begin(), outputs.end()));
}

void Network::train(int epochs){
    for(int i = 0; i<epochs;i++){
        double sumError = 0.0;
        for(data *data: *this->train_data_vector){
            std::vector<double> outputs = feedForward(data);
            std::vector<int> expected = data->get_class_vector();
            double errFinal = 0.0;
            for(int j = 0; j<outputs.size(); j++){
                errFinal += pow((double)(expected.at(j) - outputs.at(j)), 2);
            }
            sumError += errFinal;
            backPropagate(data);
            updateWeights(data);
        }
        printf("Interation: %d \t Error=%.4f\n", i,sumError);
    }
}

double Network::test(){
    double allCorrect = 0.0;
    double counter = 0.0;

    for(data *data: *this->test_data_vector){
        counter++;
        int idx = predict(data);
        if(data->get_class_vector().at(idx) == 1) allCorrect++;
    }

    double accuracy = allCorrect / counter;
    return accuracy;

}

void Network::validate(){

    double allCorrect = 0.0;
    double counter = 0.0;

    for(data *data: *this->validation_data_vector){
        counter++;
        int idx = predict(data);
        if(data->get_class_vector().at(idx) == 1) allCorrect++;
    }

    double accuracy = allCorrect / counter;
    printf("Validation Acurracy: %.4f\n", accuracy);

}


int main(){
    handle_data *hd = new handle_data();

    #ifdef MNIST
        hd->read_data("/Users/arihanttripathi/Documents/C++ML/dataset/MNIST/train-images.idx3-ubyte");
        hd->read_labels("/Users/arihanttripathi/Documents/C++ML/dataset/MNIST/train-labels.idx1-ubyte");
        hd->count_class();
    #else
        //Insert your own Data Insertion method
    #endif
        hd->split_data();
        std::vector<int> hiddenLayers = {10};
        auto lambda = [&](){
            Network *net = new Network(hiddenLayers,hd->get_train_data_vector()->at(0)->get_normalized_features()->size(),hd->get_class_counts(),0.25);
            net->set_train_data_vector(hd->get_train_data_vector());
            net->set_test_data_vector(hd->get_test_data_vector());
            net->set_validation_data_vector(hd->get_validation_data_vector());
            net->train(15);
            net->validate();
            printf("Accuracy: %.3f\n", net->test());
        }; 
    lambda();
}