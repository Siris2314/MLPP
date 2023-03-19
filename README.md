# ML++

ML++ is a C++ Machine Learning Library built from the ground up using only C/C++ standard libraries. ML++ is made not only to be used to also educate on implementing Machine Learning algorithms from scratch in C++.


## Setup

You can choose to either fork the repo or simply download the contents of the repo it, do as you see fit,understanding the structure of how the code is setup is important to understand how to use the library.

### Structure

The structure of the library is as follows:

```
├── ML
│   ├── include
│   │   ├── common.hpp
│   │   ├── data.hpp
│   │   ├── handle_data.hpp
│   ├── KMEANS
│   │   ├── include
│   │   │   ├── kmeans.hpp
│   │   ├── src
│   │   │   ├── kmeans.cc
|   |   ├── Makefile
|   ├── CNN
│   │   ├── include
│   │   │   ├── Config.hpp
│   │   │   ├── convolution.hpp
│   │   │   ├── Layer 
│   │   │   │   ├── Conv.hpp
│   │   │   │   ├── FullyConnected.hpp
│   │   │   │   ├── Pooling.hpp
│   │   │   ├── Activation
│   │   │   │   ├── ReLU.hpp
│   │   │   │   ├── Sigmoid.hpp
│   │   │   │   ├── Softmax.hpp
│   │   │   |   ├── Tanh.hpp
|   |   |   |   ├── Identity.hpp
│   ├── KNN
│   │   ├── include
│   │   │   ├── knn.hpp
│   │   ├── src
│   │   │   ├── knn.cc
|   |   ├── Makefile
│   ├── NN
│   │   ├── include
│   │   │   ├── nn.hpp
│   │   |   ├── layer.hpp
|   |   |   ├── neuron.hpp
│   │   ├── src
│   │   │   ├── nn.cc
│   │   │   ├── layer.cc
│   │   │   ├── neuron.cc
|   |   ├── Makefile
│   ├── src
│   │   ├── common.cc
│   │   ├── data.cc
│   │   ├── handle_data.cc
|   ├── add_model.sh
|   ├── exist.sh
|   ├── Makefile
├── dataset
│   ├── MNIST
```

The `ML` directory contains all the source code for the library, the `dataset` directory contains all the datasets that are used for testing the library, currently only the MNIST dataset is available, but you can test with whichever dataset you want, also be sure to change the path to the dataset in the functions that load the dataset.
### Building

To build the library you can simply run the `make` command in the root directory of the repo, this will build all the libraries and place them in the `lib` directory, you can then link the libraries to your project and use them. Each model has it's own makefile, so you can also build each model individually.

### Current Models

Currently the following models are available:

* K-Means Clustering
* K-Nearest Neighbors
* Neural Network (Feed Forward)
* Convolutional Neural Networks

I will add more models in the future.

### Adding Models

If you want to add models, use the `add_model.sh` script, it will create the necessary directories and files for you, you can then add your code to the files and build the library. This way the structure of the library is maintained.


## License

[MIT](https://choosealicense.com/licenses/mit/)
