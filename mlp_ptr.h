//
// Created by Orange.
//

#ifndef MLP_MLP_PTR_H
#define MLP_MLP_PTR_H
#include <string>
#include <vector>

class MLP {
public:
    MLP(std::vector<int> topology, double lr);

    // Function for forward propagation of data
    void forward();

    // Function for backward propagation of errors, use one-hot vector as target
    void backward();

    // activate function
    void sigmoid();

    void softmax();

    // Function to update the weights of connections.
    void optimize();

    // Train the neural net
    void train(int iter);

    void predict(std::string type);

    void read_csv(std::string filename, std::string type);

    void get_stat(std::string filename, std::string type);

    std::vector<int> topology;
    double lr;
    int label_num;
    int feature_num;
    int data_num;

    // weights and biasses
    double** weights;
    double** weights_grad;
    double** biass;
    double** biass_grad;

    // store the value of neurons
    double** layers;
    double* train_data;
    int* train_label;
    double* test_data;
    int* test_label;

};


#endif //MLP_MLP_PTR_H


