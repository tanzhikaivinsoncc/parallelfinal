//
// Created by Orange.
//

#ifndef MLP_MLP_VEC_H
#define MLP_MLP_VEC_H
#include <vector>
#include <string>

using std::vector;

typedef vector<vector<double>> M; // weight matrix
typedef vector<double> B;
typedef vector<int> I;

class MLP {
public:
    MLP(std::vector<int> topology, double lr);

    // Function for forward propagation of data
    void forward(B input);

    // Function for backward propagation of errors, use one-hot vector as target
    void backward(B input, I target);

    // activate function
    B sigmoid(const B& vec);

    B softmax(B vec);

    // Function to update the weights of connections.
    void optimize();

    // For debug use
    void set_weights();

    // Train the neural net
    void train(int iter);

    void predict(std::string type);

    void read_csv(std::string filename, std::string type);

    vector<int> topology;
    double lr;
    int label_num;
    int feature_num;

    // weights and biasses
    vector<M> weights;
    vector<M> weights_grad;
    vector<B> biass;
    vector<B> biass_grad;

    // store the value of neurons
    vector<B> layers;
    M train_data;
    I train_label;
    M test_data;
    I test_label;

};


#endif //MLP_MLP_VEC_H
