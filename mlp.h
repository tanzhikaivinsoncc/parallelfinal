//
// Created by Orange.
//

#ifndef MLP_MLP_H
#define MLP_MLP_H
#include <vector>
#include <string>

using std::vector;

typedef vector<vector<float>> M; // weight matrix
typedef vector<float> B;
typedef vector<int> I;

class MLP {
public:
    MLP(std::vector<int> topology, float lr);

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

    void predict();

    void read_csv(std::string filename);

    vector<int> topology;
    float lr;
    int label_num;
    int feature_num;

    // weights and biasses
    vector<M> weights;
    vector<M> weights_grad;
    vector<B> biass;
    vector<B> biass_grad;

    // store the value of neurons
    vector<B> layers;
    M data;
    I label;



};


#endif //MLP_MLP_H
