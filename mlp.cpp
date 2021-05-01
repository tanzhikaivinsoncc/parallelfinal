//
// Created by Orange.
//

#include "mlp.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>
#include <map>

using std::cout;

//#define DEBUG

/*
 * Initialize MLP, including topology and weights
 * */
MLP::MLP(std::vector<int> topology, float lr) {
    this->topology = topology;
    this->lr = lr;

    // first layer is input, we don't care.
    for (int i = 1; i < topology.size(); i++) {
        cout << "Initializing layer " << i << ", contains " << topology[i] << " neurons.\n";
        // Initialize weights and biass.
        M weight(topology[i], B(topology[i - 1]));
        M weight_grad(topology[i], B(topology[i - 1]));
        B bias(topology[i]);
        B bias_grad(topology[i]);
        weights.push_back(weight);
        weights_grad.push_back(weight_grad);
        biass.push_back(bias);
        biass_grad.push_back(bias_grad);

        // Initialize all neurons
        B neurons(topology[i]);
        layers.push_back(neurons);
    }
    cout << "Finished init.\n";
}

/*
 * Basic Wx + b calculation
 * */
B wx_plus_b(M weight, B bias, B input) {
    B output;
    for (int i = 0; i < weight.size(); i++) {
        float v = 0.f;
        for (int j = 0; j < weight[0].size(); j++) {
            v += weight[i][j] * input[j];
        }
        v += bias[i];
        output.push_back(v);
    }
    return output;
}

/*
 * Sigmoid activation
 * */
B MLP::sigmoid(const B& vec) {
    B activated_vec;
    for (float elem: vec)
        activated_vec.push_back(1.f / (1.f + exp(-elem)));
    return activated_vec;
}


/*
 * Softmax activation, safe version. First go to the log scale,
 * then exponentiate
 * */
B MLP::softmax(B vec) {
    B activated_vec;
    // First get the sum
    float max_term = *max_element(std::begin(vec), std::end(vec));
    float log_exp_sum = 0.0;
    for (float elem: vec)
        log_exp_sum += exp(elem - max_term);
    log_exp_sum = max_term + log(log_exp_sum);

    for (float elem: vec) {
        float log_val = exp(elem - log_exp_sum);
        activated_vec.push_back(log_val);
    }

    return activated_vec;
}

/*
 * Set the weights to pre-defined values. This is for debug use.
 * */
void MLP::set_weights() {
    for (auto & weight : weights) {
        for (int i = 0; i < weight.size(); i++) {
            for (int j = 0; j < weight[0].size(); j++) {
                weight[i][j] = i + j;
            }
        }
    }
    for (auto & bias : biass) {
        for (int i = 0; i < bias.size(); i++)
            bias[i] = i;
    }

}

void MLP::forward(B input) {
    // propagate the data forward
    for (int i = 0; i < topology.size() - 2; i++) {
        layers[i] = wx_plus_b(weights[i], biass[i], input);
        layers[i] = sigmoid(layers[i]);
        input = layers[i];
    }
#ifdef DEBUG
    cout << "last_idx " << last_idx << "\n";
    cout << "weights.size() " << weights.size() << "\n";
    cout << "biass.size() " << biass.size() << "\n";
#endif
    uint last_idx = topology.size() - 2;
    layers[last_idx] = wx_plus_b(weights[last_idx], biass[last_idx], input);
    layers[last_idx] = softmax(layers[last_idx]);

#ifdef DEBUG
    cout << "Printing out layer values.\n";
    for (const auto& layer: layers) {
        for (float elem: layer) {
            cout << elem << "\t";
        }
        cout << "\n";
    }
#endif
}

void MLP::backward(B input, I target) {
    // last layer
    uint last_idx = topology.size() - 2;
    for (int i = 0; i < target.size(); i++) {
        // biass_grad is the same as layers_grad
        biass_grad[last_idx][i] = layers[last_idx][i] - target[i];
    }
    for (int i = 0; i < biass_grad[last_idx].size(); i++) {
        for (int j = 0; j < layers[last_idx - 1].size(); j++) {
            weights_grad[last_idx][i][j] = biass_grad[last_idx][i] * layers[last_idx - 1][j];
        }
    }

    // Intermediate layers
    for (int i = topology.size() - 3; i > 0; i--) {
        // Update biass_grad, here we need weight transpose
        for (int j = 0; j < weights[i + 1][0].size(); j++) {
            float val = 0.f;
            for (int k = 0; k < weights[i + 1].size(); k++) {
                val += weights[i + 1][k][j] * biass_grad[i + 1][k];
            }
            val *= layers[i][j] * (1 - layers[i][j]);
            biass_grad[i][j] = val;
        }

        // Update weights_grad
        for (int j = 0; j < biass_grad[i].size(); j++) {
            for (int k = 0; k < layers[i - 1].size(); k++) {
                weights_grad[i][j][k] = biass_grad[i][j] * layers[i - 1][k];
            }
        }
    }


    // last layer
    // Update biass_grad
    for (int i = 0; i < weights[1][0].size(); i++) {
        float val = 0.f;
        for (int j = 0; j < weights[1].size(); j++) {
            val += weights[1][j][i] * biass_grad[1][j];
        }
        val *= layers[0][i] * (1 - layers[0][i]);
        biass_grad[0][i] = val;
    }

    // Update weights_grad
    for (int i = 0; i < biass_grad[0].size(); i++) {
        for (int j = 0; j < input.size(); j++) {
            weights_grad[0][i][j] = biass_grad[0][i] * input[j];
        }
    }

#ifdef DEBUG
    cout << "Gradient info.\n";
    cout << "weights_grad:\n";
    for (auto weight_grad: weights_grad) {
        for (int i = 0; i < weight_grad.size(); i++) {
            for (int j = 0; j < weight_grad[0].size(); j++) {
                cout << weight_grad[i][j] << "\t";
            }
            cout << "\n";
        }
        cout << "-----------------\n";
    }
    cout << "biass_grads:\n";
    for (auto bias_grad: biass_grad) {
        for (int i = 0; i < bias_grad.size(); i++) {
            cout << bias_grad[i] << "\t";
        }
        cout << "\n--------------\n";
    }
#endif

}

/*
 * Update weights based on gradient information.
 * */
void MLP::optimize() {
    for (int i = 0; i < weights.size(); i++) {
        // update weights[i] and biass[i]
        for (int j = 0; j < weights[i].size(); j++) {
            for (int k = 0; k < weights[i][0].size(); k++)
                weights[i][j][k] -= lr * weights_grad[i][j][k];
            biass[i][j] -= lr * biass_grad[i][j];
        }
    }
}

/*
 * Read data from csv file. The first row is header, used for
 * determining the number of columns. The first column is label
 * */
void MLP::read_csv(std::string filename) {
    std::ifstream file(filename);
    std::string line, word;
    // determine number of columns in file
    getline(file, line, '\n');
    std::stringstream ss(line);
    vector<std::string> parsed_vec;
    while (getline(ss, word, ',')) {
        parsed_vec.emplace_back(&word[0]);
    }

    feature_num = parsed_vec.size() - 1;
    if (file.is_open()) {
        while (getline(file, line, '\n')) {
            std::stringstream ss(line);
            B curr_data;
            int num = 0;
            while (getline(ss, word, ',')) {
                if (num == 0)
                    label.push_back(std::stoi(&word[0]));
                else
                    curr_data.push_back(std::stof(&word[0]));
                num++;
            }
            data.push_back(curr_data);
        }
    }

    // Get the total label num
    int num = 0;
    std::map<int, bool> labelMap;
    for (int elem : label)
        labelMap.insert(std::pair<int, bool>(elem, true));

    std::map<int, bool>::iterator iter;
    for (iter = labelMap.begin(); iter != labelMap.end(); iter++)
        num++;
    this->label_num = num;



#ifdef DEBUG
    cout << "number of columns: " << cols << "\n";
    cout << "number of labels: " << label_num << "\n";
    cout << "Printing out data.\n";
    for (int i = 0; i < data.size(); i++) {
        for (int j = 0; j < data[0].size(); j++) {
            cout << data[i][j] << "\t";
        }
        cout << "\n";
    }
    cout << "Printing out labels.\n";
    for (int i = 0; i < label.size(); i++)
        cout << label[i] << "\n";
#endif
}

I one_hot_vec(int label, int label_num) {
    I vec;
    for (int i = 0; i < label_num; i++) {
        if (label != i)
            vec.push_back(0);
        else
            vec.push_back(1);
    }
    return vec;
}
/*
 * Train MLP for iter iterations.
 * */
void MLP::train(int iter) {
    for (int i = 0; i < iter; i++) {
        // SGD, read every data
        cout << "[Training] On iteration " << i <<".\n";
        struct timespec before, after;
        clock_gettime(CLOCK_REALTIME, &before);
        for (int j = 0; j < data.size(); j++) {
            B curr_data = data[j];
            int curr_label = label[j];
            I label_vec = one_hot_vec(curr_label, this->label_num);
            this->forward(curr_data);
            this->backward(curr_data, label_vec);
            this->optimize();
        }
        clock_gettime(CLOCK_REALTIME, &after);
        double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
        cout << "[Training] Finished iteration " << i << ". Time passed " << delta_ms / 1000.0 << " s.\n";
        this->predict();
    }
}

/*
 * Make predictions on test data, and calculate accuracy.
 * */
void MLP::predict() {
    int correct = 0;
    for (int i = 0; i < this->data.size(); i++) {
        if (i % 100 == 0) {
            cout << "[Prediction] Finished predicting "<< i << " samples.\n";
        }
        B curr_data = data[i];
        int curr_label = label[i];
        this->forward(curr_data);
        uint last_idx = topology.size() - 2;
        B output = this->layers[last_idx];
        int best_idx = -1;
        float best_prob = -INFINITY;
        for (int j = 0; j < output.size(); j++) {
            if (output[j] > best_prob) {
                best_prob = output[j];
                best_idx = j;
            }
        }
        if (best_idx == curr_label)
            correct++;
    }
    cout << "correct: " << correct << ", total: " << data.size() << ".\n";
    cout << "Final accuracy is " << correct / data.size() << "\n";

}

int main() {
    MLP mlp(std::vector<int>{784, 1024, 512, 10}, 0.001);
    mlp.read_csv("../data/mnist_train.csv");
    cout << "Finished reading data. Begin Training.\n";
    mlp.train(10);
    mlp.data.clear();
    mlp.label.clear();
    mlp.read_csv("../data/mnist_test.csv");
    cout << "On test data\n";
    mlp.predict();

}