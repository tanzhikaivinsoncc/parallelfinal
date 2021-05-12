//
// Created by Orange.
//

#include "mlp_vec.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>
#include <map>
#include <omp.h>
#include <random>

using std::cout;

//#define DEBUG
#define NCORES 8

/*
 * Initialize MLP, including topology and weights
 * */
MLP::MLP(std::vector<int> topology, double lr) {
    this->topology = topology;
    this->lr = lr;

    // Used for random initialization
    std::default_random_engine generator(666);
    std::normal_distribution<double> distribution(0.f,0.02);

    // first layer is input, we don't care.
    for (int i = 1; i < topology.size(); i++) {
        cout << "Initializing layer " << i << ", contains " << topology[i] << " neurons.\n";
        // Initialize weights and biass.
        // 0 mean, 0.02 std
        M weight;
        B bias;
        for (int j = 0; j < topology[i]; j++) {
            B vec;
            for (int k = 0; k < topology[i - 1]; k++) {
                vec.push_back(distribution(generator));
            }
            weight.push_back(vec);
            bias.push_back(distribution(generator));
        }


        M weight_grad(topology[i], B(topology[i - 1]));
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

#ifdef DEBUG
    cout << "The weights are\n";
    for (int i = 0; i < weights.size(); i++) {
        cout << "The " << i << "th weight matrix:\n";
        for (int j = 0; j < weights[i].size(); j++) {
            for (int k = 0; k < weights[i][j].size(); k++) {
                cout << weights[i][j][k] << "\t";
            }
            cout << "\n";
        }
    }
#endif
}


double vec_prod(B vec1, B vec2, double b) {
    double val = 0.;
    for (int i = 0; i < vec1.size(); i++) {
        val += vec1[i] * vec2[i];
    }
    return val + b;
}
/*
 * Basic Wx + b calculation
 * */
B wx_plus_b(M weight, B bias, B input) {
    B output(bias.size());
#pragma omp parallel for num_threads(NCORES) schedule(static)
    for (int i = 0; i < weight.size(); i++) {
        output[i] = vec_prod(weight[i], input, bias[i]);
    }
    return output;
}

/*
 * Sigmoid activation
 * */
B MLP::sigmoid(const B& vec) {
    B activated_vec(vec.size());
#pragma omp parallel for num_threads(NCORES) schedule(static)
    for (int i = 0; i < vec.size(); i++)
        activated_vec[i] = 1.f / (1.f + exp(-vec[i]));
//        activated_vec.push_back(1.f / (1.f + exp(-elem)));
    return activated_vec;
}


/*
 * Softmax activation, safe version. First go to the log scale,
 * then exponentiate
 * */
B MLP::softmax(B vec) {
    B activated_vec(vec.size());
    // First get the sum
    double max_term = *max_element(std::begin(vec), std::end(vec));
    double log_exp_sum = 0.0;
    for (double elem: vec)
        log_exp_sum += exp(elem - max_term);
    log_exp_sum = max_term + log(log_exp_sum);
#pragma omp parallel for num_threads(NCORES) schedule(static)
    for (int i = 0; i < vec.size(); i++)
        activated_vec[i] = exp(vec[i] - log_exp_sum);
//    for (double elem: vec) {
//        double log_val = exp(elem - log_exp_sum);
//        activated_vec.push_back(log_val);
//    }

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

    uint last_idx = topology.size() - 2;
    layers[last_idx] = wx_plus_b(weights[last_idx], biass[last_idx], input);
    layers[last_idx] = softmax(layers[last_idx]);

#ifdef DEBUG
    cout << "Printing out layer values.\n";
    for (const auto& layer: layers) {
        for (double elem: layer) {
            cout << elem << "\t";
        }
        cout << "\n";
    }
#endif
}

double vec_prod_no_bias(B vec1, B vec2) {
    double val = 0.;
    for (int i = 0; i < vec1.size(); i++) {
        val += vec1[i] * vec2[i];
    }
    return val;
}


void MLP::backward(B input, I target) {
    // last layer
    uint last_idx = topology.size() - 2;
#pragma omp parallel for num_threads(NCORES) schedule(static)
    for (int i = 0; i < target.size(); i++) {
        // biass_grad is the same as layers_grad
        biass_grad[last_idx][i] = layers[last_idx][i] - target[i];
    }

    for (int i = 0; i < biass_grad[last_idx].size(); i++) {
#pragma omp parallel for num_threads(NCORES) schedule(static)
        for (int j = 0; j < layers[last_idx - 1].size(); j++) {
            weights_grad[last_idx][i][j] = biass_grad[last_idx][i] * layers[last_idx - 1][j];
        }
    }

    // Intermediate layers
    for (int i = topology.size() - 3; i > 0; i--) {
        // Update biass_grad, here we need weight transpose
#pragma omp parallel for num_threads(NCORES) schedule(static)
        for (int j = 0; j < weights[i + 1][0].size(); j++) {
            double val = 0.f;
            for (int k = 0; k < weights[i + 1].size(); k++) {
                val += weights[i + 1][k][j] * biass_grad[i + 1][k];
            }
            val *= layers[i][j] * (1 - layers[i][j]);
            biass_grad[i][j] = val;
        }

        // Update weights_grad
        for (int j = 0; j < biass_grad[i].size(); j++) {
#pragma omp parallel for num_threads(NCORES) schedule(static)
            for (int k = 0; k < layers[i - 1].size(); k++) {
                weights_grad[i][j][k] = biass_grad[i][j] * layers[i - 1][k];
            }
        }
    }


    // last layer
    // Update biass_grad
#pragma omp parallel for num_threads(NCORES) schedule(static)
    for (int i = 0; i < weights[1][0].size(); i++) {
        double val = 0.f;
        for (int j = 0; j < weights[1].size(); j++) {
            val += weights[1][j][i] * biass_grad[1][j];
        }
        val *= layers[0][i] * (1 - layers[0][i]);
        biass_grad[0][i] = val;
    }

    // Update weights_grad
    for (int i = 0; i < biass_grad[0].size(); i++) {
#pragma omp parallel for num_threads(NCORES) schedule(static)
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
#pragma omp parallel for num_threads(NCORES) schedule(dynamic)
        for (int j = 0; j < weights[i].size(); j++) {
            for (int k = 0; k < weights[i][0].size(); k++)
                weights[i][j][k] -= lr * weights_grad[i][j][k];
//            biass[i][j] -= lr * biass_grad[i][j];
        }
    }

    for (int i = 0; i < weights.size(); i++) {
#pragma omp parallel for num_threads(NCORES) schedule(dynamic)
        for (int j = 0; j < weights[i].size(); j++)
            biass[i][j] -= lr * biass_grad[i][j];
    }
#ifdef DEBUG
    cout << "After optimization, the weiths are\n";
    for (int i = 0; i < weights.size(); i++) {
        cout << "The " << i << "th weight matrix:\n";
        for (int j = 0; j < weights[i].size(); j++) {
            for (int k = 0; k < weights[i][j].size(); k++) {
                cout << weights[i][j][k] << "\t";
            }
            cout << "\n";
        }
    }
#endif
}

/*
 * Read data from csv file. The first row is header, used for
 * determining the number of columns. The first column is label
 * */
void MLP::read_csv(std::string filename, std::string type) {
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
                if (num == 0) {
                    if (type == "train") {
                        train_label.push_back(std::stoi(&word[0]));
                    } else {
                        test_label.push_back(std::stoi(&word[0]));
                    }
                }
                else
                    curr_data.push_back(std::stof(&word[0]));
                num++;
            }
            if (type == "train")
                train_data.push_back(curr_data);
            else
                test_data.push_back(curr_data);
        }
    }

    // Get the total label num
    int num = 0;
    std::map<int, bool> labelMap;
    for (int elem : train_label)
        labelMap.insert(std::pair<int, bool>(elem, true));

    std::map<int, bool>::iterator iter;
    for (iter = labelMap.begin(); iter != labelMap.end(); iter++)
        num++;
    this->label_num = num;



#ifdef DEBUG
    cout << "number of columns: " << feature_num + 1 << "\n";
    cout << "number of labels: " << label_num << "\n";
    cout << "Printing out data.\n";
    M data;
    I label;
    if (type == "train") {
        data = train_data;
        label = train_label;
    } else {
        data = test_data;
        label = test_label;
    }
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
    I vec(label_num);
    for (int i = 0; i < label_num; i++) {
        if (label != i)
            vec[i] = 0;
        else
            vec[i] = 1;
    }
    return vec;
}
/*
 * Train MLP for iter iterations.
 * */
void MLP::train(int iter) {
    int array[train_data.size()];
    for (int i = 0; i < train_data.size(); i++)
        array[i] = i;
    for (int i = 0; i < iter; i++) {
        // SGD, read every data
        // Random shuffle the data
        std::shuffle(array, array + train_data.size(), std::default_random_engine(666));
        cout << "[Training] On iteration " << i <<".\n";
        struct timespec before, after;
        clock_gettime(CLOCK_REALTIME, &before);
        double forward_time = 0;
        double backward_time = 0;
        double optimize_time = 0;
        struct timespec forward_before, backward_before, optimize_before, optimize_after;
        for (int j = 0; j < train_data.size(); j++) {
            int idx = array[j];
            B curr_data = train_data[idx];
            int curr_label = train_label[idx];
            I label_vec = one_hot_vec(curr_label, this->label_num);
            clock_gettime(CLOCK_REALTIME, &forward_before);
//            this->forward(curr_data);
            clock_gettime(CLOCK_REALTIME, &backward_before);
//            this->backward(curr_data, label_vec);
            clock_gettime(CLOCK_REALTIME, &optimize_before);
            this->optimize();
            clock_gettime(CLOCK_REALTIME, &optimize_after);
            forward_time += (double)(backward_before.tv_sec - forward_before.tv_sec) * 1000.0 + (backward_before.tv_nsec - forward_before.tv_nsec) / 1000000.0;
            backward_time += (double)(optimize_before.tv_sec - backward_before.tv_sec) * 1000.0 + (optimize_before.tv_nsec - backward_before.tv_nsec) / 1000000.0;
            optimize_time += (double)(optimize_after.tv_sec - optimize_before.tv_sec) * 1000.0 + (optimize_after.tv_nsec - optimize_before.tv_nsec) / 1000000.0;

        }
        clock_gettime(CLOCK_REALTIME, &after);
        double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
        cout << "[Training] Finished iteration " << i << ". Forward time: " << forward_time / 1000.0 << " s.\n";
        cout << "[Training] Finished iteration " << i << ". Backward time: " << backward_time / 1000.0 << " s.\n";
        cout << "[Training] Finished iteration " << i << ". Optimize time: " << optimize_time / 1000.0 << " s.\n";
        cout << "[Training] Finished iteration " << i << ". Total time:" << delta_ms / 1000.0 << " s.\n";
        cout << "[Training Acc.]\n";
        clock_gettime(CLOCK_REALTIME, &before);
        predict("train");
        clock_gettime(CLOCK_REALTIME, &after);
        delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
        cout << "[Training Acc.] Prediction time: " << delta_ms / 1000.0 << " s.\n";
        cout << "[Test Acc.]\n";
        clock_gettime(CLOCK_REALTIME, &before);
        predict("test");
        clock_gettime(CLOCK_REALTIME, &after);
        delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
        cout << "[Test Acc.] Prediction time: " << delta_ms / 1000.0 << " s.\n";
    }
}

/*
 * Make predictions on test data, and calculate accuracy.
 * */
void MLP::predict(std::string type) {
    // both training data and test data
    int correct = 0;
    M data;
    I label;
    if (type == "train") {
        data = train_data;
        label = train_label;
    } else {
        data = test_data;
        label = test_label;
    }

    for (int i = 0; i < data.size(); i++) {
        B curr_data = data[i];
        int curr_label = label[i];
        forward(curr_data);
        uint last_idx = topology.size() - 2;
        B output = layers[last_idx];
        int best_idx = -1;
        double best_prob = -INFINITY;
        for (int j = 0; j < output.size(); j++) {
            if (output[j] > best_prob) {
                best_prob = output[j];
                best_idx = j;
            }
        }
        if (best_idx == curr_label)
            correct++;
    }
    cout << "[Prediction] correct: " << correct << ", total: " << data.size() << ".\n";
    cout << "[Prediction] Final accuracy is " << (double)correct / (double)data.size() << "\n";

}

int main() {
    MLP mlp(std::vector<int>{784, 512, 10}, 0.01);
    mlp.read_csv("../data/ocr_train.csv", "train");
    mlp.read_csv("../data/ocr_test.csv", "test");
    cout << "Finished reading data. Begin Training.\n";
    mlp.train(1);
}