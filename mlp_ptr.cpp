//
// Created by Orange.
//

#include "mlp_ptr.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>
#include <map>
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

    weights = (double**)malloc(sizeof(double*) * (topology.size() - 1));
    weights_grad = (double**)malloc(sizeof(double*) * (topology.size() - 1));
    biass = (double**)malloc(sizeof(double*) * (topology.size() - 1));
    biass_grad = (double**)malloc(sizeof(double*) * (topology.size() - 1));
    layers = (double**)malloc(sizeof(double*) * (topology.size() - 1));

    for (int i = 1; i < topology.size(); i++) {
        // Go through every layer.

        cout << "Initializing layer " << i << ", contains " << topology[i] << " neurons.\n";
        // Initialize weights and biass. 0 mean, 0.02 std
        weights[i - 1] = (double*)malloc(sizeof(double) * topology[i] * topology[i - 1]);
        weights_grad[i - 1] = (double*)malloc(sizeof(double) * topology[i] * topology[i - 1]);
        biass[i - 1] = (double*)malloc(sizeof(double) * topology[i]);
        biass_grad[i - 1] = (double*)malloc(sizeof(double) * topology[i]);
        layers[i - 1] = (double*)malloc(sizeof(double) * topology[i]);

        for (int j = 0; j < topology[i] * topology[i - 1]; j++)
            weights[i - 1][j] = distribution(generator);

        for (int j = 0; j < topology[i]; j++)
            biass[i - 1][j] = distribution(generator);

    }

    cout << "Finished init.\n";

}


/*
 * Read data from csv file. The first row is header, used for
 * determining the number of columns. The first column is label
 * */
void MLP::get_stat(std::string filename, std::string type) {
    std::ifstream file(filename);
    std::string line, word;
    // determine number of columns in file
    getline(file, line, '\n');
    std::stringstream ss(line);
    std::vector<std::string> parsed_vec;
    while (getline(ss, word, ',')) {
        parsed_vec.emplace_back(&word[0]);
    }

    // Get feature num and total data num
    feature_num = parsed_vec.size() - 1;
    int data_num = 0;
    if (file.is_open()) {
        while (getline(file, line, '\n')) {
            data_num++;
        }
    }
    if (type == "train")
        train_data_num = data_num;
    else
        test_data_num = data_num;
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
    std::vector<std::string> parsed_vec;
    while (getline(ss, word, ',')) {
        parsed_vec.emplace_back(&word[0]);
    }

    if (type == "train") {
        train_data = (double*)malloc(sizeof(double) * train_data_num * feature_num);
        train_label = (int*)malloc(sizeof(int) * train_data_num);
    } else {
        test_data = (double*)malloc(test_data_num * feature_num);
        test_label = (int*)malloc(sizeof(int) * test_data_num);
    }

    if (file.is_open()) {
        int line_num = 0;
        while (getline(file, line, '\n')) {
            std::stringstream ss(line);
            int num = 0;
            while (getline(ss, word, ',')) {
                if (num == 0) {
                    if (type == "train") {
                        train_label[num] = std::stoi(&word[0]);
                    } else {
                        test_label[num] = std::stoi(&word[0]);
                    }
                } else {
                    if (type == "train")
                        train_data[line_num * feature_num + num - 1] = std::stof(&word[0]);
                    else
                        test_data[line_num * feature_num + num - 1] = std::stof(&word[0]);
                }
                num++;
            }
            line_num++;
        }
    }

    // Get the total label num
    this->label_num = topology[topology.size() - 1];
    // Initialize one hot vector
    label_vec = (int*)malloc(sizeof(int) * label_num);
}

/*
 * Basic Wx + b calculation
 * */
void wx_plus_b(int row_size, int col_size, double* weight, double* bias, double* input, double* layer) {
    // Inplace matrix operation
    for (int i = 0; i < row_size; i++) {
        double sum = 0.0;
        for (int j = 0; j < col_size; j++) {
            sum += weight[i * col_size + j] * input[j];
        }
        layer[i] = sum + bias[i];
    }
}

/*
 * Sigmoid activation
 * */
void sigmoid(int size, double* input) {
    for (int i = 0; i < size; i++) {
        input[i] = 1.0 / (1.0 + exp(-input[i]));
    }
}

/*
 * Softmax activation, safe version. First go to the log scale,
 * then exponentiate
 * */
void softmax(int size, double* input) {
    // First get the sum
    double max_term;
    for (int i = 0; i < size; i++) {
        if (i == 0)
            max_term = input[i];
        else {
            if (input[i] > max_term)
                max_term = input[i];
        }
    }
    double log_exp_sum = 0.0;
    for (int i = 0; i < size; i++)
        log_exp_sum += exp(input[i] - max_term);
    log_exp_sum = max_term + log(log_exp_sum);

    for (int i = 0; i < size; i++)
        input[i] = exp(input[i] - log_exp_sum);
}


void MLP::forward(double* curr_data) {
    // propagate the data forward
    for (int i = 0; i < topology.size() - 2; i++) {
        wx_plus_b(topology[i + 1], topology[i], weights[i], biass[i], curr_data, layers[i]);
        sigmoid(topology[i + 1], layers[i]);
        curr_data = layers[i];
    }
    uint last_idx = topology.size() - 2;
    wx_plus_b(topology[last_idx + 1], topology[last_idx], weights[last_idx], biass[last_idx], curr_data, layers[last_idx]);
    softmax(topology[last_idx + 1], layers[last_idx]);

}

void MLP::backward(double* curr_data) {
    // last layer
    uint last_idx = topology.size() - 2;

    for (int i = 0; i < label_num; i++) {
        biass_grad[last_idx][i] = layers[last_idx][i] - label_vec[i];
    }
    for (int i = 0; i < topology[last_idx + 1]; i++) {
        for (int j = 0; j < topology[last_idx]; j++)
            weights_grad[last_idx][topology[last_idx] * i + j] = biass_grad[last_idx][i] * layers[last_idx][j];
    }

    // Intermediate layers
    for (int i = topology.size() - 3; i > 0; i--) {
        // update biass_grad, here we need weight transpose
        for (int j = 0; j < topology[i + 1]; j++) {
            double val = 0.0;
            for (int k = 0; k < topology[i + 2]; k++) {
                val += weights[i + 1][k * topology[i + 1] + j] * biass_grad[i + 1][k];
            }
            val *= layers[i][j] * (1 - layers[i][j]);
            biass_grad[i][j] = val;
        }
        // Update weights_grad
        for (int j = 0; j < topology[i + 1]; j++) {
            for (int k = 0; k < topology[i]; k++) {
                weights_grad[i][j * topology[i] + k] = biass_grad[i][j] * layers[i - 1][k];
            }
        }
    }

    // last layer
    // Update biass_grad
    for (int i = 0; i < topology[1]; i++) {
        double val = 0.0;
        for (int j = 0; j < topology[2]; j++) {
            val += weights[1][j * topology[1] + i] * biass_grad[1][j];
        }
        val *= layers[0][i] * (1 - layers[0][i]);
        biass_grad[0][i] = val;
    }
    // Update weights_grad
    for (int i = 0; i < topology[1]; i++) {
        for (int j = 0; j < topology[0]; j++) {
            weights_grad[0][i * topology[0] + j] = biass_grad[0][i] * curr_data[j];
        }
    }

}

/*
 * Update weights based on gradient information.
 * */
void MLP::optimize() {
    for (int i = 0; i < topology.size() - 1; i++) {
        // Update weights[i] and biass[i]
        for (int j = 0; j < topology[i + 1]; j++) {
            for (int k = 0; k < topology[i]; k++)
                weights[i][j * topology[i] + k] -= lr * weights_grad[i][j * topology[i] + k];
            biass[i][j] -= lr * biass_grad[i][j];
        }
    }
}


void MLP::one_hot_vec(int label) {
    for (int i = 0; i < label_num; i++) {
        if (label != i)
            label_vec[i] = 0;
        else
            label_vec[i] = 1;
    }
}
/*
 * Train MLP for iter iterations.
 * */
void MLP::train(int iter) {
    int array[train_data_num];
    for (int i = 0; i < train_data_num; i++)
        array[i] = i;
    for (int i = 0; i < iter; i++) {
        // SGD, read every data
        // Random shuffle the data
        std::shuffle(array, array + train_data_num, std::default_random_engine(666));
        cout << "[Training] On iteration " << i <<".\n";
        struct timespec before, after;
        clock_gettime(CLOCK_REALTIME, &before);
        for (int j = 0; j < train_data_num; j++) {
            int idx = array[j];
            double* curr_data = train_data + feature_num * idx;
            int curr_label = train_label[idx];
            one_hot_vec(curr_label);
            this->forward(curr_data);
            this->backward(curr_data);
            this->optimize();
        }
        clock_gettime(CLOCK_REALTIME, &after);
        double delta_ms = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
        cout << "[Training] Finished iteration " << i << ". Time passed " << delta_ms / 1000.0 << " s.\n";
        cout << "[Training Acc.]\n";
        predict("train");
        cout << "[Test Acc.]\n";
        predict("test");
    }
}

void MLP::predict(std::string type) {
    // both training data and test data
    int correct = 0;
    double* data;
    int* label;
    int data_num;
    if (type == "train") {
        data = train_data;
        label = train_label;
        data_num = train_data_num;
    } else {
        data = test_data;
        label = test_label;
        data_num = test_data_num;
    }

    for (int i = 0; i < data_num; i++) {
        double* curr_data = data + feature_num * i;
        int curr_label = label[i];
        forward(curr_data);
        uint last_idx = topology.size() - 2;
        int best_idx = -1;
        double best_prob;
        for (int j = 0; j < topology[last_idx]; j++) {
            if (j == 0) {
                best_idx = j;
                best_prob = layers[last_idx][j];
            } else {
                if (layers[last_idx][j] > best_prob) {
                    best_prob = layers[last_idx][j];
                    best_idx = j;
                }
            }
        }
        if (best_idx == curr_label)
            correct++;
    }
    cout << "[Prediction] correct: " << correct << ", total: " << data_num << ".\n";
    cout << "[Prediction] Final accuracy is " << (double)correct / (double)data_num << "\n";
}

int main() {
    MLP mlp(std::vector<int>{784, 128, 10}, 0.01);
    // Get stat before read csv
    mlp.get_stat("../data/orc_train.csv", "train");
    mlp.read_csv("../data/orc_train.csv", "train");
    cout << "Finished reading training data.\n";

    mlp.get_stat("../data/ocr_test.csv", "test");
    mlp.read_csv("../data/orc_test.csv", "test");
    cout << "Finished reading test data. Begin Training.\n";
    // TODO: Write train
    mlp.train(50);
}