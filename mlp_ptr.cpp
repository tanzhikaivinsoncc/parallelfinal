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
    data_num = 0;
    if (file.is_open()) {
        while (getline(file, line, '\n')) {
            data_num++;
        }
    }
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
        train_data = (double*)malloc(sizeof(double) * data_num * feature_num);
        train_label = (int*)malloc(sizeof(int) * data_num);
    } else {
        test_data = (double*)malloc(data_num * feature_num);
        test_label = (int*)malloc(sizeof(int) * data_num);
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
}


int main() {
    MLP mlp(std::vector<int>{3, 2, 2}, 0.01);
    // Get stat before read csv
    mlp.get_stat("../data/train.csv", "train");
    mlp.read_csv("../data/train.csv", "train");
    cout << "Finished reading training data.\n";

    mlp.get_stat("../data/test.csv", "test");
    mlp.read_csv("../data/test.csv", "test");
    cout << "Finished reading test data. Begin Training.\n";
    // TODO: Write train
}