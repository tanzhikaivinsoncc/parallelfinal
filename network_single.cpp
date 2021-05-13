#include <stdio.h>
#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
#include <math.h> 

// g++ -std=c++11 -m64 -O3 -Wall -o network_single network_single.cpp 



struct forwardResult {
    float x;
    std::vector<float> a;
    std::vector<float> z;
    std::vector<float> b;
    std::vector<float> y_hat;
    float j;
};

void printMatrix(std::vector<std::vector<float> > matrix) {

    for (int i = 0; i < matrix.size(); i++) {
        printf("[ ");
        for (int j = 0; j < matrix.at(0).size(); j++) {
            printf("%f ,", matrix.at(i).at(j));
        }
        printf(" ]\n");
    }

}

struct backwardResult {
    std::vector<std::vector<float> > alpha;
    std::vector<std::vector<float> > beta;
};

struct nnResult {
    std::vector<std::vector<float> > weightAlpha;
    std::vector<std::vector<float> > weightBeta;
};



int predict(std::vector<std::vector<float> > data, nnResult nnRes);
std::vector<std::vector<float> > read_csv(std::string filename){
    // Reads a CSV file into a vector of <string, vector<int>> pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    std::vector<std::vector<float> > result;

    // Create an input filestream
    std::ifstream myFile(filename);

    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;
    int val;

    // Read data, line by line
    while(std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);        
        std::vector<float> row;

        // Extract each integer
        while(ss >> val){
            // Add the current integer to the 'colIdx' column's values vector
            row.push_back(val + 0.0);
            
            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
        }

        result.push_back(row);
    }
    // Close file
    myFile.close();

    return result;
}

std::vector<float> linearForward(std::vector<float> x, 
std::vector<std::vector<float> > weights) {
    std::vector<float> results;

    for (int i = 0; i < weights.at(0).size(); i++) {
        float res = 0.0;
        for (int j = 0; j < x.size(); j++) {
            res += weights.at(j).at(i) * x.at(j);
        }

        results.push_back(res);
    }

    return results;
}

std::vector<float> sigmoidForward(std::vector<float> x) {

    std::vector<float> results;

    for (int i = 0; i < x.size(); i++) {
        float res = 1.0 / (1.0 + exp(-1.0 * x.at(i)));
        results.push_back(res);
    }

    return results;
}

std::vector<float> softmaxForward(std::vector<float> x) {

    std::vector<float> results;
    float sum = 0.0;
    // get sum
    for (int i = 0; i < x.size(); i++) {
        float res = exp(x.at(i));
        sum += res;
        results.push_back(res);
    }

    std::vector<float> finalResults;
    // get sum
    for (int i = 0; i < x.size(); i++) {
        float res = results.at(i) / sum;
        finalResults.push_back(res);
    }

    return finalResults;
}

float crossEntropyForward(int true_label, std::vector<float> prediction) {
    float res = -1 * log(prediction.at(true_label));

}

forwardResult nnForward(std::vector<float> train_x, int train_y, 
std::vector<std::vector<float> > alpha, std::vector<std::vector<float> > beta) {
    
    forwardResult forwardres;
    std::vector<float> linearA = linearForward(train_x, alpha);
    forwardres.a = linearA;

    std::vector<float> sigmoidA = sigmoidForward(linearA);
    // add bias to sigmoidA
    sigmoidA.push_back(1.0);
    forwardres.z = sigmoidA;
    std::vector<float> linearB = linearForward(sigmoidA, beta);
    forwardres.b = linearB;
    std::vector<float> softmaxB = softmaxForward(linearB);
    forwardres.y_hat = softmaxB;
    float j = crossEntropyForward(train_y, softmaxB);
    forwardres.j = j;

    return forwardres;
}

// backwards
std::vector<float> softmaxBackwards(std::vector<float> y_hat, int y) {
    std::vector<float> y_hat_cop;
    y_hat_cop = y_hat;

    y_hat_cop.at(y) = -1.0 * (1.0 - y_hat.at(y));

    return y_hat_cop;
}

// get dl/dweights
//  size should be (size of inputs * size of output)
// weights are of shape (size of inputs * size of output) - each row 
// is weight from one node to all nodes in other layer
std::vector<std::vector<float> > linearbackwardsWeight (std::vector<float> input,
std::vector<float> output, std::vector<float> prev_dev) {

    std::vector<std::vector<float> > g_weight;

    for (int i = 0; i < input.size(); i++) {
        std::vector<float> inner_g_weight;
        for (int j = 0; j < prev_dev.size(); j++) {
            inner_g_weight.push_back(input.at(i) * prev_dev.at(j));
        }

        g_weight.push_back(inner_g_weight);
    }

    return g_weight;
}

// weights are row-major order
//  get dl/dinput
// size should be 1 * size of input
std::vector<float> linearbackwardsInput (std::vector<float> input,
std::vector<float> output, std::vector<float> prev_dev, 
std::vector<std::vector<float> > weights) {

    std::vector<float> g_weight;
    for (int i = 0; i < input.size(); i++) {
        float tots = 0.0;
        for (int j = 0; j < weights.at(0).size(); j++) {
            tots += prev_dev.at(j) * weights.at(i).at(j);
        }
        g_weight.push_back(tots);
    }

    return g_weight;
}

std::vector<float> sigmoidBackwards(std::vector<float> input,
std::vector<float> output, std::vector<float> prev_dev) {
    std::vector<float> results;

    for (int i = 0; i < input.size(); i++) {
        float haha = output.at(i) * (1 - output.at(i));
        haha *= prev_dev.at(i);
        results.push_back(haha);
    }

    return results;
}

backwardResult nnBackward(std::vector<float> train_x, int train_y, 
std::vector<std::vector<float> > alpha, std::vector<std::vector<float> > beta, 
forwardResult forwardres) {
    std::vector<float> g_b = softmaxBackwards(forwardres.y_hat, train_y);
    std::vector<std::vector<float> > g_beta = linearbackwardsWeight(forwardres.z,
    forwardres.b, g_b);
    std::vector<float > g_z = linearbackwardsInput (forwardres.z,
    forwardres.b, g_b, beta);
    std::vector<float> g_a = sigmoidBackwards(forwardres.a, forwardres.z, g_z);
    std::vector<std::vector<float> > g_alpha = linearbackwardsWeight(train_x,
    forwardres.a, g_a);
    backwardResult backwardres;
    backwardres.alpha = g_alpha;
    backwardres.beta = g_beta;
    
    // printMatrix(g_beta);
    // printMatrix(g_alpha);

    return backwardres;
}

std::vector<std::vector<float> > initializeWeights(int numInput, int numOutput) {

    std::vector<std::vector<float> > weights;

    // change accordingly
    for (int i = 0; i < numInput; i++) {
        std::vector<float> innerweights;
        for (int i = 0; i < numOutput; i++) {
            innerweights.push_back(0.0);
        }
        weights.push_back(innerweights);
    }
    
    return weights;
}

std::vector<std::vector<float> > updateWeights(
    std::vector<std::vector<float> > weights, std::vector<std::vector<float> > dev,
    float learning_rate) {

    std::vector<std::vector<float> > results;

    // change accordingly
    for (int i = 0; i < weights.size(); i++) {
        std::vector<float> innerweights;
        for (int j = 0; j < weights.at(0).size(); j++) {

            float res = weights.at(i).at(j) - (learning_rate * dev.at(i).at(j));

            innerweights.push_back(res);
        }
        results.push_back(innerweights);
    }
    
    return results;
}

nnResult train(std::vector<std::vector<float> > data, int hidden_units, int num_labels, float learning_rate) {
    

    std::vector<std::vector<float> >weightAlpha = initializeWeights(data.front().size() - 1, hidden_units);
    std::vector<std::vector<float> > weightBeta = initializeWeights(hidden_units + 1, num_labels);

    for (int epoch = 0; epoch < 200; epoch++) {
        for (int i = 0; i < data.size() ; i++) {
        std::vector<float> data_cop;
        data_cop.insert(data_cop.begin(), data.at(i).begin() + 0, data.at(i).begin() + data.at(i).size());
        // printf("copy size is %d\n", data_cop.size());
        int label = data_cop.at(0);
        std::vector<float>::iterator it;
        it = data_cop.begin();
        data_cop.erase(it);
        // printf("copy size is %d\n", data_cop.size());
        forwardResult forwardres = nnForward(data_cop, label, weightAlpha, weightBeta);
        backwardResult backres = nnBackward(data_cop, label, weightAlpha, weightBeta, 
        forwardres);

        weightAlpha = updateWeights(weightAlpha, backres.alpha, learning_rate);
        weightBeta = updateWeights(weightBeta, backres.beta, learning_rate);
    }
    }
    nnResult nnRes;
    nnRes.weightAlpha = weightAlpha;
    nnRes.weightBeta = weightBeta;

    predict(data, nnRes); 

    return nnRes;
}

int findMax(std::vector<float> y_hat) {

    int index = 0;
    float max = y_hat.at(0);

    for (int i = 0; i < y_hat.size(); i++) {
        if (max < y_hat.at(i)) {
            max = y_hat.at(i);
            index = i;
        }
    }

    return index;
}

int predict(std::vector<std::vector<float> > data, 
nnResult nnRes) {

    float total = data.size() + 0.0;
    float correct = 0.0;

    for (int i = 0; i < data.size() ; i++) {
        
        // remove label column
        int label = data.at(i).at(0);
        std::vector<float>::iterator it;
        it = data.at(i).begin();
        data.at(i).erase(it);

        forwardResult forwardres = nnForward(data.at(i), label, nnRes.weightAlpha, nnRes.weightBeta);
        
        if ((int)label == findMax(forwardres.y_hat)) {
            correct += 1.0;
        }
    }

    float accuracy = correct/total;
    printf("Accuracy is %f\n", accuracy);
    return 0;
}

int main(int argc, char** argv) {

    std::vector<std::vector<float> > data = read_csv("largeTrain.csv");

    printf("Dataset size %d\n", data.size());
    // column 0 is label
    printf("Input size %d\n", data.front().size() - 1);

    nnResult nnRes = train(data, 40, 10, 0.1); 

    // printMatrix(nnRes.weightAlpha);
    // printMatrix(nnRes.weightBeta);

    std::vector<std::vector<float> > dataTest = read_csv("largeTest.csv");
    predict(dataTest, nnRes);
    return 0;
}

