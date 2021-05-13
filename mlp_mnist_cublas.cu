#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <stdio.h>

#include "CycleTimer.h"

using namespace std;

// each element stores number of neurons in that layer
vector<int> neurons;
int num_layers;
vector<float **> weights;
vector<float *> bias;

vector<float **> zs;
vector<float **> as;

// p contains the softmax arrays
// e.g. p[0] = [label_1_prob, ..., label_10_prob]
float **p;

float **dJ_dp;
vector<float **> dp_da;
vector<float **> dJ_da;
vector<float **> dJ_dz;
vector<float **> dJ_dW;
vector<float *> dJ_db;

enum ACTIVATION_TYPE { SIGMOID, TANH };
int num_classes;
int num_epochs;
float learning_rate;
int batch_size;

const int num_train_data = 60000;
const int num_test_data = 10000;
const int pixels = 28 * 28;

struct dataset {
  float trainingImages[num_train_data][pixels];
  float testImages[num_test_data][pixels];

  int trainingLabels[num_train_data];
  int testLabels[num_test_data];
};

const char *train_label_file = "train-labels-idx1-ubyte";
const char *train_file = "train-images-idx3-ubyte";
const char *test_label_file = "t10k-labels-idx1-ubyte";
const char *test_file = "t10k-images-idx3-ubyte";

dataset dataSet;

int reverseInt(int i) {
  unsigned char c1, c2, c3, c4;

  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;

  return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist_labels(const char *label_file, int *label_array) {
  // dataset dataSet;

  ifstream file(label_file);
  if (file.is_open()) {

    int magic_number = 0;
    int number_of_images = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    for (int i = 0; i < number_of_images; ++i) {
      unsigned char temp = 0;
      file.read((char *)&temp, sizeof(temp));
      label_array[i] = temp;
    }
  }
}

void read_mnist_values(const char *image_file, float image_array[][pixels]) {
  // dataset dataSet;

  ifstream file(image_file);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    file.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    file.read((char *)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    file.read((char *)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);

    for (int i = 0; i < number_of_images; ++i) {
      for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_cols; ++c) {
          unsigned char temp = 0;
          file.read((char *)&temp, sizeof(temp));

          // normalize input???
          image_array[r][c] = (static_cast<float>(temp) / 255.0);
        }
      }
    }
  }
}

void read_mnist() {

  read_mnist_labels(train_label_file, dataSet.trainingLabels);
  read_mnist_labels(test_label_file, dataSet.testLabels);

  read_mnist_values(train_file, dataSet.trainingImages);
  read_mnist_values(test_file, dataSet.testImages);
}

// try to initialize weights to be very small
void InitializeWeights() {

  weights = vector<float **>(num_layers);
  bias = vector<float *>(num_layers);
  for (int i = 1; i < num_layers; i++) {
    float **w = new float *[neurons[i - 1]];
    for (int j = 0; j < neurons[i - 1]; j++) {
      w[j] = new float[neurons[i]];
      for (int k = 0; k < neurons[i]; k++)
        w[j][k] =
            (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) / 10.0;
    }
    weights[i] = w;
    float *b = new float[neurons[i]];
    for (int j = 0; j < neurons[i]; j++) {
      b[j] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) / 10.0;
      // printf("(%0.5f, )",b[j]);
    }

    bias[i] = b;
  }
}

void Init() {

  zs = vector<float **>(num_layers);
  for (int i = 0; i < num_layers; i++) {
    float **z = new float *[batch_size];
    for (int j = 0; j < batch_size; j++)
      z[j] = new float[neurons[i]];
    zs[i] = z;
  }

  as = vector<float **>(num_layers);
  for (int i = 0; i < num_layers; i++) {
    float **a = new float *[batch_size];
    for (int j = 0; j < batch_size; j++)
      a[j] = new float[neurons[i]];
    as[i] = a;
  }

  p = new float *[batch_size];
  for (int i = 0; i < batch_size; i++)
    p[i] = new float[num_classes];

  // contains the derivatives
  dJ_dp = new float *[batch_size];
  for (int i = 0; i < batch_size; i++)
    dJ_dp[i] = new float[num_classes];

  dp_da = vector<float **>(batch_size);
  for (int i = 0; i < batch_size; i++) {
    float **_dp_da = new float *[num_classes];
    for (int j = 0; j < num_classes; j++)
      _dp_da[j] = new float[num_classes];
    dp_da[i] = _dp_da;
  }

  dJ_da = vector<float **>(num_layers);
  for (int i = 0; i < num_layers; i++) {
    float **_dJ_da = new float *[batch_size];
    for (int j = 0; j < batch_size; j++)
      _dJ_da[j] = new float[neurons[i]];
    dJ_da[i] = _dJ_da;
  }

  dJ_dz = vector<float **>(num_layers);
  for (int i = 0; i < num_layers; i++) {
    float **_dJ_dz = new float *[batch_size];
    for (int j = 0; j < batch_size; j++)
      _dJ_dz[j] = new float[neurons[i]];
    dJ_dz[i] = _dJ_dz;
  }

  dJ_dW = vector<float **>(num_layers);
  dJ_db = vector<float *>(num_layers);
  for (int i = 1; i < num_layers; i++) {
    float **_dJ_dW = new float *[neurons[i - 1]];
    for (int j = 0; j < neurons[i - 1]; j++)
      _dJ_dW[j] = new float[neurons[i]];
    dJ_dW[i] = _dJ_dW;

    float *_dJ_db = new float[neurons[i]];
    dJ_db[i] = _dJ_db;
  }
}

// activation function is sigmoid
float ActivationFunction(const float z) {
  // sigmoid
  // return 1.0 / (1.0 + exp(-z));

  return tanh(z);
}

float ActivationDerivative(const float a) {
  // return a * (1.0 - a);

  // tanh
  // (e^z - e^-z)/(e^z + e^-z)
  return 1.0 - (tanh(a) * tanh(a));
}

void Softmax(float **const a) {
  for (int i = 0; i < batch_size; i++) {
    float exp_sum = 0.0;
    for (int j = 0; j < num_classes; j++) {
      p[i][j] = exp(a[i][j]);
      exp_sum += p[i][j];
    }
    for (int j = 0; j < num_classes; j++)
      p[i][j] /= exp_sum;
  }
}

void Forward() {
  for (int n = 1; n < num_layers; n++) {

    // double startTime = CycleTimer::currentSeconds();
    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < neurons[n]; j++) {
        float sum = 0.0;
        for (int k = 0; k < neurons[n - 1]; k++)
          sum += as[n - 1][i][k] * weights[n][k][j];
        zs[n][i][j] = sum + bias[n][j];
        as[n][i][j] = ActivationFunction(zs[n][i][j]);
      }
    }
    // double endTime = CycleTimer::currentSeconds();
    // double overallDuration =
    //     (endTime - startTime);
    // printf("Forward: %.3f ms\t\t (%d to %d)\n\n", 1000.f * overallDuration, n
    // - 1, n);
  }

  //   double startTimeSM = CycleTimer::currentSeconds();
  Softmax(as[num_layers - 1]);
  //   double endTimeSM = CycleTimer::currentSeconds();
  //     double overallDurationSM =
  //         (endTimeSM - startTimeSM);
  //     printf("Forward: %.10f ms\t\t (last to softmax)\n\n", 1000.f *
  //     overallDurationSM);
}

float Loss(float **const y) {
  float sum = 0.0;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_classes; j++) {
      if (p[i][j] > 0)
        sum += -log(p[i][j]) * y[i][j];
    }
  }

  return sum / (batch_size * num_classes);
}

int Accuracy(const int start_idx) {
  int acc_cnt = 0;
  for (int i = 0; i < batch_size; i++) {
    int max_label = -1;
    float max_prob = 0.0;
    for (int j = 0; j < num_classes; j++) {
      if (p[i][j] > max_prob) {
        max_label = j;
        max_prob = p[i][j];
      }
    }

    if (max_label == dataSet.testLabels[start_idx + i])
      acc_cnt++;
  }

  return acc_cnt;
}

// y is the true label
// p is the softmax
void Backpropagation(float **const y) {

  //   double startTimeSoftmax = CycleTimer::currentSeconds();

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_classes; j++) {
      dJ_dp[i][j] = -y[i][j] / p[i][j];
    }
  }

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_classes; j++) {
      for (int k = 0; k < num_classes; k++) {
        if (j == k)
          dp_da[i][j][k] = p[i][j] - p[i][j] * p[i][j];
        else
          dp_da[i][j][k] = -p[i][j] * p[i][k];
      }
    }
  }

  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < num_classes; j++) {
      float sum = 0.0;
      for (int k = 0; k < num_classes; k++)
        sum += dp_da[i][j][k] * dJ_dp[i][k];
      dJ_da[num_layers - 1][i][j] = sum;
    }
  }

  //   double endTimeSoftmax = CycleTimer::currentSeconds();
  //   double overallDurationSoftmax = (endTimeSoftmax - startTimeSoftmax);
  //   printf("Forward: %.3f ms\t\t softmax\n\n", 1000.f *
  //   overallDurationSoftmax);

  for (int n = num_layers - 1; n >= 1; n--) {
    // double startTime = CycleTimer::currentSeconds();

    for (int i = 0; i < batch_size; i++)
      for (int j = 0; j < neurons[n]; j++)
        dJ_dz[n][i][j] = dJ_da[n][i][j] * ActivationDerivative(as[n][i][j]);

    for (int i = 0; i < neurons[n - 1]; i++) {
      for (int j = 0; j < neurons[n]; j++) {
        float sum = 0.0;
        for (int k = 0; k < batch_size; k++)
          sum += as[n - 1][k][i] * dJ_dz[n][k][j];
        dJ_dW[n][i][j] = sum / batch_size;
      }
    }

    for (int i = 0; i < neurons[n]; i++) {
      float sum = 0.0;
      for (int j = 0; j < batch_size; j++)
        sum += dJ_dz[n][j][i];
      dJ_db[n][i] = sum / batch_size;
    }

    for (int i = 0; i < batch_size; i++) {
      for (int j = 0; j < neurons[n - 1]; j++) {
        float sum = 0.0;
        for (int k = 0; k < neurons[n]; k++)
          sum += dJ_dz[n][i][k] * weights[n][j][k];
        dJ_da[n - 1][i][j] = sum;
      }
    }

    //     double endTime = CycleTimer::currentSeconds();
    //   double overallDuration = (endTime - startTime);
    //   printf("Forward: %.3f ms\t\t (%d to %d)\n\n", 1000.f * overallDuration,
    //   n,
    //          n - 1);
  }
}

void UpdateGradient() {
  for (int n = 1; n < num_layers; n++) {
    for (int i = 0; i < neurons[n - 1]; i++)
      for (int j = 0; j < neurons[n]; j++)
        weights[n][i][j] -= learning_rate * dJ_dW[n][i][j];
    for (int i = 0; i < neurons[n]; i++) {
      bias[n][i] -= learning_rate * dJ_db[n][i];
    }
  }
}

void Train() {
  int num_batches = num_train_data / batch_size;

  for (int epoch = 1; epoch <= num_epochs; epoch++) {

    double forwardTime = 0.0;
    double backTime = 0.0;

    float epoch_loss = 0.0;
    for (int b = 0; b < num_batches; b++) {
      float **y_batch = new float *[batch_size];
      for (int i = 0; i < batch_size; i++)
        y_batch[i] = new float[num_classes];

      int start_idx = b * batch_size;
      for (int i = 0; i < batch_size; i++) {
        // change this to use label - y_batch is the true label
        y_batch[i][dataSet.trainingLabels[start_idx + i]] = 1.0;

        for (int j = 0; j < neurons[0]; j++)
          as[0][i][j] = dataSet.trainingImages[start_idx + i][j];
      }

      double startTime = CycleTimer::currentSeconds();
      Forward();
      double endTime = CycleTimer::currentSeconds();
      double overallDuration = endTime - startTime;
      forwardTime += overallDuration;

      epoch_loss += Loss(y_batch);

      double backStartTime = CycleTimer::currentSeconds();
      Backpropagation(y_batch);
      double backendTime = CycleTimer::currentSeconds();
      double backOverallDuration = backendTime - backStartTime;
      backTime += backOverallDuration;

      UpdateGradient();
    }

    printf("Forward: %.3f ms\t\t Backward: %.3f ms\t\t\n\n",
           1000.f * forwardTime, 1000.f * backTime);

    epoch_loss /= num_batches;
    printf("Epoch %3d, loss=%.6f\n", epoch, epoch_loss);
  }
}

void Test() {
  int num_batches = num_test_data / batch_size;

  int acc = 0;
  for (int b = 0; b < num_batches; b++) {
    int start_idx = b * batch_size;
    for (int i = 0; i < batch_size; i++)
      for (int j = 0; j < neurons[0]; j++)
        as[0][i][j] = dataSet.testImages[start_idx + i][j];

    Forward();
    acc += Accuracy(start_idx);
  }

  printf("Test Accuracy: %.6f\n", (float)acc / num_test_data);
}

int main(int argc, char *argv[]) {

//   int layers[] = {784, 128, 64, 10};
  int layers[] = {784, 512, 256, 128, 64,84, 10};
  num_layers = 6;

  neurons = vector<int>(layers, layers + num_layers);

  // 10 classes for
  num_classes = 10;

  // set the parameters here
  num_epochs = 1;
  learning_rate = 1e-1;
  batch_size = 100;

  srand(time(0));

  printf("Step 1: Read Dataset - Start\n");
  read_mnist();
  printf("Step 1: Read Dataset - Done\n");

  printf("Step 2: Initialize Weights - Start\n");
  InitializeWeights();
  printf("Step 2: Initialize Weights - Done\n");

  printf("Step 3: Initialize Backprop Arrays - Start\n");
  Init();
  printf("Step 3: Initialize Backprop Arrays - Done\n");

  printf("Step 4: Training - Start\n");
  Train();
  printf("Step 4: Training - Done\n");

  Test();

  return 0;
}
