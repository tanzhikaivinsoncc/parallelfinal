//
// Created by 袁玮哲 on 5/12/21.
//

#include <vector>
#include <string>
#include <iostream>

using std::vector;
using std::cout;

typedef vector<vector<double>> M; // weight matrix
typedef vector<double> B;
typedef vector<int> I;


double vec_prod(B vec1, B vec2, double b) {
    double val = 0.;
    for (int i = 0; i < vec1.size(); i++) {
        val += vec1[i] * vec2[i];
    }
    return val + b;
}

B wx_plus_b(M weight, B bias, B input) {
    B output(bias.size());
    for (int i = 0; i < weight.size(); i++) {
        output[i] = vec_prod(weight[i], input, bias[i]);
    }
    return output;
}

B wx_plus_b_omp(M weight, B bias, B input) {
    B output(bias.size());
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < weight.size(); i++) {

        output[i] = vec_prod(weight[i], input, bias[i]);
    }
    return output;
}


int main() {
    B sz{1000, 2000};

    for (int i = 0; i < sz.size(); i++) {
        B vec1(sz[i]);
        B vec2(sz[i]);
        M m(sz[i], vec1);
        struct timespec before, after;
        clock_gettime(CLOCK_REALTIME, &before);
        for (int j = 0; j < 100; j++) {
            wx_plus_b(m, vec1, vec2);
        }
        clock_gettime(CLOCK_REALTIME, &after);
        double time = (double)(after.tv_sec - before.tv_sec) * 1000.0 + (after.tv_nsec - before.tv_nsec) / 1000000.0;
        cout << "size: " << sz[i] << "\t" << "Time passed " << time / 1000.0 << "s.\n";
    }


}