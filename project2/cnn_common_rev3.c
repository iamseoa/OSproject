
#include "cnn_common.h"
#include <omp.h>

#define FC1_IN ((CONV_OUT / 2) * (CONV_OUT / 2) * CONV_DEPTH)
#define FC2_IN FC1_OUT

void initialize_weights(CNNModel* model) {
    int kernel[3][3] = {{1,2,1}, {2,4,2}, {1,2,1}};

    #pragma omp parallel for
    for (int d = 0; d < CONV_DEPTH; d++)
        model->conv.biases[d] = 1.0f;

    #pragma omp parallel for collapse(4)
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int c = 0; c < CHANNELS; c++)
            for (int i = 0; i < KERNEL_SIZE; i++)
                for (int j = 0; j < KERNEL_SIZE; j++)
                    model->conv.weights[d][c][i][j] = (float)kernel[i][j];

    #pragma omp parallel for
    for (int i = 0; i < FC1_OUT; i++) {
        model->fc1.biases[i] = 1.0f;
        for (int j = 0; j < ((CONV_OUT / 2) * (CONV_OUT / 2) * CONV_DEPTH); j++)
            model->fc1.weights[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    #pragma omp parallel for
    for (int i = 0; i < FC2_OUT; i++) {
        model->fc2.biases[i] = 1.0f;
        for (int j = 0; j < FC1_OUT; j++)
            model->fc2.weights[i][j] = (i == j) ? 1.0f : 0.0f;
    }
}

void initialize_input(float input[INPUT_SIZE][INPUT_SIZE][CHANNELS], int id) {
    float center = 9.0f * (id + 1);
    #pragma omp parallel for collapse(3)
    for (int c = 0; c < CHANNELS; c++)
        for (int i = 0; i < INPUT_SIZE; i++)
            for (int j = 0; j < INPUT_SIZE; j++)
                input[i][j][c] = (i == 1 && j == 1) ? center : 1.0f;
}

void conv_forward(CNNModel* model, float input[INPUT_SIZE][INPUT_SIZE][CHANNELS], float output[CONV_OUT][CONV_OUT][CONV_DEPTH]) {
    #pragma omp parallel for collapse(3)
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT; i++)
            for (int j = 0; j < CONV_OUT; j++) {
                float sum = model->conv.biases[d];
                for (int c = 0; c < CHANNELS; c++)
                    for (int ki = 0; ki < KERNEL_SIZE; ki++)
                        for (int kj = 0; kj < KERNEL_SIZE; kj++)
                            sum += model->conv.weights[d][c][ki][kj] * input[i+ki][j+kj][c];
                output[i][j][d] = sum;
            }
}

void relu_forward(float in[CONV_OUT][CONV_OUT][CONV_DEPTH], float out[CONV_OUT][CONV_OUT][CONV_DEPTH]) {
    #pragma omp parallel for collapse(3)
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT; i++)
            for (int j = 0; j < CONV_OUT; j++)
                out[i][j][d] = in[i][j][d] > 0 ? in[i][j][d] : 0;
}

void maxpool_forward(float in[CONV_OUT][CONV_OUT][CONV_DEPTH], float out[CONV_OUT/2][CONV_OUT/2][CONV_DEPTH]) {
    #pragma omp parallel for collapse(3)
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT; i += 2)
            for (int j = 0; j < CONV_OUT; j += 2) {
                float maxval = in[i][j][d];
                for (int di = 0; di < 2; di++)
                    for (int dj = 0; dj < 2; dj++)
                        if (in[i+di][j+dj][d] > maxval) maxval = in[i+di][j+dj][d];
                out[i/2][j/2][d] = maxval;
            }
}

void flatten(float in[CONV_OUT/2][CONV_OUT/2][CONV_DEPTH], float out[]) {
    #pragma omp parallel for collapse(3)
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT/2; i++)
            for (int j = 0; j < CONV_OUT/2; j++)
                out[d * (CONV_OUT/2)*(CONV_OUT/2) + i * (CONV_OUT/2) + j] = in[i][j][d];
}

void fc1_forward(CNNModel* model, float input[], float output[FC1_OUT]) {
    #pragma omp parallel for
    for (int i = 0; i < FC1_OUT; i++) {
        float sum = model->fc1.biases[i];
        for (int j = 0; j < FC1_IN; j++)
            sum += model->fc1.weights[i][j] * input[j];
        output[i] = sum;
    }
}

void fc2_forward(CNNModel* model, float input[], float output[FC2_OUT]) {
    #pragma omp parallel for
    for (int i = 0; i < FC2_OUT; i++) {
        float sum = model->fc2.biases[i];
        for (int j = 0; j < FC1_OUT; j++)
            sum += model->fc2.weights[i][j] * input[j];
        output[i] = sum;
    }
}
