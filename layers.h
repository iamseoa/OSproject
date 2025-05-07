#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SIZE 128
#define VALID_SIZE (SIZE - 2)
#define POOL_SIZE (VALID_SIZE / 2)
#define FC1_SIZE 128
#define FC2_SIZE 20
#define FC1_INPUT_SIZE (16 * POOL_SIZE * POOL_SIZE)

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    double ****weights;
} Conv2DLayer;

typedef struct {
    int pool_size;
} MaxPool2DLayer;

typedef struct {
    int input_size;
    int output_size;
    double **weights;
    double *bias;
} FullyConnected1Layer;

typedef struct {
    int input_size;
    int output_size;
    double **weights;
    double *bias;
} FullyConnected2Layer;

// 함수 선언
void conv2d_forward(Conv2DLayer* layer, double input[3][SIZE][SIZE], double output[16][VALID_SIZE][VALID_SIZE]);
void relu_forward(double data[16][VALID_SIZE][VALID_SIZE]);
void maxpool2d_forward(MaxPool2DLayer* layer, double input[16][VALID_SIZE][VALID_SIZE], double output[16][POOL_SIZE][POOL_SIZE]);
void flatten_forward(double input[16][POOL_SIZE][POOL_SIZE], double* output);
void fc1_forward(FullyConnected1Layer* layer, double* input, double* output);
void fc2_forward(FullyConnected2Layer* layer, double* input, double* output);

#endif

// ===== 함수 정의 시작 =====

void conv2d_forward(Conv2DLayer* layer, double input[3][SIZE][SIZE], double output[16][VALID_SIZE][VALID_SIZE]) {
    for (int f = 0; f < layer->out_channels; f++) {
        for (int i = 1; i < SIZE-1; i++) {
            for (int j = 1; j < SIZE-1; j++) {
                double sum = 0.0;
                for (int c = 0; c < layer->in_channels; c++) {
                    for (int ki = -1; ki <= 1; ki++) {
                        for (int kj = -1; kj <= 1; kj++) {
                            sum += input[c][i+ki][j+kj] * layer->weights[f][c][ki+1][kj+1];
                        }
                    }
                }
                output[f][i-1][j-1] = sum;
            }
        }
    }
}

void relu_forward(double data[16][VALID_SIZE][VALID_SIZE]) {
    for (int f = 0; f < 16; f++) {
        for (int i = 0; i < VALID_SIZE; i++) {
            for (int j = 0; j < VALID_SIZE; j++) {
                if (data[f][i][j] < 0)
                    data[f][i][j] = 0;
            }
        }
    }
}

void maxpool2d_forward(MaxPool2DLayer* layer, double input[16][VALID_SIZE][VALID_SIZE], double output[16][POOL_SIZE][POOL_SIZE]) {
    for (int f = 0; f < 16; f++) {
        for (int i = 0; i < VALID_SIZE; i += 2) {
            for (int j = 0; j < VALID_SIZE; j += 2) {
                double max_val = input[f][i][j];
                if (input[f][i+1][j] > max_val) max_val = input[f][i+1][j];
                if (input[f][i][j+1] > max_val) max_val = input[f][i][j+1];
                if (input[f][i+1][j+1] > max_val) max_val = input[f][i+1][j+1];
                output[f][i/2][j/2] = max_val;
            }
        }
    }
}

void flatten_forward(double input[16][POOL_SIZE][POOL_SIZE], double* output) {
    int idx = 0;
    for (int f = 0; f < 16; f++) {
        for (int i = 0; i < POOL_SIZE; i++) {
            for (int j = 0; j < POOL_SIZE; j++) {
                output[idx++] = input[f][i][j];
            }
        }
    }
}

void fc1_forward(FullyConnected1Layer* layer, double* input, double* output) {
    for (int j = 0; j < layer->output_size; j++) {
        output[j] = layer->bias[j];
        for (int i = 0; i < layer->input_size; i++) {
            output[j] += input[i] * layer->weights[i][j];
        }
    }
}

void fc2_forward(FullyConnected2Layer* layer, double* input, double* output) {
    for (int j = 0; j < layer->output_size; j++) {
        output[j] = layer->bias[j];
        for (int i = 0; i < layer->input_size; i++) {
            output[j] += input[i] * layer->weights[i][j];
        }
    }
}
