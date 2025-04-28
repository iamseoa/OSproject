#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include "layers.h"

#define NUM_INPUTS 9

double input_streams[NUM_INPUTS][3][SIZE][SIZE] = {0};

void init_input_streams() {
    for (int n = 0; n < NUM_INPUTS; n++)
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    input_streams[n][c][i][j] = (double)(n + 1);
}

void print_resource_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    printf("\n=== Resource Usage ===\n");
    printf("Max RSS (memory): %ld KB\n", usage.ru_maxrss);
    printf("Voluntary Context Switches: %ld\n", usage.ru_nvcsw);
    printf("Involuntary Context Switches: %ld\n", usage.ru_nivcsw);
    printf("Minor Page Faults: %ld\n", usage.ru_minflt);
    printf("Major Page Faults: %ld\n", usage.ru_majflt);
}

int main() {
    init_input_streams();

    Conv2DLayer conv_layer = {3, 16, 3};
    MaxPool2DLayer pool_layer = {2};
    FullyConnectedLayer fc1_layer = {FC1_INPUT_SIZE, FC1_SIZE};
    FullyConnectedLayer fc2_layer = {FC1_SIZE, FC2_SIZE};

    for (int f = 0; f < 16; f++)
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    conv_layer.weights[f][c][i][j] = (i % 2 == 0) ? 0.5 : 0.0;

    for (int i = 0; i < FC1_INPUT_SIZE; i++)
        for (int j = 0; j < FC1_SIZE; j++)
            fc1_layer.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;

    for (int j = 0; j < FC1_SIZE; j++)
        fc1_layer.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    for (int i = 0; i < FC1_SIZE; i++)
        for (int j = 0; j < FC2_SIZE; j++)
            fc2_layer.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;

    for (int j = 0; j < FC2_SIZE; j++)
        fc2_layer.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    // output 메모리
    double conv_output[16][VALID_SIZE][VALID_SIZE] = {0};
    double pool_output[16][POOL_SIZE][POOL_SIZE] = {0};
    double flatten_output[FC1_INPUT_SIZE] = {0};
    double fc1_output[FC1_SIZE] = {0};
    double fc2_output[FC2_SIZE] = {0};

    for (int idx = 0; idx < NUM_INPUTS; idx++) {
        printf("\n===== Finished input stream #%d =====\n", idx + 1);

        conv2d_forward(&conv_layer, input_streams[idx], conv_output);
        relu_forward(conv_output);
        maxpool2d_forward(&pool_layer, conv_output, pool_output);
        flatten_forward(pool_output, flatten_output);
        fc_forward(&fc1_layer, flatten_output, fc1_output);
        fc_forward(&fc2_layer, fc1_output, fc2_output);

        printf("Conv2D sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", conv_output[0][i][i]);
        printf("\n");

        printf("FC1 sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", fc1_output[i]);
        printf("\n");

        printf("FC2 sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", fc2_output[i]);
        printf("\n");
    }

    print_resource_usage();
    return 0;
}
