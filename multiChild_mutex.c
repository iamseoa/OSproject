#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <pthread.h>
#include "layers.h"

#define NUM_INPUTS 9

double input_streams[NUM_INPUTS][3][SIZE][SIZE];

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

void process_stream(int idx) {
    double conv_output[16][VALID_SIZE][VALID_SIZE] = {0};
    double pool_output[16][POOL_SIZE][POOL_SIZE] = {0};
    double flatten_output[FC1_INPUT_SIZE];
    double fc1_output[FC1_SIZE];
    double fc2_output[FC2_SIZE];

    Conv2DLayer conv_layer;
    conv_layer.in_channels = 3;
    conv_layer.out_channels = 16;
    conv_layer.kernel_size = 3;
    for (int f = 0; f < 16; f++)
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    conv_layer.weights[f][c][i][j] = (i % 2 == 0) ? 0.5 : 0.0;

    MaxPool2DLayer pool_layer = {2};

    FullyConnected1Layer fc1_layer;
    fc1_layer.input_size = FC1_INPUT_SIZE;
    fc1_layer.output_size = FC1_SIZE;
    for (int i = 0; i < FC1_INPUT_SIZE; i++)
        for (int j = 0; j < FC1_SIZE; j++)
            fc1_layer.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
    for (int j = 0; j < FC1_SIZE; j++)
        fc1_layer.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    FullyConnected2Layer fc2_layer;
    fc2_layer.input_size = FC1_SIZE;
    fc2_layer.output_size = FC2_SIZE;
    for (int i = 0; i < FC1_SIZE; i++)
        for (int j = 0; j < FC2_SIZE; j++)
            fc2_layer.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
    for (int j = 0; j < FC2_SIZE; j++)
        fc2_layer.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    // Conv → ReLU → Pool → FC1 → FC2
    conv2d_forward(&conv_layer, input_streams[idx], conv_output);
    relu_forward(conv_output);
    maxpool2d_forward(&pool_layer, conv_output, pool_output);
    flatten_forward(pool_output, flatten_output);
    fc1_forward(&fc1_layer, flatten_output, fc1_output);
    fc2_forward(&fc2_layer, fc1_output, fc2_output);

    // 출력
    printf("\n===== Finished input stream #%d (PID: %d) =====\n", idx + 1, getpid());

    printf("Conv2D output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", conv_output[0][i][i]);
    printf("\n");

    printf("FC1 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc1_output[i]);
    printf("\n");

    printf("FC2 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc2_output[i]);
    printf("\n");
}

int main() {
    init_input_streams();

    for (int idx = 0; idx < NUM_INPUTS; idx++) {
        pid_t pid = fork();
        if (pid == 0) {
            process_stream(idx);
            exit(0);
        }
    }

    for (int idx = 0; idx < NUM_INPUTS; idx++) {
        wait(NULL);
    }

    print_resource_usage();
    return 0;
}
