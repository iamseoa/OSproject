#include "layers.h"
#include <pthread.h>
#include <sys/resource.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_INPUTS 9
#define NUM_THREADS 4

double input_streams[NUM_INPUTS][3][SIZE][SIZE] = {0};
pthread_mutex_t fc_mutex;

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

typedef struct {
    int idx;
    Conv2DLayer* conv_layer;
    MaxPool2DLayer* pool_layer;
    FullyConnected1Layer* fc1_layer;
    FullyConnected2Layer* fc2_layer;
} ThreadArg;

void* thread_worker(void* arg_ptr) {
    ThreadArg* arg = (ThreadArg*)arg_ptr;
    int idx = arg->idx;

    double conv_out[16][VALID_SIZE][VALID_SIZE];
    double pool_out[16][POOL_SIZE][POOL_SIZE];
    double* flat_out = malloc(sizeof(double) * FC1_INPUT_SIZE);
    double* fc1_out = malloc(sizeof(double) * FC1_SIZE);
    double* fc2_out = malloc(sizeof(double) * FC2_SIZE);

    conv2d_forward(arg->conv_layer, input_streams[idx], conv_out);
    relu_forward(conv_out);
    maxpool2d_forward(arg->pool_layer, conv_out, pool_out);
    flatten_forward(pool_out, flat_out);

    pthread_mutex_lock(&fc_mutex);
    fc1_forward(arg->fc1_layer, flat_out, fc1_out);
    fc2_forward(arg->fc2_layer, fc1_out, fc2_out);
    pthread_mutex_unlock(&fc_mutex);

    printf("\n===== Finished input stream #%d =====\n", idx + 1);
    printf("Conv2D output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", conv_out[0][i][i]);
    printf("\nFC1 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc1_out[i]);
    printf("\nFC2 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc2_out[i]);
    printf("\n");

    free(flat_out);
    free(fc1_out);
    free(fc2_out);
    return NULL;
}

int main() {
    init_input_streams();
    pthread_mutex_init(&fc_mutex, NULL);

    Conv2DLayer conv_layer;
    conv_layer.in_channels = 3;
    conv_layer.out_channels = 16;
    conv_layer.kernel_size = 3;
    conv_layer.weights = malloc(sizeof(double***) * 16);
    for (int f = 0; f < 16; f++) {
        conv_layer.weights[f] = malloc(sizeof(double**) * 3);
        for (int c = 0; c < 3; c++) {
            conv_layer.weights[f][c] = malloc(sizeof(double*) * 3);
            for (int i = 0; i < 3; i++) {
                conv_layer.weights[f][c][i] = malloc(sizeof(double) * 3);
                for (int j = 0; j < 3; j++)
                    conv_layer.weights[f][c][i][j] = (i % 2 == 0) ? 0.5 : 0.0;
            }
        }
    }

    MaxPool2DLayer pool_layer = {2};

    FullyConnected1Layer fc1_layer;
    fc1_layer.input_size = FC1_INPUT_SIZE;
    fc1_layer.output_size = FC1_SIZE;
    fc1_layer.weights = malloc(sizeof(double*) * FC1_INPUT_SIZE);
    fc1_layer.bias = malloc(sizeof(double) * FC1_SIZE);
    for (int i = 0; i < FC1_INPUT_SIZE; i++) {
        fc1_layer.weights[i] = malloc(sizeof(double) * FC1_SIZE);
        for (int j = 0; j < FC1_SIZE; j++)
            fc1_layer.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
    }
    for (int j = 0; j < FC1_SIZE; j++)
        fc1_layer.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    FullyConnected2Layer fc2_layer;
    fc2_layer.input_size = FC1_SIZE;
    fc2_layer.output_size = FC2_SIZE;
    fc2_layer.weights = malloc(sizeof(double*) * FC1_SIZE);
    fc2_layer.bias = malloc(sizeof(double) * FC2_SIZE);
    for (int i = 0; i < FC1_SIZE; i++) {
        fc2_layer.weights[i] = malloc(sizeof(double) * FC2_SIZE);
        for (int j = 0; j < FC2_SIZE; j++)
            fc2_layer.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
    }
    for (int j = 0; j < FC2_SIZE; j++)
        fc2_layer.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    pthread_t threads[NUM_INPUTS];
    ThreadArg args[NUM_INPUTS];

    for (int i = 0; i < NUM_INPUTS; i++) {
        args[i].idx = i;
        args[i].conv_layer = &conv_layer;
        args[i].pool_layer = &pool_layer;
        args[i].fc1_layer = &fc1_layer;
        args[i].fc2_layer = &fc2_layer;
        pthread_create(&threads[i], NULL, thread_worker, &args[i]);
    }

    for (int i = 0; i < NUM_INPUTS; i++)
        pthread_join(threads[i], NULL);

    print_resource_usage();
    pthread_mutex_destroy(&fc_mutex);
    return 0;
}
