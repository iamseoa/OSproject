#include "layers.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/resource.h>
#include <sys/syscall.h>

#define NUM_INPUTS 9
#define THREAD_POOL_SIZE 3

static double input_streams[NUM_INPUTS][3][SIZE][SIZE];

sem_t print_sem;

typedef struct {
    Conv2DLayer* conv;
    MaxPool2DLayer* pool;
    FullyConnected1Layer* fc1;
    FullyConnected2Layer* fc2;
    int idx;
} ThreadArg;

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

void* worker_thread(void* arg) {
    ThreadArg* args = (ThreadArg*)arg;

    double (*conv_output)[VALID_SIZE][VALID_SIZE] = malloc(sizeof(double) * 16 * VALID_SIZE * VALID_SIZE);
    double (*pool_output)[POOL_SIZE][POOL_SIZE] = malloc(sizeof(double) * 16 * POOL_SIZE * POOL_SIZE);
    double* flatten_output = malloc(sizeof(double) * FC1_INPUT_SIZE);
    double* fc1_output = malloc(sizeof(double) * FC1_SIZE);
    double* fc2_output = malloc(sizeof(double) * FC2_SIZE);

    conv2d_forward(args->conv, input_streams[args->idx], conv_output);
    relu_forward(conv_output);
    maxpool2d_forward(args->pool, conv_output, pool_output);
    flatten_forward(pool_output, flatten_output);
    fc1_forward(args->fc1, flatten_output, fc1_output);
    fc2_forward(args->fc2, fc1_output, fc2_output);

    sem_wait(&print_sem);
    printf("\n===== Finished input stream #%d =====\n", args->idx + 1);
    printf("Conv2D output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", conv_output[0][i][i]);
    printf("\n");

    printf("FC1 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc1_output[i]);
    printf("\n");

    printf("FC2 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc2_output[i]);
    printf("\n");
    printf("[PID %d][TID %ld] finished.\n", getpid(), syscall(SYS_gettid));
    sem_post(&print_sem);

    free(conv_output);
    free(pool_output);
    free(flatten_output);
    free(fc1_output);
    free(fc2_output);
    free(arg);
    return NULL;
}

int main() {
    init_input_streams();

    Conv2DLayer conv_layer;
    conv_layer.in_channels = 3;
    conv_layer.out_channels = 16;
    conv_layer.kernel_size = 3;
    conv_layer.weights = malloc(sizeof(double***) * conv_layer.out_channels);
    for (int f = 0; f < conv_layer.out_channels; f++) {
        conv_layer.weights[f] = malloc(sizeof(double**) * conv_layer.in_channels);
        for (int c = 0; c < conv_layer.in_channels; c++) {
            conv_layer.weights[f][c] = malloc(sizeof(double*) * 3);
            for (int i = 0; i < 3; i++) {
                conv_layer.weights[f][c][i] = malloc(sizeof(double) * 3);
                for (int j = 0; j < 3; j++) {
                    conv_layer.weights[f][c][i][j] = (i % 2 == 0) ? 0.5 : 0.0;
                }
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
    sem_init(&print_sem, 0, 1);

    for (int i = 0; i < NUM_INPUTS; i++) {
        ThreadArg* args = malloc(sizeof(ThreadArg));
        args->conv = &conv_layer;
        args->pool = &pool_layer;
        args->fc1 = &fc1_layer;
        args->fc2 = &fc2_layer;
        args->idx = i;
        pthread_create(&threads[i], NULL, worker_thread, args);
    }
    for (int i = 0; i < NUM_INPUTS; i++)
        pthread_join(threads[i], NULL);

    sem_destroy(&print_sem);
    print_resource_usage();
    return 0;
}
