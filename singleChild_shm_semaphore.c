#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <string.h>
#include "layers.h"

#define NUM_INPUTS 9

typedef struct {
    Conv2DLayer conv_layer;
    FullyConnected1Layer fc1_layer;
    FullyConnected2Layer fc2_layer;
    MaxPool2DLayer pool_layer;

    sem_t conv_sem;
    sem_t pool_sem;
    sem_t fc1_sem;
    sem_t fc2_sem;
} SharedLayers;

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

void* shared_alloc(size_t size) {
    void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap failed");
        exit(1);
    }
    return ptr;
}

void init_input(double input[3][SIZE][SIZE], int val) {
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                input[c][i][j] = (double)val;
}

void init_shared_layers(SharedLayers* shared) {
    sem_init(&shared->conv_sem, 1, 1);
    sem_init(&shared->pool_sem, 1, 1);
    sem_init(&shared->fc1_sem, 1, 1);
    sem_init(&shared->fc2_sem, 1, 1);

    shared->conv_layer.in_channels = 3;
    shared->conv_layer.out_channels = 16;
    shared->conv_layer.kernel_size = 3;
    shared->conv_layer.weights = shared_alloc(sizeof(double***) * 16);
    for (int f = 0; f < 16; f++) {
        shared->conv_layer.weights[f] = shared_alloc(sizeof(double**) * 3);
        for (int c = 0; c < 3; c++) {
            shared->conv_layer.weights[f][c] = shared_alloc(sizeof(double*) * 3);
            for (int i = 0; i < 3; i++) {
                shared->conv_layer.weights[f][c][i] = shared_alloc(sizeof(double) * 3);
                for (int j = 0; j < 3; j++)
                    shared->conv_layer.weights[f][c][i][j] = 0.5;
            }
        }
    }

    shared->fc1_layer.input_size = FC1_INPUT_SIZE;
    shared->fc1_layer.output_size = FC1_SIZE;
    shared->fc1_layer.weights = shared_alloc(sizeof(double*) * FC1_INPUT_SIZE);
    for (int i = 0; i < FC1_INPUT_SIZE; i++) {
        shared->fc1_layer.weights[i] = shared_alloc(sizeof(double) * FC1_SIZE);
        for (int j = 0; j < FC1_SIZE; j++)
            shared->fc1_layer.weights[i][j] = 0.5;
    }
    shared->fc1_layer.bias = shared_alloc(sizeof(double) * FC1_SIZE);
    for (int j = 0; j < FC1_SIZE; j++)
        shared->fc1_layer.bias[j] = 0.5;

    shared->fc2_layer.input_size = FC1_SIZE;
    shared->fc2_layer.output_size = FC2_SIZE;
    shared->fc2_layer.weights = shared_alloc(sizeof(double*) * FC1_SIZE);
    for (int i = 0; i < FC1_SIZE; i++) {
        shared->fc2_layer.weights[i] = shared_alloc(sizeof(double) * FC2_SIZE);
        for (int j = 0; j < FC2_SIZE; j++)
            shared->fc2_layer.weights[i][j] = 0.5;
    }
    shared->fc2_layer.bias = shared_alloc(sizeof(double) * FC2_SIZE);
    for (int j = 0; j < FC2_SIZE; j++)
        shared->fc2_layer.bias[j] = 0.5;

    shared->pool_layer.pool_size = 2;
}

int main() {
    SharedLayers* shared = shared_alloc(sizeof(SharedLayers));
    init_shared_layers(shared);

    double input[3][SIZE][SIZE];
    double conv_output[16][VALID_SIZE][VALID_SIZE];
    double pool_output[16][POOL_SIZE][POOL_SIZE];
    double* flatten_output = malloc(sizeof(double) * FC1_INPUT_SIZE);
    double* fc1_output = malloc(sizeof(double) * FC1_SIZE);
    double* fc2_output = malloc(sizeof(double) * FC2_SIZE);

    for (int idx = 0; idx < NUM_INPUTS; idx++) {
        init_input(input, idx + 1);

        sem_wait(&shared->conv_sem);
        conv2d_forward(&shared->conv_layer, input, conv_output);
        relu_forward(conv_output);
        sem_post(&shared->conv_sem);

        sem_wait(&shared->pool_sem);
        maxpool2d_forward(&shared->pool_layer, conv_output, pool_output);
        flatten_forward(pool_output, flatten_output);
        sem_post(&shared->pool_sem);

        sem_wait(&shared->fc1_sem);
        fc1_forward(&shared->fc1_layer, flatten_output, fc1_output);
        sem_post(&shared->fc1_sem);

        sem_wait(&shared->fc2_sem);
        fc2_forward(&shared->fc2_layer, fc1_output, fc2_output);
        sem_post(&shared->fc2_sem);

        printf("===== Finished input stream #%d =====\n", idx + 1);

        printf("Conv2D output sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", conv_output[0][0][i]);
        printf("\n");

        printf("FC1 output sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", fc1_output[i]);
        printf("\n");

        printf("FC2 output sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", fc2_output[i]);
        printf("\n\n");
    }

    print_resource_usage();

    free(flatten_output);
    free(fc1_output);
    free(fc2_output);

    return 0;
}
