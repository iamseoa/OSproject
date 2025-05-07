#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <string.h>
#include "layers.h"

#define NUM_INPUTS 9
#define THREAD_POOL_SIZE 4

double input_streams[NUM_INPUTS][3][SIZE][SIZE] = {0};

Conv2DLayer* conv_layer;
MaxPool2DLayer* pool_layer;
FullyConnected1Layer* fc1_layer;
FullyConnected2Layer* fc2_layer;

pthread_mutex_t print_mutex = PTHREAD_MUTEX_INITIALIZER;
int task_index = 0;
pthread_mutex_t task_mutex = PTHREAD_MUTEX_INITIALIZER;

void init_input_streams() {
    for (int n = 0; n < NUM_INPUTS; n++)
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    input_streams[n][c][i][j] = (double)(n + 1);
}

void* checked_mmap(size_t size) {
    void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap failed");
        exit(1);
    }
    memset(ptr, 0, size);
    return ptr;
}

void init_shared_model() {
    // Conv Layer
    conv_layer = checked_mmap(sizeof(Conv2DLayer));
    conv_layer->in_channels = 3;
    conv_layer->out_channels = 16;
    conv_layer->kernel_size = 3;
    conv_layer->weights = checked_mmap(sizeof(double***) * 16);
    for (int f = 0; f < 16; f++) {
        conv_layer->weights[f] = checked_mmap(sizeof(double**) * 3);
        for (int c = 0; c < 3; c++) {
            conv_layer->weights[f][c] = checked_mmap(sizeof(double*) * 3);
            for (int i = 0; i < 3; i++) {
                conv_layer->weights[f][c][i] = checked_mmap(sizeof(double) * 3);
                for (int j = 0; j < 3; j++)
                    conv_layer->weights[f][c][i][j] = (i % 2 == 0) ? 0.5 : 0.0;
            }
        }
    }

    pool_layer = checked_mmap(sizeof(MaxPool2DLayer));
    pool_layer->pool_size = 2;

    fc1_layer = checked_mmap(sizeof(FullyConnected1Layer));
    fc1_layer->input_size = FC1_INPUT_SIZE;
    fc1_layer->output_size = FC1_SIZE;
    fc1_layer->weights = checked_mmap(sizeof(double*) * FC1_INPUT_SIZE);
    fc1_layer->bias = checked_mmap(sizeof(double) * FC1_SIZE);
    for (int i = 0; i < FC1_INPUT_SIZE; i++) {
        fc1_layer->weights[i] = checked_mmap(sizeof(double) * FC1_SIZE);
        for (int j = 0; j < FC1_SIZE; j++)
            fc1_layer->weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
    }
    for (int j = 0; j < FC1_SIZE; j++)
        fc1_layer->bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    fc2_layer = checked_mmap(sizeof(FullyConnected2Layer));
    fc2_layer->input_size = FC1_SIZE;
    fc2_layer->output_size = FC2_SIZE;
    fc2_layer->weights = checked_mmap(sizeof(double*) * FC1_SIZE);
    fc2_layer->bias = checked_mmap(sizeof(double) * FC2_SIZE);
    for (int i = 0; i < FC1_SIZE; i++) {
        fc2_layer->weights[i] = checked_mmap(sizeof(double) * FC2_SIZE);
        for (int j = 0; j < FC2_SIZE; j++)
            fc2_layer->weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
    }
    for (int j = 0; j < FC2_SIZE; j++)
        fc2_layer->bias[j] = (j % 2 == 0) ? 0.5 : 0.0;
}

void* worker(void* arg) {
    while (1) {
        pthread_mutex_lock(&task_mutex);
        int idx = task_index++;
        pthread_mutex_unlock(&task_mutex);
        if (idx >= NUM_INPUTS) return NULL;

        double (*conv_output)[VALID_SIZE][VALID_SIZE] = malloc(sizeof(double) * 16 * VALID_SIZE * VALID_SIZE);
        double (*pool_output)[POOL_SIZE][POOL_SIZE] = malloc(sizeof(double) * 16 * POOL_SIZE * POOL_SIZE);
        double* flatten_output = malloc(sizeof(double) * FC1_INPUT_SIZE);
        double* fc1_output = malloc(sizeof(double) * FC1_SIZE);
        double* fc2_output = malloc(sizeof(double) * FC2_SIZE);

        if (!conv_output || !pool_output || !flatten_output || !fc1_output || !fc2_output) {
            fprintf(stderr, "Memory allocation failed in thread!\n");
            exit(1);
        }

        conv2d_forward(conv_layer, input_streams[idx], conv_output);
        relu_forward(conv_output);
        maxpool2d_forward(pool_layer, conv_output, pool_output);
        flatten_forward(pool_output, flatten_output);
        fc1_forward(fc1_layer, flatten_output, fc1_output);
        fc2_forward(fc2_layer, fc1_output, fc2_output);

        pthread_mutex_lock(&print_mutex);
        printf("\n===== Finished input stream #%d =====\n", idx + 1);
        printf("Conv2D output sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", conv_output[0][i][i]);
        printf("\nFC1 output sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", fc1_output[i]);
        printf("\nFC2 output sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", fc2_output[i]);
        printf("\n");
        pthread_mutex_unlock(&print_mutex);

        free(conv_output);
        free(pool_output);
        free(flatten_output);
        free(fc1_output);
        free(fc2_output);
    }
    return NULL;
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
    init_shared_model();

    pthread_t threads[THREAD_POOL_SIZE];
    for (int i = 0; i < THREAD_POOL_SIZE; i++)
        pthread_create(&threads[i], NULL, worker, NULL);

    for (int i = 0; i < THREAD_POOL_SIZE; i++)
        pthread_join(threads[i], NULL);

    print_resource_usage();
    return 0;
}
