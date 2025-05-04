#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include "layers.h"

#define NUM_INPUTS 9

// 공유 자원
pthread_mutex_t *conv_mutex, *pool_mutex, *fc1_mutex, *fc2_mutex;
double (*input_streams)[3][SIZE][SIZE];
double ****conv_weights;
double **fc1_weights, *fc1_bias;
double **fc2_weights, *fc2_bias;

void* shared_alloc(size_t size) {
    void* ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap failed");
        exit(1);
    }
    return ptr;
}

void init_mutex(pthread_mutex_t *mutex) {
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(mutex, &attr);
}

void init_input_streams() {
    for (int n = 0; n < NUM_INPUTS; n++)
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    input_streams[n][c][i][j] = (double)(n + 1);
}

void init_weights() {
    // Conv
    conv_weights = shared_alloc(sizeof(double***) * 16);
    for (int f = 0; f < 16; f++) {
        conv_weights[f] = shared_alloc(sizeof(double**) * 3);
        for (int c = 0; c < 3; c++) {
            conv_weights[f][c] = shared_alloc(sizeof(double*) * 3);
            for (int i = 0; i < 3; i++) {
                conv_weights[f][c][i] = shared_alloc(sizeof(double) * 3);
                for (int j = 0; j < 3; j++)
                    conv_weights[f][c][i][j] = 0.5;
            }
        }
    }

    fc1_weights = shared_alloc(sizeof(double*) * FC1_INPUT_SIZE);
    for (int i = 0; i < FC1_INPUT_SIZE; i++) {
        fc1_weights[i] = shared_alloc(sizeof(double) * FC1_SIZE);
        for (int j = 0; j < FC1_SIZE; j++)
            fc1_weights[i][j] = 0.5;
    }
    fc1_bias = shared_alloc(sizeof(double) * FC1_SIZE);
    for (int j = 0; j < FC1_SIZE; j++) fc1_bias[j] = 0.5;

    fc2_weights = shared_alloc(sizeof(double*) * FC1_SIZE);
    for (int i = 0; i < FC1_SIZE; i++) {
        fc2_weights[i] = shared_alloc(sizeof(double) * FC2_SIZE);
        for (int j = 0; j < FC2_SIZE; j++)
            fc2_weights[i][j] = 0.5;
    }
    fc2_bias = shared_alloc(sizeof(double) * FC2_SIZE);
    for (int j = 0; j < FC2_SIZE; j++) fc2_bias[j] = 0.5;
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
    input_streams = shared_alloc(sizeof(double) * NUM_INPUTS * 3 * SIZE * SIZE);
    conv_mutex = shared_alloc(sizeof(pthread_mutex_t));
    pool_mutex = shared_alloc(sizeof(pthread_mutex_t));
    fc1_mutex  = shared_alloc(sizeof(pthread_mutex_t));
    fc2_mutex  = shared_alloc(sizeof(pthread_mutex_t));

    init_mutex(conv_mutex);
    init_mutex(pool_mutex);
    init_mutex(fc1_mutex);
    init_mutex(fc2_mutex);

    init_input_streams();
    init_weights();

    pid_t pid = fork();
    if (pid == 0) {
        Conv2DLayer conv_layer = {3, 16, 3, conv_weights};
        MaxPool2DLayer pool_layer = {2};
        FullyConnected1Layer fc1_layer = {FC1_INPUT_SIZE, FC1_SIZE, fc1_weights, fc1_bias};
        FullyConnected2Layer fc2_layer = {FC1_SIZE, FC2_SIZE, fc2_weights, fc2_bias};

        double (*conv_output)[VALID_SIZE][VALID_SIZE] = malloc(sizeof(double) * 16 * VALID_SIZE * VALID_SIZE);
        double (*pool_output)[POOL_SIZE][POOL_SIZE] = malloc(sizeof(double) * 16 * POOL_SIZE * POOL_SIZE);
        double* flatten_output = malloc(sizeof(double) * FC1_INPUT_SIZE);
        double* fc1_output = malloc(sizeof(double) * FC1_SIZE);
        double* fc2_output = malloc(sizeof(double) * FC2_SIZE);

        for (int idx = 0; idx < NUM_INPUTS; idx++) {
            pthread_mutex_lock(conv_mutex);
            conv2d_forward(&conv_layer, input_streams[idx], conv_output);
            relu_forward(conv_output);
            pthread_mutex_unlock(conv_mutex);

            pthread_mutex_lock(pool_mutex);
            maxpool2d_forward(&pool_layer, conv_output, pool_output);
            flatten_forward(pool_output, flatten_output);
            pthread_mutex_unlock(pool_mutex);

            pthread_mutex_lock(fc1_mutex);
            fc1_forward(&fc1_layer, flatten_output, fc1_output);
            pthread_mutex_unlock(fc1_mutex);

            pthread_mutex_lock(fc2_mutex);
            fc2_forward(&fc2_layer, fc1_output, fc2_output);
            pthread_mutex_unlock(fc2_mutex);

            printf("===== [PID %d] Finished input stream #%d =====\n", getpid(), idx + 1);
            printf("Conv2D output sample: ");
            for (int i = 0; i < 5; i++) printf("%.5f ", conv_output[0][0][i]);
            printf("\nFC1 output sample: ");
            for (int i = 0; i < 5; i++) printf("%.5f ", fc1_output[i]);
            printf("\nFC2 output sample: ");
            for (int i = 0; i < 5; i++) printf("%.5f ", fc2_output[i]);
            printf("\n\n");
        }

        print_resource_usage();
        exit(0);
    } else {
        wait(NULL);
    }

    return 0;
}
