#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <semaphore.h>
#include "layers.h"

#define NUM_INPUTS 9

static double input_streams[NUM_INPUTS][3][SIZE][SIZE];
sem_t* print_sem;

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

    pid_t pid = getpid();
    long tid = syscall(SYS_gettid);

    printf("\n=== Resource Usage ===\n");
    printf("PID: %d | TID: %ld\n", pid, tid);
    printf("%-32s %ld KB\n",  "Max RSS (memory):", usage.ru_maxrss);
    printf("%-32s %ld\n",     "Voluntary Context Switches:", usage.ru_nvcsw);
    printf("%-32s %ld\n",     "Involuntary Context Switches:", usage.ru_nivcsw);
    printf("%-32s %ld\n",     "Minor Page Faults:", usage.ru_minflt);
    printf("%-32s %ld\n",     "Major Page Faults:", usage.ru_majflt);
}

void process_input(int idx) {
    Conv2DLayer conv_layer;
    conv_layer.in_channels = 3;
    conv_layer.out_channels = 16;
    conv_layer.kernel_size = 3;

    conv_layer.weights = (double****)malloc(sizeof(double***) * 16);
    for (int f = 0; f < 16; f++) {
        conv_layer.weights[f] = (double***)malloc(sizeof(double**) * 3);
        for (int c = 0; c < 3; c++) {
            conv_layer.weights[f][c] = (double**)malloc(sizeof(double*) * 3);
            for (int i = 0; i < 3; i++) {
                conv_layer.weights[f][c][i] = (double*)malloc(sizeof(double) * 3);
                for (int j = 0; j < 3; j++)
                    conv_layer.weights[f][c][i][j] = (i % 2 == 0) ? 0.5 : 0.0;
            }
        }
    }

    MaxPool2DLayer pool_layer = {2};

    FullyConnected1Layer fc1_layer;
    fc1_layer.input_size = FC1_INPUT_SIZE;
    fc1_layer.output_size = FC1_SIZE;
    fc1_layer.weights = (double**)malloc(sizeof(double*) * FC1_INPUT_SIZE);
    for (int i = 0; i < FC1_INPUT_SIZE; i++) {
        fc1_layer.weights[i] = (double*)malloc(sizeof(double) * FC1_SIZE);
        for (int j = 0; j < FC1_SIZE; j++)
            fc1_layer.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
    }
    fc1_layer.bias = (double*)malloc(sizeof(double) * FC1_SIZE);
    for (int j = 0; j < FC1_SIZE; j++)
        fc1_layer.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    FullyConnected2Layer fc2_layer;
    fc2_layer.input_size = FC1_SIZE;
    fc2_layer.output_size = FC2_SIZE;
    fc2_layer.weights = (double**)malloc(sizeof(double*) * FC1_SIZE);
    for (int i = 0; i < FC1_SIZE; i++) {
        fc2_layer.weights[i] = (double*)malloc(sizeof(double) * FC2_SIZE);
        for (int j = 0; j < FC2_SIZE; j++)
            fc2_layer.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
    }
    fc2_layer.bias = (double*)malloc(sizeof(double) * FC2_SIZE);
    for (int j = 0; j < FC2_SIZE; j++)
        fc2_layer.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    double (*conv_output)[VALID_SIZE][VALID_SIZE] = malloc(sizeof(double) * 16 * VALID_SIZE * VALID_SIZE);
    double (*pool_output)[POOL_SIZE][POOL_SIZE] = malloc(sizeof(double) * 16 * POOL_SIZE * POOL_SIZE);
    double* flatten_output = malloc(sizeof(double) * FC1_INPUT_SIZE);
    double* fc1_output = malloc(sizeof(double) * FC1_SIZE);
    double* fc2_output = malloc(sizeof(double) * FC2_SIZE);

    conv2d_forward(&conv_layer, input_streams[idx], conv_output);
    relu_forward(conv_output);
    maxpool2d_forward(&pool_layer, conv_output, pool_output);
    flatten_forward(pool_output, flatten_output);
    fc1_forward(&fc1_layer, flatten_output, fc1_output);
    fc2_forward(&fc2_layer, fc1_output, fc2_output);

    sem_wait(print_sem);
    printf("\n===== Finished input stream #%d =====\n", idx + 1);
    printf("Conv2D output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", conv_output[0][i][i]);
    printf("\nFC1 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc1_output[i]);
    printf("\nFC2 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc2_output[i]);
    printf("\n[PID %d][TID %ld] finished.\n", getpid(), syscall(SYS_gettid));
    sem_post(print_sem);

    free(flatten_output);
    free(fc1_output);
    free(fc2_output);
    free(conv_output);
    free(pool_output);
}

int main() {
    init_input_streams();

    print_sem = mmap(NULL, sizeof(sem_t), PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    sem_init(print_sem, 1, 1);

    for (int i = 0; i < NUM_INPUTS; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            process_input(i);
            exit(0);
        }
    }
    for (int i = 0; i < NUM_INPUTS; i++) wait(NULL);

    print_resource_usage();
    return 0;
}

