#include "layers.h"
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

#define NUM_INPUTS 9

void init_input_streams(double input_streams[NUM_INPUTS][3][SIZE][SIZE]) {
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
    // Shared memory setup
    int input_shmid = shmget(IPC_PRIVATE, sizeof(double) * NUM_INPUTS * 3 * SIZE * SIZE, IPC_CREAT | 0666);
    double (*input_streams)[3][SIZE][SIZE] = shmat(input_shmid, NULL, 0);
    init_input_streams(input_streams);

    int conv_weight_shmid = shmget(IPC_PRIVATE, sizeof(double) * 16 * 3 * 3 * 3, IPC_CREAT | 0666);
    double *conv_raw = shmat(conv_weight_shmid, NULL, 0);
    for (int f = 0; f < 16; f++)
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    conv_raw[f*27 + c*9 + i*3 + j] = (i % 2 == 0) ? 0.5 : 0.0;

    int fc1_weight_shmid = shmget(IPC_PRIVATE, sizeof(double) * FC1_INPUT_SIZE * FC1_SIZE, IPC_CREAT | 0666);
    double *fc1_raw = shmat(fc1_weight_shmid, NULL, 0);
    for (int i = 0; i < FC1_INPUT_SIZE; i++)
        for (int j = 0; j < FC1_SIZE; j++)
            fc1_raw[i * FC1_SIZE + j] = (i % 2 == 0) ? 0.5 : 0.0;

    int fc1_bias_shmid = shmget(IPC_PRIVATE, sizeof(double) * FC1_SIZE, IPC_CREAT | 0666);
    double *fc1_bias = shmat(fc1_bias_shmid, NULL, 0);
    for (int j = 0; j < FC1_SIZE; j++)
        fc1_bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    int fc2_weight_shmid = shmget(IPC_PRIVATE, sizeof(double) * FC1_SIZE * FC2_SIZE, IPC_CREAT | 0666);
    double *fc2_raw = shmat(fc2_weight_shmid, NULL, 0);
    for (int i = 0; i < FC1_SIZE; i++)
        for (int j = 0; j < FC2_SIZE; j++)
            fc2_raw[i * FC2_SIZE + j] = (i % 2 == 0) ? 0.5 : 0.0;

    int fc2_bias_shmid = shmget(IPC_PRIVATE, sizeof(double) * FC2_SIZE, IPC_CREAT | 0666);
    double *fc2_bias = shmat(fc2_bias_shmid, NULL, 0);
    for (int j = 0; j < FC2_SIZE; j++)
        fc2_bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    pid_t pid = fork();
    if (pid == 0) {
        // child process
        double ****conv_weights = malloc(sizeof(double***) * 16);
        for (int f = 0; f < 16; f++) {
            conv_weights[f] = malloc(sizeof(double**) * 3);
            for (int c = 0; c < 3; c++) {
                conv_weights[f][c] = malloc(sizeof(double*) * 3);
                for (int i = 0; i < 3; i++)
                    conv_weights[f][c][i] = &conv_raw[f*27 + c*9 + i*3];
            }
        }

        Conv2DLayer conv_layer = {3, 16, 3, conv_weights};
        MaxPool2DLayer pool_layer = {2};

        FullyConnected1Layer fc1_layer = {FC1_INPUT_SIZE, FC1_SIZE, NULL, fc1_bias};
        fc1_layer.weights = malloc(sizeof(double*) * FC1_INPUT_SIZE);
        for (int i = 0; i < FC1_INPUT_SIZE; i++)
            fc1_layer.weights[i] = &fc1_raw[i * FC1_SIZE];

        FullyConnected2Layer fc2_layer = {FC1_SIZE, FC2_SIZE, NULL, fc2_bias};
        fc2_layer.weights = malloc(sizeof(double*) * FC1_SIZE);
        for (int i = 0; i < FC1_SIZE; i++)
            fc2_layer.weights[i] = &fc2_raw[i * FC2_SIZE];

        double (*conv_output)[VALID_SIZE][VALID_SIZE] = malloc(sizeof(double) * 16 * VALID_SIZE * VALID_SIZE);
        double (*pool_output)[POOL_SIZE][POOL_SIZE] = malloc(sizeof(double) * 16 * POOL_SIZE * POOL_SIZE);
        double* flatten_output = malloc(sizeof(double) * FC1_INPUT_SIZE);
        double* fc1_output = malloc(sizeof(double) * FC1_SIZE);
        double* fc2_output = malloc(sizeof(double) * FC2_SIZE);

        for (int idx = 0; idx < NUM_INPUTS; idx++) {
            conv2d_forward(&conv_layer, input_streams[idx], conv_output);
            relu_forward(conv_output);
            maxpool2d_forward(&pool_layer, conv_output, pool_output);
            flatten_forward(pool_output, flatten_output);
            fc1_forward(&fc1_layer, flatten_output, fc1_output);
            fc2_forward(&fc2_layer, fc1_output, fc2_output);

            printf("\n===== [PID %d] Finished input stream #%d =====\n", getpid(), idx + 1);
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

        print_resource_usage();
        exit(0);
    } else {
        wait(NULL);
    }

    shmctl(input_shmid, IPC_RMID, NULL);
    shmctl(conv_weight_shmid, IPC_RMID, NULL);
    shmctl(fc1_weight_shmid, IPC_RMID, NULL);
    shmctl(fc1_bias_shmid, IPC_RMID, NULL);
    shmctl(fc2_weight_shmid, IPC_RMID, NULL);
    shmctl(fc2_bias_shmid, IPC_RMID, NULL);

    return 0;
}

