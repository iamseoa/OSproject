#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <semaphore.h>
#include "layers.h"  // Conv2D, MaxPool2D, FullyConnected 정의 포함

#define NUM_STREAMS 9
#define NUM_CHILDREN 4

sem_t sem;

void process_stream(int stream_id, double input[3][SIZE][SIZE], Conv2DLayer* conv_layer, MaxPool2DLayer* pool_layer, FullyConnected1Layer* fc1_layer, FullyConnected2Layer* fc2_layer) {
    double conv_output[16][VALID_SIZE][VALID_SIZE] = {0};
    double pool_output[16][POOL_SIZE][POOL_SIZE] = {0};
    double *flatten_output = (double*)malloc(sizeof(double) * FC1_INPUT_SIZE);
    double *fc1_output = (double*)malloc(sizeof(double) * FC1_SIZE);
    double *fc2_output = (double*)malloc(sizeof(double) * FC2_SIZE);

    conv2d_forward(conv_layer, input, conv_output);
    relu_forward(conv_output);
    maxpool2d_forward(pool_layer, conv_output, pool_output);
    flatten_forward(pool_output, flatten_output);
    fc1_forward(fc1_layer, flatten_output, fc1_output);
    fc2_forward(fc2_layer, fc1_output, fc2_output);

    sem_wait(&sem);
    printf("\n===== Finished input stream #%d (Child PID: %d) =====\n", stream_id, getpid());
    printf("Conv2D output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", conv_output[0][i][i]);
    printf("\n");

    printf("FC1 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc1_output[i]);
    printf("\n");

    printf("FC2 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc2_output[i]);
    printf("\n");
    sem_post(&sem);

    free(flatten_output);
    free(fc1_output);
    free(fc2_output);
}

int main() {
    double input_streams[NUM_STREAMS][3][SIZE][SIZE];
    for (int n = 0; n < NUM_STREAMS; n++)
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    input_streams[n][c][i][j] = (double)(n + 1);

    // 레이어 초기화
    Conv2DLayer conv_layer = {3, 16, 3, NULL};
    conv_layer.weights = (double****)malloc(sizeof(double***) * 16);
    for (int f = 0; f < 16; f++) {
        conv_layer.weights[f] = (double***)malloc(sizeof(double**) * 3);
        for (int c = 0; c < 3; c++) {
            conv_layer.weights[f][c] = (double**)malloc(sizeof(double*) * 3);
            for (int i = 0; i < 3; i++) {
                conv_layer.weights[f][c][i] = (double*)malloc(sizeof(double) * 3);
                for (int j = 0; j < 3; j++) {
                    conv_layer.weights[f][c][i][j] = (i % 2 == 0) ? 0.5 : 0.0;
                }
            }
        }
    }

    MaxPool2DLayer pool_layer = {2};

    FullyConnected1Layer fc1_layer = {FC1_INPUT_SIZE, FC1_SIZE, NULL, NULL};
    fc1_layer.weights = (double**)malloc(sizeof(double*) * FC1_INPUT_SIZE);
    for (int i = 0; i < FC1_INPUT_SIZE; i++) {
        fc1_layer.weights[i] = (double*)malloc(sizeof(double) * FC1_SIZE);
        for (int j = 0; j < FC1_SIZE; j++) {
            fc1_layer.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
        }
    }
    fc1_layer.bias = (double*)malloc(sizeof(double) * FC1_SIZE);
    for (int j = 0; j < FC1_SIZE; j++) {
        fc1_layer.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;
    }

    FullyConnected2Layer fc2_layer = {FC1_SIZE, FC2_SIZE, NULL, NULL};
    fc2_layer.weights = (double**)malloc(sizeof(double*) * FC1_SIZE);
    for (int i = 0; i < FC1_SIZE; i++) {
        fc2_layer.weights[i] = (double*)malloc(sizeof(double) * FC2_SIZE);
        for (int j = 0; j < FC2_SIZE; j++) {
            fc2_layer.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
        }
    }
    fc2_layer.bias = (double*)malloc(sizeof(double) * FC2_SIZE);
    for (int j = 0; j < FC2_SIZE; j++) {
        fc2_layer.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;
    }

    sem_init(&sem, 1, 1); // 1: 프로세스 간 공유 세마포어

    int current_stream = 0;
    pid_t children[NUM_CHILDREN];

    for (int c = 0; c < NUM_CHILDREN; c++) {
        pid_t pid = fork();
        if (pid == 0) {
            while (1) {
                int my_stream;
                sem_wait(&sem);
                if (current_stream < NUM_STREAMS) {
                    my_stream = current_stream;
                    current_stream++;
                    sem_post(&sem);
                } else {
                    sem_post(&sem);
                    break;
                }

                process_stream(my_stream + 1, input_streams[my_stream], &conv_layer, &pool_layer, &fc1_layer, &fc2_layer);
            }
            exit(0);
        } else {
            children[c] = pid;
        }
    }

    for (int c = 0; c < NUM_CHILDREN; c++) {
        waitpid(children[c], NULL, 0);
    }

    sem_destroy(&sem);

    return 0;
}
