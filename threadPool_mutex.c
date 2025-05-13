#include "layers.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/resource.h>

#define NUM_INPUTS 9
#define THREAD_POOL_SIZE 3

// 입력 스트림
static double input_streams[NUM_INPUTS][3][SIZE][SIZE];

// 출력 버퍼
static double conv_output_buffer[NUM_INPUTS][16][VALID_SIZE][VALID_SIZE];
static double pool_output_buffer[NUM_INPUTS][16][POOL_SIZE][POOL_SIZE];
static double flatten_buffer[NUM_INPUTS][FC1_INPUT_SIZE];
static double fc1_buffer[NUM_INPUTS][FC1_SIZE];
static double fc2_buffer[NUM_INPUTS][FC2_SIZE];

pthread_mutex_t task_mutex = PTHREAD_MUTEX_INITIALIZER;
int task_index = 0;

Conv2DLayer conv_layer;
MaxPool2DLayer pool_layer;
FullyConnected1Layer fc1_layer;
FullyConnected2Layer fc2_layer;

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

void init_model() {
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

    pool_layer.pool_size = 2;

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
}

void* worker(void* arg) {
    while (1) {
        pthread_mutex_lock(&task_mutex);
        int idx = task_index++;
        pthread_mutex_unlock(&task_mutex);

        if (idx >= NUM_INPUTS) break;

        conv2d_forward(&conv_layer, input_streams[idx], conv_output_buffer[idx]);
        relu_forward(conv_output_buffer[idx]);
        maxpool2d_forward(&pool_layer, conv_output_buffer[idx], pool_output_buffer[idx]);
        flatten_forward(pool_output_buffer[idx], flatten_buffer[idx]);
        fc1_forward(&fc1_layer, flatten_buffer[idx], fc1_buffer[idx]);
        fc2_forward(&fc2_layer, fc1_buffer[idx], fc2_buffer[idx]);
    }
    return NULL;
}

int main() {
    init_input_streams();
    init_model();

    pthread_t threads[THREAD_POOL_SIZE];
    for (int i = 0; i < THREAD_POOL_SIZE; i++)
        pthread_create(&threads[i], NULL, worker, NULL);

    for (int i = 0; i < THREAD_POOL_SIZE; i++)
        pthread_join(threads[i], NULL);

    for (int idx = 0; idx < NUM_INPUTS; idx++) {
        printf("\n===== Finished input stream #%d =====\n", idx + 1);
        printf("Conv2D output sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", conv_output_buffer[idx][0][i][i]);
        printf("\n");
        printf("FC1 output sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", fc1_buffer[idx][i]);
        printf("\n");
        printf("FC2 output sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", fc2_buffer[idx][i]);
        printf("\n");
    }

    print_resource_usage();
    return 0;
}

