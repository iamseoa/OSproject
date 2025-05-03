#include "layers.h"
#include <sys/resource.h>
#include <pthread.h>
#include <unistd.h> // getpid()
#include <string.h> // memset

#define NUM_INPUTS 9

pthread_mutex_t print_mutex;

double input_streams[NUM_INPUTS][3][SIZE][SIZE] = {0};

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

// 입력 인덱스를 인자로 받기 위한 구조체
typedef struct {
    int idx;
    Conv2DLayer* conv_layer;
    MaxPool2DLayer* pool_layer;
    FullyConnected1Layer* fc1_layer;
    FullyConnected2Layer* fc2_layer;
} ThreadArgs;

void* process_input(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int idx = args->idx;
    pid_t pid = getpid();

    double (*conv_output)[VALID_SIZE][VALID_SIZE] = malloc(sizeof(double) * 16 * VALID_SIZE * VALID_SIZE);
    double (*pool_output)[POOL_SIZE][POOL_SIZE] = malloc(sizeof(double) * 16 * POOL_SIZE * POOL_SIZE);
    double* flatten_output = (double*)malloc(sizeof(double) * FC1_INPUT_SIZE);
    double* fc1_output = (double*)malloc(sizeof(double) * FC1_SIZE);
    double* fc2_output = (double*)malloc(sizeof(double) * FC2_SIZE);

    conv2d_forward(args->conv_layer, input_streams[idx], conv_output);
    relu_forward(conv_output);
    maxpool2d_forward(args->pool_layer, conv_output, pool_output);
    flatten_forward(pool_output, flatten_output);
    fc1_forward(args->fc1_layer, flatten_output, fc1_output);
    fc2_forward(args->fc2_layer, fc1_output, fc2_output);

    pthread_mutex_lock(&print_mutex);
    printf("\n===== [PID %d] Finished input stream #%d =====\n", pid, idx + 1);
    printf("Conv2D output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", conv_output[0][i][i]);
    printf("\n");
    printf("FC1 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc1_output[i]);
    printf("\n");
    printf("FC2 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc2_output[i]);
    printf("\n");
    pthread_mutex_unlock(&print_mutex);

    free(flatten_output);
    free(fc1_output);
    free(fc2_output);
    free(conv_output);
    free(pool_output);
    free(arg);
    pthread_exit(NULL);
}

int main() {
    init_input_streams();
    pthread_mutex_init(&print_mutex, NULL);

    Conv2DLayer conv_layer;
    conv_layer.in_channels = 3;
    conv_layer.out_channels = 16;
    conv_layer.kernel_size = 3;
    conv_layer.weights = (double****)malloc(sizeof(double***) * conv_layer.out_channels);
    for (int f = 0; f < conv_layer.out_channels; f++) {
        conv_layer.weights[f] = (double***)malloc(sizeof(double**) * conv_layer.in_channels);
        for (int c = 0; c < conv_layer.in_channels; c++) {
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

    pthread_t threads[NUM_INPUTS];

    for (int i = 0; i < NUM_INPUTS; i++) {
        ThreadArgs* args = malloc(sizeof(ThreadArgs));
        args->idx = i;
        args->conv_layer = &conv_layer;
        args->pool_layer = &pool_layer;
        args->fc1_layer = &fc1_layer;
        args->fc2_layer = &fc2_layer;
        pthread_create(&threads[i], NULL, process_input, args);
    }

    for (int i = 0; i < NUM_INPUTS; i++) {
        pthread_join(threads[i], NULL);
    }

    print_resource_usage();
    pthread_mutex_destroy(&print_mutex);

    for (int f = 0; f < conv_layer.out_channels; f++) {
        for (int c = 0; c < conv_layer.in_channels; c++) {
            for (int i = 0; i < 3; i++) {
                free(conv_layer.weights[f][c][i]);
            }
            free(conv_layer.weights[f][c]);
        }
        free(conv_layer.weights[f]);
    }
    free(conv_layer.weights);

    for (int i = 0; i < fc1_layer.input_size; i++)
        free(fc1_layer.weights[i]);
    free(fc1_layer.weights);
    free(fc1_layer.bias);

    for (int i = 0; i < fc2_layer.input_size; i++)
        free(fc2_layer.weights[i]);
    free(fc2_layer.weights);
    free(fc2_layer.bias);

    return 0;
}
