#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include "layers.h"

#define NUM_THREADS 9

//    CNN 레이어와 mutex 포함 구조체
typedef struct {
    Conv2DLayer conv;
    MaxPool2DLayer pool;
    FullyConnected1Layer fc1;
    FullyConnected2Layer fc2;
    pthread_mutex_t mutex_conv;
    pthread_mutex_t mutex_pool;
    pthread_mutex_t mutex_fc1;
    pthread_mutex_t mutex_fc2;
} SharedLayers;

// 각 스레드에 전달할 구조체
typedef struct {
    int id;
    SharedLayers* shared;
} ThreadArgs;

// 리소스 사용 현황 출력
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

// 스레드 함수
void* run_thread(void* arg) {
    ThreadArgs* args = (ThreadArgs*)arg;
    int id = args->id;
    SharedLayers* shared = args->shared;
    pid_t tid = syscall(SYS_gettid);

    double (*input)[SIZE][SIZE] = malloc(sizeof(double) * 3 * SIZE * SIZE);
    double (*conv_out)[VALID_SIZE][VALID_SIZE] = malloc(sizeof(double) * 16 * VALID_SIZE * VALID_SIZE);
    double (*pool_out)[POOL_SIZE][POOL_SIZE] = malloc(sizeof(double) * 16 * POOL_SIZE * POOL_SIZE);
    double* flatten = malloc(sizeof(double) * FC1_INPUT_SIZE);
    double* fc1_out = malloc(sizeof(double) * FC1_SIZE);
    double* fc2_out = malloc(sizeof(double) * FC2_SIZE);

    if (!input || !conv_out || !pool_out || !flatten || !fc1_out || !fc2_out) {
        fprintf(stderr, "[Thread %d] Memory allocation failed\n", id);
        pthread_exit(NULL);
    }

    for (int c = 0; c < 3; c++)
        for (int i = 0; i < SIZE; i++)
            for (int j = 0; j < SIZE; j++)
                input[c][i][j] = (double)(id);

    pthread_mutex_lock(&shared->mutex_conv);
    conv2d_forward(&shared->conv, input, conv_out);
    relu_forward(conv_out);
    pthread_mutex_unlock(&shared->mutex_conv);

    pthread_mutex_lock(&shared->mutex_pool);
    maxpool2d_forward(&shared->pool, conv_out, pool_out);
    flatten_forward(pool_out, flatten);
    pthread_mutex_unlock(&shared->mutex_pool);

    pthread_mutex_lock(&shared->mutex_fc1);
    fc1_forward(&shared->fc1, flatten, fc1_out);
    pthread_mutex_unlock(&shared->mutex_fc1);

    pthread_mutex_lock(&shared->mutex_fc2);
    fc2_forward(&shared->fc2, fc1_out, fc2_out);
    pthread_mutex_unlock(&shared->mutex_fc2);

    // 즉시 출력
    printf("===== [PID %d] [TID %d] Finished input stream #%d =====\n", getpid(), tid, id);
    printf("Conv2D output sample: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", conv_out[0][0][i]);
    printf("\nFC1 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", fc1_out[i]);
    printf("\nFC2 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.4f ", fc2_out[i]);
    printf("\n\n");

    free(input);
    free(conv_out);
    free(pool_out);
    free(flatten);
    free(fc1_out);
    free(fc2_out);

    pthread_exit(NULL);
}

// 레이어 초기화
void init_shared_layers(SharedLayers* s) {
    pthread_mutex_init(&s->mutex_conv, NULL);
    pthread_mutex_init(&s->mutex_pool, NULL);
    pthread_mutex_init(&s->mutex_fc1, NULL);
    pthread_mutex_init(&s->mutex_fc2, NULL);

    s->conv.in_channels = 3;
    s->conv.out_channels = 16;
    s->conv.kernel_size = 3;
    s->conv.weights = malloc(sizeof(double***) * 16);
    for (int f = 0; f < 16; f++) {
        s->conv.weights[f] = malloc(sizeof(double**) * 3);
        for (int c = 0; c < 3; c++) {
            s->conv.weights[f][c] = malloc(sizeof(double*) * 3);
            for (int i = 0; i < 3; i++) {
                s->conv.weights[f][c][i] = malloc(sizeof(double) * 3);
                for (int j = 0; j < 3; j++)
                    s->conv.weights[f][c][i][j] = (i % 2 == 0) ? 0.5 : 0.0;
            }
        }
    }

    s->pool.pool_size = 2;

    s->fc1.input_size = FC1_INPUT_SIZE;
    s->fc1.output_size = FC1_SIZE;
    s->fc1.weights = malloc(sizeof(double*) * FC1_INPUT_SIZE);
    for (int i = 0; i < FC1_INPUT_SIZE; i++) {
        s->fc1.weights[i] = malloc(sizeof(double) * FC1_SIZE);
        for (int j = 0; j < FC1_SIZE; j++)
            s->fc1.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
    }
    s->fc1.bias = malloc(sizeof(double) * FC1_SIZE);
    for (int j = 0; j < FC1_SIZE; j++) s->fc1.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    s->fc2.input_size = FC1_SIZE;
    s->fc2.output_size = FC2_SIZE;
    s->fc2.weights = malloc(sizeof(double*) * FC1_SIZE);
    for (int i = 0; i < FC1_SIZE; i++) {
        s->fc2.weights[i] = malloc(sizeof(double) * FC2_SIZE);
        for (int j = 0; j < FC2_SIZE; j++)
            s->fc2.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
    }
    s->fc2.bias = malloc(sizeof(double) * FC2_SIZE);
    for (int j = 0; j < FC2_SIZE; j++) s->fc2.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;
}

int main() {
    SharedLayers shared;
    init_shared_layers(&shared);

    pthread_t threads[NUM_THREADS];
    ThreadArgs* thread_args[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i] = malloc(sizeof(ThreadArgs));
        thread_args[i]->id = i + 1;
        thread_args[i]->shared = &shared;
        pthread_create(&threads[i], NULL, run_thread, thread_args[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
        free(thread_args[i]);
    }

    print_resource_usage();
    return 0;
}

