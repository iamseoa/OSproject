#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#define gettid() syscall(SYS_gettid)

#define INPUT_SIZE 224
#define CHANNELS 3
#define KERNEL_SIZE 3
#define CONV_DEPTH 64
#define CONV_OUT (INPUT_SIZE - KERNEL_SIZE + 1)
#define FC1_OUT 256
#define FC2_OUT 100
#define NUM_INPUTS 8

pthread_mutex_t* print_mutex;

typedef struct {
    float weights[CONV_DEPTH][CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    float biases[CONV_DEPTH];
} ConvLayer;

typedef struct {
    float weights[FC1_OUT][(CONV_OUT / 2) * (CONV_OUT / 2) * CONV_DEPTH];
    float biases[FC1_OUT];
} FullyConnectedLayer1;

typedef struct {
    float weights[FC2_OUT][FC1_OUT];
    float biases[FC2_OUT];
} FullyConnectedLayer2;

typedef struct {
    ConvLayer conv;
    FullyConnectedLayer1 fc1;
    FullyConnectedLayer2 fc2;
} CNNModel;

CNNModel* model;
float (*inputs)[INPUT_SIZE][INPUT_SIZE][CHANNELS];
double *conv_time, *relu_time, *pool_time, *fc1_time, *fc2_time;

typedef struct {
    CNNModel* model;
    float (*input)[INPUT_SIZE][CHANNELS];
    float (*output)[CONV_OUT][CONV_DEPTH];
} ConvArgs;

typedef struct {
    CNNModel* model;
    float* input;
    float* output;
} FCArgs;

double timer_ms(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
}

void initialize_weights(CNNModel* model) {
    int kernel[3][3] = {{1,2,1},{2,4,2},{1,2,1}};
    for (int d = 0; d < CONV_DEPTH; d++) {
        model->conv.biases[d] = 1.0f;
        for (int c = 0; c < CHANNELS; c++)
            for (int i = 0; i < KERNEL_SIZE; i++)
                for (int j = 0; j < KERNEL_SIZE; j++)
                    model->conv.weights[d][c][i][j] = kernel[i][j];
    }
    for (int i = 0; i < FC1_OUT; i++) {
        model->fc1.biases[i] = 1.0f;
        for (int j = 0; j < (CONV_OUT / 2)*(CONV_OUT / 2)*CONV_DEPTH; j++)
            model->fc1.weights[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    for (int i = 0; i < FC2_OUT; i++) {
        model->fc2.biases[i] = 1.0f;
        for (int j = 0; j < FC1_OUT; j++)
            model->fc2.weights[i][j] = (i == j) ? 1.0f : 0.0f;
    }
}

void initialize_input(float input[INPUT_SIZE][INPUT_SIZE][CHANNELS], int id) {
    float center = 9.0f * (id + 1);
    for (int c = 0; c < CHANNELS; c++)
        for (int i = 0; i < INPUT_SIZE; i++)
            for (int j = 0; j < INPUT_SIZE; j++)
                input[i][j][c] = (i == 1 && j == 1) ? center : 1.0f;
}

void* conv_thread(void* args) {
    ConvArgs* arg = (ConvArgs*) args;
    CNNModel* model = arg->model;
    float (*input)[INPUT_SIZE][CHANNELS] = arg->input;
    float (*output)[CONV_OUT][CONV_DEPTH] = arg->output;
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT; i++)
            for (int j = 0; j < CONV_OUT; j++) {
                float sum = model->conv.biases[d];
                for (int c = 0; c < CHANNELS; c++)
                    for (int ki = 0; ki < KERNEL_SIZE; ki++)
                        for (int kj = 0; kj < KERNEL_SIZE; kj++)
                            sum += model->conv.weights[d][c][ki][kj] * input[i + ki][j + kj][c];
                output[i][j][d] = sum;
            }
    return NULL;
}

void* fc1_thread(void* args) {
    FCArgs* arg = (FCArgs*) args;
    CNNModel* model = arg->model;
    float* input = arg->input;
    float* output = arg->output;
    for (int i = 0; i < FC1_OUT; i++) {
        output[i] = model->fc1.biases[i];
        for (int j = 0; j < (CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH; j++)
            output[i] += model->fc1.weights[i][j] * input[j];
    }
    return NULL;
}

void* fc2_thread(void* args) {
    FCArgs* arg = (FCArgs*) args;
    CNNModel* model = arg->model;
    float* input = arg->input;
    float* output = arg->output;
    for (int i = 0; i < FC2_OUT; i++) {
        output[i] = model->fc2.biases[i];
        for (int j = 0; j < FC1_OUT; j++)
            output[i] += model->fc2.weights[i][j] * input[j];
    }
    return NULL;
}

void print_resource_usage(struct timeval start, struct timeval end, double conv_t, double relu_t, double pool_t, double fc1_t, double fc2_t) {
    struct rusage usage_self;
    getrusage(RUSAGE_SELF, &usage_self);
    double wall_time = timer_ms(start, end);
    double user_cpu_self = usage_self.ru_utime.tv_sec * 1000.0 + usage_self.ru_utime.tv_usec / 1000.0;
    double sys_cpu_self  = usage_self.ru_stime.tv_sec * 1000.0 + usage_self.ru_stime.tv_usec / 1000.0;
    double total_self = user_cpu_self + sys_cpu_self;

    printf("\n== Performance Metrics ==\n");
    printf("Wall Clock Time    : %.3f ms\n", wall_time);
    printf("User CPU Time      : %.3f ms\n", user_cpu_self);
    printf("System CPU Time    : %.3f ms\n", sys_cpu_self);
    printf("CPU Utilization    : %.2f %%\n", (total_self / wall_time) * 100.0);
    printf("Max RSS Memory     : %ld KB\n", usage_self.ru_maxrss);
    printf("\n== Layer-wise Time (sum for all inputs) ==\n");
    printf("Conv Layer   : %.3f ms\n", conv_t);
    printf("ReLU Layer   : %.3f ms\n", relu_t);
    printf("MaxPool      : %.3f ms\n", pool_t);
    printf("FC1 Layer    : %.3f ms\n", fc1_t);
    printf("FC2 Layer    : %.3f ms\n", fc2_t);
}

int main() {
    model = mmap(NULL, sizeof(CNNModel), PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    inputs = mmap(NULL, sizeof(float) * NUM_INPUTS * INPUT_SIZE * INPUT_SIZE * CHANNELS,
                PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    print_mutex = mmap(NULL, sizeof(pthread_mutex_t),
                    PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(print_mutex, &attr);

    initialize_weights(model);
    for (int i = 0; i < NUM_INPUTS; i++)
        initialize_input(inputs[i], i);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < NUM_INPUTS; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            struct timeval t0, t1;
            double conv_time = 0, relu_time = 0, pool_time = 0, fc1_time = 0, fc2_time = 0;
 
            float (*conv_out)[CONV_OUT][CONV_DEPTH] = malloc(sizeof(float) * CONV_OUT * CONV_OUT * CONV_DEPTH);
            float (*relu_out)[CONV_OUT][CONV_DEPTH] = malloc(sizeof(float) * CONV_OUT * CONV_OUT * CONV_DEPTH);
            float (*pool_out)[CONV_OUT / 2][CONV_OUT / 2][CONV_DEPTH] = (float(*)[CONV_OUT / 2][CONV_OUT / 2][CONV_DEPTH]) 
                malloc(sizeof(float) * (CONV_OUT / 2) * (CONV_OUT / 2) * CONV_DEPTH);
            float *flat = malloc(sizeof(float) * (CONV_OUT / 2) * (CONV_OUT / 2) * CONV_DEPTH);
            float *fc1_out = malloc(sizeof(float) * FC1_OUT);
            float *fc2_out = malloc(sizeof(float) * FC2_OUT);

            gettimeofday(&t0, NULL);
            ConvArgs cargs = { model, inputs[i], conv_out };
            pthread_t conv_tid;
            pthread_create(&conv_tid, NULL, conv_thread, &cargs);
            pthread_join(conv_tid, NULL);
            gettimeofday(&t1, NULL);
            conv_time = timer_ms(t0, t1);

            gettimeofday(&t0, NULL);
            for (int d = 0; d < CONV_DEPTH; d++)
                for (int i2 = 0; i2 < CONV_OUT; i2++)
                    for (int j = 0; j < CONV_OUT; j++)
                        relu_out[i2][j][d] = (conv_out[i2][j][d] > 0) ? conv_out[i2][j][d] : 0;
            gettimeofday(&t1, NULL);
            relu_time = timer_ms(t0, t1);

            gettimeofday(&t0, NULL);
            for (int d = 0; d < CONV_DEPTH; d++)
                for (int i2 = 0; i2 < CONV_OUT; i2 += 2)
                    for (int j = 0; j < CONV_OUT; j += 2) {
                        float maxval = relu_out[i2][j][d];
                        for (int di = 0; di < 2; di++)
                            for (int dj = 0; dj < 2; dj++) {
                                float val = relu_out[i2 + di][j + dj][d];
                                if (val > maxval) maxval = val;
                            }
                        (*pool_out)[i2 / 2][j / 2][d] = maxval;
                    }
            gettimeofday(&t1, NULL);
            pool_time = timer_ms(t0, t1);

            int idx = 0;
            for (int d = 0; d < CONV_DEPTH; d++)
                for (int i2 = 0; i2 < CONV_OUT / 2; i2++)
                    for (int j = 0; j < CONV_OUT / 2; j++)
                        flat[idx++] = (*pool_out)[i2][j][d];

            gettimeofday(&t0, NULL);
            FCArgs f1args = { model, flat, fc1_out };
            pthread_t fc1_tid;
            pthread_create(&fc1_tid, NULL, fc1_thread, &f1args);
            pthread_join(fc1_tid, NULL);
            gettimeofday(&t1, NULL);
            fc1_time = timer_ms(t0, t1);

            gettimeofday(&t0, NULL);
            FCArgs f2args = { model, fc1_out, fc2_out };
            pthread_t fc2_tid;
            pthread_create(&fc2_tid, NULL, fc2_thread, &f2args);
            pthread_join(fc2_tid, NULL);
            gettimeofday(&t1, NULL);
            fc2_time = timer_ms(t0, t1);

            pthread_mutex_lock(print_mutex);
            printf("\n== Input %d (PID %d, TID %ld) ==\n", i, getpid(), gettid());
            printf("Conv Time: %.3f ms | ReLU: %.3f ms | Pool: %.3f ms | FC1: %.3f ms | FC2: %.3f ms\n",
                   conv_time, relu_time, pool_time, fc1_time, fc2_time);

            printf("Input Patch [0:3][0:3][0]:\n");
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) printf("%.1f ", inputs[i][x][y][0]);
                printf("\n");
            }
            printf("Conv Output [0:5][0][0] = ");
            for (int j = 0; j < 5; j++) printf("%.2f ", conv_out[j][0][0]);
            printf("\nfc1[0:5] = ");
            for (int j = 0; j < 5; j++) printf("%.2f ", fc1_out[j]);
            printf("\nfc2[0:5] = ");
            for (int j = 0; j < 5; j++) printf("%.2f ", fc2_out[j]);
            printf("\n");
            pthread_mutex_unlock(print_mutex);

            free(conv_out); free(relu_out); free(pool_out);
            free(flat); free(fc1_out); free(fc2_out);
            exit(0);
        }
    }

    for (int i = 0; i < NUM_INPUTS; i++) wait(NULL);

    gettimeofday(&end, NULL);
    print_resource_usage(start, end, 0, 0, 0, 0, 0);  // 개별 레이어 시간은 각 child에서 출력했으므로 0으로

    return 0;
}
