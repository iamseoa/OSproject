#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/syscall.h>

#define gettid() syscall(SYS_gettid)

#define INPUT_SIZE 224
#define CHANNELS 3
#define KERNEL_SIZE 3
#define CONV_DEPTH 64
#define CONV_OUT (INPUT_SIZE - KERNEL_SIZE + 1)
#define FC1_OUT 256
#define FC2_OUT 100
#define NUM_INPUTS 8
#define NUM_THREADS 4

typedef struct {
    float weights[CONV_DEPTH][CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    float biases[CONV_DEPTH];
} ConvLayer;

typedef struct {
    float weights[FC1_OUT][(CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH];
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
    float conv_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float relu_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float pooled_output[CONV_OUT/2][CONV_OUT/2][CONV_DEPTH];
    float fc1_output[FC1_OUT];
    float fc2_output[FC2_OUT];
} CNNModel;

CNNModel model;
float inputs[NUM_INPUTS][INPUT_SIZE][INPUT_SIZE][CHANNELS];

typedef struct {
    CNNModel* model;
    float (*input)[INPUT_SIZE][CHANNELS];
    float (*output)[CONV_OUT][CONV_DEPTH];
    int from, to;
    double* times;
    int tid;
} ConvArgs;

typedef struct {
    CNNModel* model;
    float* input;
    float* output;
    int from, to;
    double* times;
    int tid;
} FCArgs;

double timer_ms(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 +
           (end.tv_usec - start.tv_usec) / 1000.0;
}

void initialize_weights(CNNModel* model) {
    int kernel[3][3] = {{1,2,1},{2,4,2},{1,2,1}};
    for (int d = 0; d < CONV_DEPTH; d++) {
        model->conv.biases[d] = 1.0f;
        for (int c = 0; c < CHANNELS; c++)
            for (int i = 0; i < KERNEL_SIZE; i++)
                for (int j = 0; j < KERNEL_SIZE; j++)
                    model->conv.weights[d][c][i][j] = (float)kernel[i][j];
    }
    for (int i = 0; i < FC1_OUT; i++) {
        model->fc1.biases[i] = 1.0f;
        for (int j = 0; j < (CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH; j++)
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

void* conv_worker(void* args) {
    ConvArgs* arg = (ConvArgs*)args;
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    for (int d = arg->from; d < arg->to; d++)
        for (int i = 0; i < CONV_OUT; i++)
            for (int j = 0; j < CONV_OUT; j++) {
                float sum = arg->model->conv.biases[d];
                for (int c = 0; c < CHANNELS; c++)
                    for (int ki = 0; ki < KERNEL_SIZE; ki++)
                        for (int kj = 0; kj < KERNEL_SIZE; kj++)
                            sum += arg->model->conv.weights[d][c][ki][kj] * arg->input[i+ki][j+kj][c];
                arg->output[i][j][d] = sum;
            }
    gettimeofday(&t1, NULL);
    arg->times[arg->tid] = timer_ms(t0, t1);
    return NULL;
}

void* fc1_worker(void* args) {
    FCArgs* arg = (FCArgs*)args;
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    for (int i = arg->from; i < arg->to; i++) {
        arg->output[i] = arg->model->fc1.biases[i];
        for (int j = 0; j < (CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH; j++)
            arg->output[i] += arg->model->fc1.weights[i][j] * arg->input[j];
    }
    gettimeofday(&t1, NULL);
    arg->times[arg->tid] = timer_ms(t0, t1);
    return NULL;
}

void* fc2_worker(void* args) {
    FCArgs* arg = (FCArgs*)args;
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    for (int i = arg->from; i < arg->to; i++) {
        arg->output[i] = arg->model->fc2.biases[i];
        for (int j = 0; j < FC1_OUT; j++)
            arg->output[i] += arg->model->fc2.weights[i][j] * arg->input[j];
    }
    gettimeofday(&t1, NULL);
    arg->times[arg->tid] = timer_ms(t0, t1);
    return NULL;
}

void relu_forward(float in[CONV_OUT][CONV_OUT][CONV_DEPTH], float out[CONV_OUT][CONV_OUT][CONV_DEPTH]) {
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT; i++)
            for (int j = 0; j < CONV_OUT; j++)
                out[i][j][d] = (in[i][j][d] > 0) ? in[i][j][d] : 0;
}

void maxpool_forward(float in[CONV_OUT][CONV_OUT][CONV_DEPTH], float out[CONV_OUT/2][CONV_OUT/2][CONV_DEPTH]) {
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT; i += 2)
            for (int j = 0; j < CONV_OUT; j += 2) {
                float maxval = in[i][j][d];
                for (int di = 0; di < 2; di++)
                    for (int dj = 0; dj < 2; dj++)
                        if (in[i+di][j+dj][d] > maxval)
                            maxval = in[i+di][j+dj][d];
                out[i/2][j/2][d] = maxval;
            }
}

void flatten(float in[CONV_OUT/2][CONV_OUT/2][CONV_DEPTH], float out[]) {
    int idx = 0;
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT/2; i++)
            for (int j = 0; j < CONV_OUT/2; j++)
                out[idx++] = in[i][j][d];
}

void print_resource_usage(struct timeval start, struct timeval end,
                          double conv_t, double relu_t, double pool_t, double fc1_t, double fc2_t) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    double wall_time = timer_ms(start, end);
    double user_cpu = usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0;
    double sys_cpu  = usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0;
    double total_cpu = user_cpu + sys_cpu;

    printf("\n== Performance Metrics ==\n");
    printf("Wall Clock Time    : %.3f ms\n", wall_time);
    printf("User CPU Time      : %.3f ms\n", user_cpu);
    printf("System CPU Time    : %.3f ms\n", sys_cpu);
    printf("CPU Utilization    : %.2f %%\n", (total_cpu / wall_time) * 100.0);
    printf("Max RSS Memory     : %ld KB\n", usage.ru_maxrss);
    printf("\n== Layer-wise Time (sum for all inputs) ==\n");
    printf("Conv Layer   : %.3f ms\n", conv_t);
    printf("ReLU Layer   : %.3f ms\n", relu_t);
    printf("MaxPool      : %.3f ms\n", pool_t);
    printf("FC1 Layer    : %.3f ms\n", fc1_t);
    printf("FC2 Layer    : %.3f ms\n", fc2_t);
}

int main() {
    initialize_weights(&model);
    for (int i = 0; i < NUM_INPUTS; i++) initialize_input(inputs[i], i);

    struct timeval start, end, t0, t1;
    double conv_total = 0, relu_total = 0, pool_total = 0, fc1_total = 0, fc2_total = 0;

    gettimeofday(&start, NULL);

    for (int idx = 0; idx < NUM_INPUTS; idx++) {
        float flat[(CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH];
        double conv_times[NUM_THREADS] = {0}, fc1_times[NUM_THREADS] = {0}, fc2_times[NUM_THREADS] = {0};
        pthread_t conv_threads[NUM_THREADS], fc1_threads[NUM_THREADS], fc2_threads[NUM_THREADS];
        ConvArgs conv_args[NUM_THREADS];
        FCArgs fc1_args[NUM_THREADS], fc2_args[NUM_THREADS];

        gettimeofday(&t0, NULL);
        for (int t = 0; t < NUM_THREADS; t++) {
            conv_args[t] = (ConvArgs){&model, inputs[idx], model.conv_output,
                t*(CONV_DEPTH/NUM_THREADS), (t+1)*(CONV_DEPTH/NUM_THREADS), conv_times, t};
            pthread_create(&conv_threads[t], NULL, conv_worker, &conv_args[t]);
        }
        for (int t = 0; t < NUM_THREADS; t++) pthread_join(conv_threads[t], NULL);
        gettimeofday(&t1, NULL);
        conv_total += timer_ms(t0, t1);

        gettimeofday(&t0, NULL);
        relu_forward(model.conv_output, model.relu_output);
        gettimeofday(&t1, NULL);
        relu_total += timer_ms(t0, t1);

        gettimeofday(&t0, NULL);
        maxpool_forward(model.relu_output, model.pooled_output);
        gettimeofday(&t1, NULL);
        pool_total += timer_ms(t0, t1);

        flatten(model.pooled_output, flat);

        gettimeofday(&t0, NULL);
        for (int t = 0; t < NUM_THREADS; t++) {
            fc1_args[t] = (FCArgs){&model, flat, model.fc1_output,
                t*(FC1_OUT/NUM_THREADS), (t+1)*(FC1_OUT/NUM_THREADS), fc1_times, t};
            pthread_create(&fc1_threads[t], NULL, fc1_worker, &fc1_args[t]);
        }
        for (int t = 0; t < NUM_THREADS; t++) pthread_join(fc1_threads[t], NULL);
        gettimeofday(&t1, NULL);
        fc1_total += timer_ms(t0, t1);

        gettimeofday(&t0, NULL);
        for (int t = 0; t < NUM_THREADS; t++) {
            fc2_args[t] = (FCArgs){&model, model.fc1_output, model.fc2_output,
                t*(FC2_OUT/NUM_THREADS), (t+1)*(FC2_OUT/NUM_THREADS), fc2_times, t};
            pthread_create(&fc2_threads[t], NULL, fc2_worker, &fc2_args[t]);
        }
        for (int t = 0; t < NUM_THREADS; t++) pthread_join(fc2_threads[t], NULL);
        gettimeofday(&t1, NULL);
        fc2_total += timer_ms(t0, t1);

        printf("\n== Input %d (PID %d, TID %ld) ==\n", idx, getpid(), gettid());
        printf("Input Patch [0:3][0:3][0]:\n");
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) printf("%.1f ", inputs[idx][x][y][0]);
            printf("\n");
        }
        printf("Conv Output [0:5][0][0] = ");
        for (int j = 0; j < 5; j++) printf("%.2f ", model.conv_output[j][0][0]);
        printf("\nfc1[0:5] = ");
        for (int j = 0; j < 5; j++) printf("%.2f ", model.fc1_output[j]);
        printf("\nfc2[0:5] = ");
        for (int j = 0; j < 5; j++) printf("%.2f ", model.fc2_output[j]);

        printf("\n[Conv]");
        for (int t = 0; t < NUM_THREADS; t++) printf(" [T%d] %.3f ms", t, conv_times[t]);
        printf("\n[FC1 ]");
        for (int t = 0; t < NUM_THREADS; t++) printf(" [T%d] %.3f ms", t, fc1_times[t]);
        printf("\n[FC2 ]");
        for (int t = 0; t < NUM_THREADS; t++) printf(" [T%d] %.3f ms", t, fc2_times[t]);
        printf("\n");
    }

    gettimeofday(&end, NULL);
    print_resource_usage(start, end, conv_total, relu_total, pool_total, fc1_total, fc2_total);
    return 0;
}
