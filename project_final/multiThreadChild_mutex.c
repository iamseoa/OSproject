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
#define NUM_THREADS 4

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

typedef struct {
    int tid;
    double wall_time;
    double user_time;
    double sys_time;
    double utilization;
} ThreadStat;

typedef struct {
    CNNModel* model;
    float (*input)[INPUT_SIZE][CHANNELS];
    float (*output)[CONV_OUT][CONV_DEPTH];
    int from, to, tid;
    ThreadStat* stats;
} ConvArgs;

typedef struct {
    CNNModel* model;
    float* input;
    float* output;
    int from, to, tid;
    ThreadStat* stats;
} FCArgs;

CNNModel* model;
float (*inputs)[INPUT_SIZE][INPUT_SIZE][CHANNELS];
double *conv_time, *relu_time, *pool_time, *fc1_time, *fc2_time;
float (*conv_out)[CONV_OUT][CONV_DEPTH];
float (*relu_out)[CONV_OUT][CONV_DEPTH];
float ***pool_out;
float *flat;
float *fc1_out;
float *fc2_out;
pthread_mutex_t *mutex;

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
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    for (int d = arg->from; d < arg->to; d++)
        for (int i = 0; i < CONV_OUT; i++)
            for (int j = 0; j < CONV_OUT; j++) {
                float sum = arg->model->conv.biases[d];
                for (int c = 0; c < CHANNELS; c++)
                    for (int ki = 0; ki < KERNEL_SIZE; ki++)
                        for (int kj = 0; kj < KERNEL_SIZE; kj++)
                            sum += arg->model->conv.weights[d][c][ki][kj] * arg->input[i + ki][j + kj][c];
                arg->output[i][j][d] = sum;
            }
    gettimeofday(&t1, NULL);
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double user = usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0;
    double sys  = usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0;
    double wall = timer_ms(t0, t1);
    double util = (user + sys) / wall * 100.0;
    arg->stats[arg->tid] = (ThreadStat){arg->tid, wall, user, sys, util};
    return NULL;
}

void* fc1_thread(void* args) {
    FCArgs* arg = (FCArgs*) args;
    struct timeval t0, t1;
    gettimeofday(&t0, NULL);
    for (int i = arg->from; i < arg->to; i++) {
        arg->output[i] = arg->model->fc1.biases[i];
        for (int j = 0; j < (CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH; j++)
            arg->output[i] += arg->model->fc1.weights[i][j] * arg->input[j];
    }
    gettimeofday(&t1, NULL);
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double user = usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0;
    double sys  = usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0;
    double wall = timer_ms(t0, t1);
    double util = (user + sys) / wall * 100.0;
    arg->stats[arg->tid] = (ThreadStat){arg->tid, wall, user, sys, util};
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
    model = mmap(NULL, sizeof(CNNModel), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    inputs = mmap(NULL, sizeof(float) * NUM_INPUTS * INPUT_SIZE * INPUT_SIZE * CHANNELS,
                  PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    conv_time = mmap(NULL, sizeof(double) * NUM_INPUTS, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    relu_time = mmap(NULL, sizeof(double) * NUM_INPUTS, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    pool_time = mmap(NULL, sizeof(double) * NUM_INPUTS, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    fc1_time  = mmap(NULL, sizeof(double) * NUM_INPUTS, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    fc2_time  = mmap(NULL, sizeof(double) * NUM_INPUTS, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    conv_out = (float (*)[CONV_OUT][CONV_DEPTH]) mmap(NULL,
        sizeof(float) * CONV_OUT * CONV_OUT * CONV_DEPTH,
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    
    relu_out = (float (*)[CONV_OUT][CONV_DEPTH]) mmap(NULL,
        sizeof(float) * CONV_OUT * CONV_OUT * CONV_DEPTH,
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    
    float ***pool_out = malloc((CONV_OUT/2) * sizeof(float**));
    for (int i = 0; i < CONV_OUT/2; i++) {
        pool_out[i] = malloc((CONV_OUT/2) * sizeof(float*));
        for (int j = 0; j < CONV_OUT/2; j++)
            pool_out[i][j] = malloc(CONV_DEPTH * sizeof(float));
    }

    flat = (float *) mmap(NULL,
        sizeof(float) * (CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH,
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    
    fc1_out = (float *) mmap(NULL,
        sizeof(float) * FC1_OUT,
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    
    fc2_out = (float *) mmap(NULL,
        sizeof(float) * FC2_OUT,
        PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    pthread_mutex_t* mutex = mmap(NULL, sizeof(pthread_mutex_t),
                                  PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(mutex, &attr);

    initialize_weights(model);
    for (int i = 0; i < NUM_INPUTS; i++)
        initialize_input(inputs[i], i);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < NUM_INPUTS; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            pthread_mutex_lock(mutex);

            ThreadStat conv_stats[NUM_THREADS];
            ThreadStat fc1_stats[NUM_THREADS];

            struct timeval t0, t1;

            gettimeofday(&t0, NULL);
            pthread_t conv_threads[NUM_THREADS];
            ConvArgs conv_args[NUM_THREADS];
            for (int t = 0; t < NUM_THREADS; t++) {
                conv_args[t] = (ConvArgs){
                    .model = model,
                    .input = inputs[i],
                    .output = conv_out,
                    .from = t * (CONV_DEPTH / NUM_THREADS),
                    .to   = (t + 1) * (CONV_DEPTH / NUM_THREADS),
                    .tid  = t,
                    .stats = conv_stats
                };
                pthread_create(&conv_threads[t], NULL, conv_thread, &conv_args[t]);
            }
            for (int t = 0; t < NUM_THREADS; t++)
                pthread_join(conv_threads[t], NULL);
            gettimeofday(&t1, NULL);
            conv_time[i] = timer_ms(t0, t1);

            gettimeofday(&t0, NULL);
            for (int d = 0; d < CONV_DEPTH; d++)
                for (int x = 0; x < CONV_OUT; x++)
                    for (int y = 0; y < CONV_OUT; y++)
                        relu_out[x][y][d] = (conv_out[x][y][d] > 0) ? conv_out[x][y][d] : 0;
            gettimeofday(&t1, NULL);
            relu_time[i] = timer_ms(t0, t1);

            gettimeofday(&t0, NULL);
            for (int d = 0; d < CONV_DEPTH; d++) {
                for (int x = 0; x < CONV_OUT; x += 2) {
                    for (int y = 0; y < CONV_OUT; y += 2) {
                        float maxval = relu_out[x][y][d];
                        for (int dx = 0; dx < 2; dx++)
                            for (int dy = 0; dy < 2; dy++)
                                if (relu_out[x+dx][y+dy][d] > maxval)
                                    maxval = relu_out[x+dx][y+dy][d];

                        pool_out[x/2][y/2][d] = maxval;
                    }
                }
            }
            gettimeofday(&t1, NULL);
            pool_time[i] = timer_ms(t0, t1);

            gettimeofday(&t0, NULL);
            int idx_flat = 0;
            for (int d = 0; d < CONV_DEPTH; d++) {
                for (int x = 0; x < CONV_OUT/2; x++) {
                    for (int y = 0; y < CONV_OUT/2; y++) {
                        flat[idx_flat++] = pool_out[x][y][d];
                    }
                }
            }
            gettimeofday(&t1, NULL);

            gettimeofday(&t0, NULL);
            pthread_t fc1_threads[NUM_THREADS];
            FCArgs fc1_args[NUM_THREADS];
            for (int t = 0; t < NUM_THREADS; t++) {
                fc1_args[t] = (FCArgs){
                    .model = model,
                    .input = flat,
                    .output = fc1_out,
                    .from = t * (FC1_OUT / NUM_THREADS),
                    .to   = (t + 1) * (FC1_OUT / NUM_THREADS),
                    .tid  = t,
                    .stats = fc1_stats
                };
                pthread_create(&fc1_threads[t], NULL, fc1_thread, &fc1_args[t]);
            }
            for (int t = 0; t < NUM_THREADS; t++)
                pthread_join(fc1_threads[t], NULL);
            gettimeofday(&t1, NULL);
            fc1_time[i] = timer_ms(t0, t1);

            gettimeofday(&t0, NULL);
            for (int j = 0; j < FC2_OUT; j++) {
                fc2_out[j] = model->fc2.biases[j];
                for (int k = 0; k < FC1_OUT; k++)
                    fc2_out[j] += model->fc2.weights[j][k] * fc1_out[k];
            }
            gettimeofday(&t1, NULL);
            fc2_time[i] = timer_ms(t0, t1);

            struct rusage usage;
            getrusage(RUSAGE_SELF, &usage);
            double user_ms = usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0;
            double sys_ms  = usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0;
            double total_ms = conv_time[i] + relu_time[i] + pool_time[i] + fc1_time[i] + fc2_time[i];
            double cpu_util = (user_ms + sys_ms) / total_ms * 100.0;

            printf("\n== Input %d Child Process Resource Usage (PID %d, TID %ld) ==\n", i, getpid(), gettid());
            printf("User CPU Time      : %.3f ms\n", user_ms);
            printf("System CPU Time    : %.3f ms\n", sys_ms);
            printf("CPU Utilization    : %.2f %%\n", cpu_util);
            printf("Max RSS Memory     : %ld KB\n", usage.ru_maxrss);

            printf("Input Patch [0:3][0:3][0]:\n");
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++)
                    printf("%.1f ", inputs[i][x][y][0]);
                printf("\n");
            }

            printf("Conv Output [0:5][0][0] = ");
            for (int j = 0; j < 5; j++) printf("%.2f ", conv_out[j][0][0]);
            printf("\nfc1[0:5] = ");
            for (int j = 0; j < 5; j++) printf("%.2f ", fc1_out[j]);
            printf("\nfc2[0:5] = ");
            for (int j = 0; j < 5; j++) printf("%.2f ", fc2_out[j]);
            printf("\n");

            pthread_mutex_unlock(mutex);

            exit(0);
        }
    }

    for (int i = 0; i < NUM_INPUTS; i++)
        wait(NULL);
    gettimeofday(&end, NULL);

    double conv_sum = 0, relu_sum = 0, pool_sum = 0, fc1_sum = 0, fc2_sum = 0;
    for (int i = 0; i < NUM_INPUTS; i++) {
        conv_sum += conv_time[i];
        relu_sum += relu_time[i];
        pool_sum += pool_time[i];
        fc1_sum  += fc1_time[i];
        fc2_sum  += fc2_time[i];
    }

    print_resource_usage(start, end, conv_sum, relu_sum, pool_sum, fc1_sum, fc2_sum);
    return 0;
}


