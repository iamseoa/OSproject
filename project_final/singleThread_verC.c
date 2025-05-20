#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <pthread.h>
#define gettid() syscall(SYS_gettid)

#define INPUT_SIZE 224
#define CHANNELS 3
#define KERNEL_SIZE 3
#define CONV_DEPTH 64
#define CONV_OUT (INPUT_SIZE - KERNEL_SIZE + 1)
#define FC1_OUT 256
#define FC2_OUT 100
#define NUM_INPUTS 8
#define NUM_PROCESS 4

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

CNNModel *model;
float (*inputs)[INPUT_SIZE][INPUT_SIZE][CHANNELS];
pthread_mutex_t *model_mutex;
double *conv_time, *relu_time, *pool_time, *fc1_time, *fc2_time;

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

void conv_forward_parallel(CNNModel* model, float input[INPUT_SIZE][INPUT_SIZE][CHANNELS], float output[CONV_OUT][CONV_OUT][CONV_DEPTH]) {
    int num_proc = NUM_PROCESS;
    int pid;
    for (int p = 0; p < num_proc; p++) {
        pid = fork();
        if (pid == 0) {
            int start = (CONV_DEPTH / num_proc) * p;
            int end = (p == num_proc - 1) ? CONV_DEPTH : (CONV_DEPTH / num_proc) * (p + 1);
            for (int d = start; d < end; d++) {
                for (int i = 0; i < CONV_OUT; i++) {
                    for (int j = 0; j < CONV_OUT; j++) {
                        float sum = model->conv.biases[d];
                        for (int c = 0; c < CHANNELS; c++) {
                            for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                                for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                                    sum += model->conv.weights[d][c][ki][kj] * input[i + ki][j + kj][c];
                                }
                            }
                        }
                        output[i][j][d] = sum;
                    }
                }
            }
            exit(0);
        }
    }
    for (int p = 0; p < num_proc; p++) wait(NULL);
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
                    for (int dj = 0; dj < 2; dj++) {
                        float val = in[i+di][j+dj][d];
                        if (val > maxval) maxval = val;
                    }
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

void fc1_forward_parallel(CNNModel* model, float input[], float output[FC1_OUT]) {
    int num_proc = NUM_PROCESS;
    int pid;
    for (int p = 0; p < num_proc; p++) {
        pid = fork();
        if (pid == 0) {
            int start = (FC1_OUT / num_proc) * p;
            int end = (p == num_proc - 1) ? FC1_OUT : (FC1_OUT / num_proc) * (p + 1);
            for (int i = start; i < end; i++) {
                output[i] = model->fc1.biases[i];
                for (int j = 0; j < (CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH; j++) {
                    output[i] += model->fc1.weights[i][j] * input[j];
                }
            }
            exit(0);
        }
    }
    for (int p = 0; p < num_proc; p++) wait(NULL);
}

void fc2_forward(CNNModel* model, float input[], float output[FC2_OUT]) {
    for (int i = 0; i < FC2_OUT; i++) {
        output[i] = model->fc2.biases[i];
        for (int j = 0; j < FC1_OUT; j++)
            output[i] += model->fc2.weights[i][j] * input[j];
    }
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
    model_mutex = mmap(NULL, sizeof(pthread_mutex_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    pthread_mutex_t *print_mutex = mmap(NULL, sizeof(pthread_mutex_t), PROT_READ | PROT_WRITE,
                                        MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    conv_time = mmap(NULL, sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    relu_time = mmap(NULL, sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    pool_time = mmap(NULL, sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    fc1_time  = mmap(NULL, sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    fc2_time  = mmap(NULL, sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(model_mutex, &attr);
    pthread_mutex_init(print_mutex, &attr);

    initialize_weights(model);
    for (int i = 0; i < NUM_INPUTS; i++)
        initialize_input(inputs[i], i);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < NUM_INPUTS; i++) {
        float (*conv_out)[CONV_OUT][CONV_DEPTH] = mmap(NULL, sizeof(float) * CONV_OUT * CONV_OUT * CONV_DEPTH,
            PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        float (*relu_out)[CONV_OUT][CONV_DEPTH] = malloc(sizeof(float) * CONV_OUT * CONV_OUT * CONV_DEPTH);
        float (*pool_out)[CONV_OUT/2][CONV_DEPTH] = malloc(sizeof(float) * (CONV_OUT/2) * (CONV_OUT/2) * CONV_DEPTH);
        float *flat = malloc(sizeof(float) * (CONV_OUT/2) * (CONV_OUT/2) * CONV_DEPTH);
        float *fc1_out = mmap(NULL, sizeof(float) * FC1_OUT, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        float *fc2_out = malloc(sizeof(float) * FC2_OUT);

        struct timeval t0, t1;
        double ct, rt, pt, f1t, f2t;

        // Conv
        gettimeofday(&t0, NULL);
        conv_forward_parallel(model, inputs[i], conv_out);  // 내부에서 여러 C# fork 발생
        gettimeofday(&t1, NULL);
        ct = timer_ms(t0, t1);

        // ReLU
        gettimeofday(&t0, NULL);
        relu_forward(conv_out, relu_out);
        gettimeofday(&t1, NULL);
        rt = timer_ms(t0, t1);

        // MaxPool
        gettimeofday(&t0, NULL);
        maxpool_forward(relu_out, pool_out);
        gettimeofday(&t1, NULL);
        pt = timer_ms(t0, t1);

        // Flatten
        flatten(pool_out, flat);

        // FC1
        gettimeofday(&t0, NULL);
        fc1_forward_parallel(model, flat, fc1_out);  // 내부에서 여러 C# fork 발생
        gettimeofday(&t1, NULL);
        f1t = timer_ms(t0, t1);

        // FC2
        gettimeofday(&t0, NULL);
        fc2_forward(model, fc1_out, fc2_out);
        gettimeofday(&t1, NULL);
        f2t = timer_ms(t0, t1);

        pthread_mutex_lock(model_mutex);
        *conv_time += ct;
        *relu_time += rt;
        *pool_time += pt;
        *fc1_time += f1t;
        *fc2_time += f2t;
        pthread_mutex_unlock(model_mutex);

        pthread_mutex_lock(print_mutex);
        printf("\n== Input %d (PID %d, TID %ld) ==\n", i, getpid(), gettid());
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

        printf("[Conv] [C0] %.3f ms\n", ct);
        printf("[FC1 ] [C0] %.3f ms\n", f1t);
        printf("[FC2 ] [C0] %.3f ms\n", f2t);
        pthread_mutex_unlock(print_mutex);

        munmap(conv_out, sizeof(float) * CONV_OUT * CONV_OUT * CONV_DEPTH);
        munmap(fc1_out, sizeof(float) * FC1_OUT);
        free(relu_out); free(pool_out); free(flat); free(fc2_out);
    }

    gettimeofday(&end, NULL);
    print_resource_usage(start, end, *conv_time, *relu_time, *pool_time, *fc1_time, *fc2_time);
    return 0;
}
