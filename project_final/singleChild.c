#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <string.h>
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
    float conv_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float relu_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float pooled_output[CONV_OUT / 2][CONV_OUT / 2][CONV_DEPTH];
    float fc1_output[FC1_OUT];
    float fc2_output[FC2_OUT];
} CNNModel;

CNNModel *model;
float inputs[NUM_INPUTS][INPUT_SIZE][INPUT_SIZE][CHANNELS];

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

void conv_forward(CNNModel* model, float input[INPUT_SIZE][INPUT_SIZE][CHANNELS], float output[CONV_OUT][CONV_OUT][CONV_DEPTH]) {
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

void fc1_forward(CNNModel* model, float input[], float output[FC1_OUT]) {
    for (int i = 0; i < FC1_OUT; i++) {
        output[i] = model->fc1.biases[i];
        for (int j = 0; j < (CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH; j++)
            output[i] += model->fc1.weights[i][j] * input[j];
    }
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
    printf("Voluntary Context Switches   : %ld\n", usage_self.ru_nvcsw);
    printf("Involuntary Context Switches : %ld\n", usage_self.ru_nivcsw);
    
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
    if (model == MAP_FAILED) {
        perror("mmap failed for model");
        exit(1);
    }

    initialize_weights(model);
    for (int i = 0; i < NUM_INPUTS; i++)
        initialize_input(inputs[i], i);

    double *conv_time = mmap(NULL, sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    double *relu_time = mmap(NULL, sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    double *pool_time = mmap(NULL, sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    double *fc1_time  = mmap(NULL, sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    double *fc2_time  = mmap(NULL, sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < NUM_INPUTS; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            float flat[(CONV_OUT / 2) * (CONV_OUT / 2) * CONV_DEPTH];
            struct timeval t0, t1;
            double ct, rt, pt, f1t, f2t;

            gettimeofday(&t0, NULL);
            conv_forward(model, inputs[i], model->conv_output);
            gettimeofday(&t1, NULL);
            ct = timer_ms(t0, t1);
            *conv_time += ct;

            gettimeofday(&t0, NULL);
            relu_forward(model->conv_output, model->relu_output);
            gettimeofday(&t1, NULL);
            rt = timer_ms(t0, t1);
            *relu_time += rt;

            gettimeofday(&t0, NULL);
            maxpool_forward(model->relu_output, model->pooled_output);
            gettimeofday(&t1, NULL);
            pt = timer_ms(t0, t1);
            *pool_time += pt;

            flatten(model->pooled_output, flat);

            gettimeofday(&t0, NULL);
            fc1_forward(model, flat, model->fc1_output);
            gettimeofday(&t1, NULL);
            f1t = timer_ms(t0, t1);
            *fc1_time += f1t;

            gettimeofday(&t0, NULL);
            fc2_forward(model, model->fc1_output, model->fc2_output);
            gettimeofday(&t1, NULL);
            f2t = timer_ms(t0, t1);
            *fc2_time += f2t;

            struct rusage ru;
            getrusage(RUSAGE_SELF, &ru);
            double user_cpu = ru.ru_utime.tv_sec * 1000.0 + ru.ru_utime.tv_usec / 1000.0;
            double sys_cpu  = ru.ru_stime.tv_sec * 1000.0 + ru.ru_stime.tv_usec / 1000.0;
            double total_cpu = user_cpu + sys_cpu;
            double total_time = ct + rt + pt + f1t + f2t;
            double cpu_util = (total_time > 0.0) ? (total_cpu / total_time) * 100.0 : 0.0;

            printf("\n== Child Process Resource Usage (PID %d, TID %ld) ==\n", getpid(), gettid());
            printf("User CPU Time      : %.3f ms\n", user_cpu);
            printf("System CPU Time    : %.3f ms\n", sys_cpu);
            printf("CPU Utilization    : %.2f %%\n", cpu_util);
            printf("Max RSS Memory     : %ld KB\n", ru.ru_maxrss);

            printf("Input Patch [0:3][0:3][0]:\n");
            for (int x = 0; x < 3; x++) {
                for (int y = 0; y < 3; y++) {
                    printf("%.1f ", inputs[i][x][y][0]);
                }
                printf("\n");
            }

            printf("Conv Output [0:5][0][0] = ");
            for (int j = 0; j < 5; j++) printf("%.2f ", model->conv_output[j][0][0]);
            printf("\n");

            printf("fc1[0:5] = ");
            for (int j = 0; j < 5; j++) printf("%.2f ", model->fc1_output[j]);
            printf("\n");

            printf("fc2[0:5] = ");
            for (int j = 0; j < 5; j++) printf("%.2f ", model->fc2_output[j]);
            printf("\n");

            exit(0);
        }
        
        wait(NULL);
    }

    gettimeofday(&end, NULL);
    print_resource_usage(start, end, *conv_time, *relu_time, *pool_time, *fc1_time, *fc2_time);
    return 0;
}
