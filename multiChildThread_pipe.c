



#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <semaphore.h>
#include <fcntl.h>
#include <string.h>
#include <sys/wait.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>

#define NUM_INPUTS 9
#define INPUT_SIZE 32
#define CHANNELS 3
#define KERNEL_SIZE 3
#define CONV_OUT 30
#define CONV_DEPTH 16
#define FC1_OUT 256
#define FC2_OUT 100
#define NUM_THREADS 4

typedef struct {
    float (*input)[INPUT_SIZE][CHANNELS];
    float conv_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float relu_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float pooled_output[CONV_OUT / 2][CONV_OUT / 2][CONV_DEPTH];
    float flat[(CONV_OUT / 2)*(CONV_OUT / 2)*CONV_DEPTH];
    float fc1_output[FC1_OUT];
    float fc2_output[FC2_OUT];
    int id;
} Sample;

typedef struct {
    float weights[CONV_DEPTH][CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    float biases[CONV_DEPTH];
} ConvLayer;

typedef struct {
    float weights[FC1_OUT][(CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH];
    float biases[FC1_OUT];
} FC1Layer;

typedef struct {
    float weights[FC2_OUT][FC1_OUT];
    float biases[FC2_OUT];
} FC2Layer;

typedef struct {
    ConvLayer conv;
    FC1Layer fc1;
    FC2Layer fc2;
} CNNModel;

typedef struct {
    int tid, from, to;
    CNNModel* model;
    float (*input)[INPUT_SIZE][CHANNELS];
    float (*output)[CONV_OUT][CONV_DEPTH];
} ConvArgs;

typedef struct {
    int tid, from, to;
    CNNModel* model;
    float* input;
    float* output;
} FCArgs;


// Add timer_ms definition at top of file
double timer_ms(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
}


// Dummy CNNModel initialization
void init_model(CNNModel* model) {
    for (int d = 0; d < CONV_DEPTH; d++) model->conv.biases[d] = 0.1f;
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int c = 0; c < CHANNELS; c++)
            for (int i = 0; i < KERNEL_SIZE; i++)
                for (int j = 0; j < KERNEL_SIZE; j++)
                    model->conv.weights[d][c][i][j] = 0.01f;

    for (int i = 0; i < FC1_OUT; i++) model->fc1.biases[i] = 0.2f;
    for (int i = 0; i < FC1_OUT; i++)
        for (int j = 0; j < (CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH; j++)
            model->fc1.weights[i][j] = 0.005f;

    for (int i = 0; i < FC2_OUT; i++) model->fc2.biases[i] = 0.1f;
    for (int i = 0; i < FC2_OUT; i++)
        for (int j = 0; j < FC1_OUT; j++)
            model->fc2.weights[i][j] = 0.002f;
}

void* conv_thread(void* args) {
    ConvArgs* arg = (ConvArgs*) args;
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
    return NULL;
}

void* fc1_thread(void* args) {
    FCArgs* arg = (FCArgs*) args;
    for (int i = arg->from; i < arg->to; i++) {
        arg->output[i] = arg->model->fc1.biases[i];
        for (int j = 0; j < (CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH; j++)
            arg->output[i] += arg->model->fc1.weights[i][j] * arg->input[j];
    }
    return NULL;
}


// 기타 레이어 연산
void relu_forward(float in[CONV_OUT][CONV_OUT][CONV_DEPTH], float out[CONV_OUT][CONV_OUT][CONV_DEPTH]) {
    for (int i = 0; i < CONV_OUT; i++)
        for (int j = 0; j < CONV_OUT; j++)
            for (int d = 0; d < CONV_DEPTH; d++)
                out[i][j][d] = in[i][j][d] > 0 ? in[i][j][d] : 0;
}

void maxpool_forward(float in[CONV_OUT][CONV_OUT][CONV_DEPTH], float out[CONV_OUT/2][CONV_OUT/2][CONV_DEPTH]) {
    for (int i = 0; i < CONV_OUT/2; i++)
        for (int j = 0; j < CONV_OUT/2; j++)
            for (int d = 0; d < CONV_DEPTH; d++)
                out[i][j][d] = in[i*2][j*2][d];
}

void flatten(float in[CONV_OUT/2][CONV_OUT/2][CONV_DEPTH], float* out) {
    int idx = 0;
    for (int i = 0; i < CONV_OUT/2; i++)
        for (int j = 0; j < CONV_OUT/2; j++)
            for (int d = 0; d < CONV_DEPTH; d++)
                out[idx++] = in[i][j][d];
}

void fc2_forward(CNNModel* model, float* in, float* out) {
    for (int i = 0; i < FC2_OUT; i++) {
        out[i] = model->fc2.biases[i];
        for (int j = 0; j < FC1_OUT; j++)
            out[i] += model->fc2.weights[i][j] * in[j];
    }
}

int main() {
    Sample* samples = mmap(NULL, sizeof(Sample) * NUM_INPUTS, PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    CNNModel* model = mmap(NULL, sizeof(CNNModel), PROT_READ | PROT_WRITE,
                           MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    sem_t* sem_conv_done = mmap(NULL, sizeof(sem_t) * NUM_INPUTS, PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    sem_t* sem_relu_done = mmap(NULL, sizeof(sem_t) * NUM_INPUTS, PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    sem_t* sem_pool_done = mmap(NULL, sizeof(sem_t) * NUM_INPUTS, PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    sem_t* sem_fc1_done  = mmap(NULL, sizeof(sem_t) * NUM_INPUTS, PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    init_model(model);

    for (int i = 0; i < NUM_INPUTS; i++) {
        samples[i].id = i;
        for (int x = 0; x < INPUT_SIZE; x++)
            for (int y = 0; y < INPUT_SIZE; y++)
                for (int c = 0; c < CHANNELS; c++)
                    samples[i].input[x][y][c] = 1.0f;
        sem_init(&sem_conv_done[i], 1, 0);
        sem_init(&sem_relu_done[i], 1, 0);
        sem_init(&sem_pool_done[i], 1, 0);
        sem_init(&sem_fc1_done[i],  1, 0);
    }

    // Conv Process
    if (fork() == 0) {
        for (int i = 0; i < NUM_INPUTS; i++) {
            pthread_t threads[NUM_THREADS];
            ConvArgs args[NUM_THREADS];
            int per = CONV_DEPTH / NUM_THREADS;
            for (int t = 0; t < NUM_THREADS; t++) {
                args[t] = (ConvArgs){t, t * per, (t+1)*per, model,
                                     samples[i].input, samples[i].conv_output};
                pthread_create(&threads[t], NULL, conv_thread, &args[t]);
            }
            for (int t = 0; t < NUM_THREADS; t++) pthread_join(threads[t], NULL);
            sem_post(&sem_conv_done[i]);
        }
        exit(0);
    }

    // ReLU Process
    if (fork() == 0) {
        for (int i = 0; i < NUM_INPUTS; i++) {
            sem_wait(&sem_conv_done[i]);
            relu_forward(samples[i].conv_output, samples[i].relu_output);
            sem_post(&sem_relu_done[i]);
        }
        exit(0);
    }

    // Pool Process
    if (fork() == 0) {
        for (int i = 0; i < NUM_INPUTS; i++) {
            sem_wait(&sem_relu_done[i]);
            maxpool_forward(samples[i].relu_output, samples[i].pooled_output);
            sem_post(&sem_pool_done[i]);
        }
        exit(0);
    }

    // FC1 Process
    if (fork() == 0) {
        for (int i = 0; i < NUM_INPUTS; i++) {
            sem_wait(&sem_pool_done[i]);
            flatten(samples[i].pooled_output, samples[i].flat);
            pthread_t threads[NUM_THREADS];
            FCArgs args[NUM_THREADS];
            int per = FC1_OUT / NUM_THREADS;
            for (int t = 0; t < NUM_THREADS; t++) {
                args[t] = (FCArgs){t, t * per, (t+1)*per, model,
                                   samples[i].flat, samples[i].fc1_output};
                pthread_create(&threads[t], NULL, fc1_thread, &args[t]);
            }
            for (int t = 0; t < NUM_THREADS; t++) pthread_join(threads[t], NULL);
            sem_post(&sem_fc1_done[i]);
        }
        exit(0);
    }

    // FC2 Process
    if (fork() == 0) {
        for (int i = 0; i < NUM_INPUTS; i++) {
            sem_wait(&sem_fc1_done[i]);
            fc2_forward(model, samples[i].fc1_output, samples[i].fc2_output);
            printf("[Input %d] fc2_output[0] = %.2f\n", i, samples[i].fc2_output[0]);
        }
        exit(0);
    }

    for (int i = 0; i < 5; i++) wait(NULL);
    return 0;
}


#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

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

void print_child_usage(int i, float inputs[NUM_INPUTS][INPUT_SIZE][INPUT_SIZE][CHANNELS]) {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    double user_cpu = ru.ru_utime.tv_sec * 1000.0 + ru.ru_utime.tv_usec / 1000.0;
    double sys_cpu  = ru.ru_stime.tv_sec * 1000.0 + ru.ru_stime.tv_usec / 1000.0;
    double cpu_util = user_cpu + sys_cpu;

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
}
