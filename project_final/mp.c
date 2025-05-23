
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#define gettid() syscall(SYS_gettid)

#define INPUT_SIZE 224
#define CHANNELS 3
#define KERNEL_SIZE 3
#define CONV_DEPTH 64
#define CONV_OUT (INPUT_SIZE - KERNEL_SIZE + 1)
#define FC1_OUT 256
#define FC2_OUT 100
#define NUM_INPUTS 8
#define NUM_PROCESSES 4
#define QUEUE_SIZE 2


typedef struct {
    float conv_out[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float relu_out[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float pool_out[CONV_OUT/2][CONV_OUT/2][CONV_DEPTH];
    float flat[(CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH];
    float fc1_out[FC1_OUT];
    float fc2_out[FC2_OUT];
    float input[INPUT_SIZE][INPUT_SIZE][CHANNELS];
    int input_id;
} Task;

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
    Task* buffer[QUEUE_SIZE];
    int front, rear, count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty, not_full;
} TaskQueue;

CNNModel* model;
Task* task_pool;
TaskQueue* queue;
int* task_done_count;
int* producer_finished;

void initialize_weights(CNNModel* model) {
    int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

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
void initialize_input(Task* t, int id) {
    float center = 9.0f * (id + 1);
    t->input_id = id;
    for (int c = 0; c < CHANNELS; c++)
        for (int i = 0; i < INPUT_SIZE; i++)
            for (int j = 0; j < INPUT_SIZE; j++)
                t->input[i][j][c] = (i == 1 && j == 1) ? center : 1.0f;
}

void conv_relu_pool_fc(Task* t) {
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT; i++)
            for (int j = 0; j < CONV_OUT; j++) {
                float sum = model->conv.biases[d];
                for (int c = 0; c < CHANNELS; c++)
                    for (int ki = 0; ki < KERNEL_SIZE; ki++)
                        for (int kj = 0; kj < KERNEL_SIZE; kj++)
                            sum += model->conv.weights[d][c][ki][kj] * t->input[i + ki][j + kj][c];
                t->conv_out[i][j][d] = sum;
                t->relu_out[i][j][d] = (sum > 0) ? sum : 0;
            }
    int idx = 0;
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int x = 0; x < CONV_OUT; x += 2)
            for (int y = 0; y < CONV_OUT; y += 2) {
                float maxval = t->relu_out[x][y][d];
                for (int dx = 0; dx < 2; dx++)
                    for (int dy = 0; dy < 2; dy++)
                        if (t->relu_out[x + dx][y + dy][d] > maxval)
                            maxval = t->relu_out[x + dx][y + dy][d];
                t->pool_out[x/2][y/2][d] = maxval;
                t->flat[idx++] = maxval;
            }
    for (int i = 0; i < FC1_OUT; i++) {
        float sum = model->fc1.biases[i];
        for (int j = 0; j < idx; j++)
            sum += model->fc1.weights[i][j] * t->flat[j];
        t->fc1_out[i] = sum;
    }
    for (int i = 0; i < FC2_OUT; i++) {
        float sum = model->fc2.biases[i];
        for (int j = 0; j < FC1_OUT; j++)
            sum += model->fc2.weights[i][j] * t->fc1_out[j];
        t->fc2_out[i] = sum;
    }
}

void producer() {
    for (int i = 0; i < NUM_INPUTS; i++) {
        Task* t = &task_pool[i];
        initialize_input(t, i);
        while (queue->count == QUEUE_SIZE);
        queue->buffer[queue->rear] = t;
        queue->rear = (queue->rear + 1) % QUEUE_SIZE;
        queue->count++;
    }
    *producer_finished = 1;
}

void consumer_loop() {
    while (1) {
        if (*producer_finished && queue->count == 0) return;
        if (queue->count == 0) continue;
        Task* t = queue->buffer[queue->front];
        queue->front = (queue->front + 1) % QUEUE_SIZE;
        queue->count--;

        struct rusage usage_start, usage_end;
        struct timespec wall_start, wall_end;
        getrusage(RUSAGE_SELF, &usage_start);
        clock_gettime(CLOCK_MONOTONIC, &wall_start);

        conv_relu_pool_fc(t);

        clock_gettime(CLOCK_MONOTONIC, &wall_end);
        getrusage(RUSAGE_SELF, &usage_end);

        double user_usec = (usage_end.ru_utime.tv_sec - usage_start.ru_utime.tv_sec) * 1e6 +
                           (usage_end.ru_utime.tv_usec - usage_start.ru_utime.tv_usec);
        double sys_usec = (usage_end.ru_stime.tv_sec - usage_start.ru_stime.tv_sec) * 1e6 +
                          (usage_end.ru_stime.tv_usec - usage_start.ru_stime.tv_usec);
        double wall_msec = (wall_end.tv_sec - wall_start.tv_sec) * 1e3 +
                           (wall_end.tv_nsec - wall_start.tv_nsec) / 1e6;
        double cpu_util = 100.0 * (user_usec + sys_usec) / 1000.0 / wall_msec;

        printf("[Consumer %d] Input ID: %d\n", getpid(), t->input_id);
        printf("Input Patch [0:3][0:3][0]:\n");
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++)
                printf("%.1f ", t->input[x][y][0]);
            printf("\n");
        }
        printf("Conv Output [0][0][0] = %.2f\n", t->conv_out[0][0][0]);
        printf("fc1[0:5] = ");
        for (int j = 0; j < 5; j++) printf("%.2f ", t->fc1_out[j]);
        printf("\nfc2[0:5] = ");
        for (int j = 0; j < 5; j++) printf("%.2f ", t->fc2_out[j]);
        printf("\n\n");

        (*task_done_count)++;
    }
}

int main() {
    model = mmap(NULL, sizeof(CNNModel), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    task_pool = mmap(NULL, sizeof(Task) * NUM_INPUTS, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    queue = mmap(NULL, sizeof(TaskQueue), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    task_done_count = mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    producer_finished = mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    *task_done_count = 0;
    *producer_finished = 0;
    queue->front = queue->rear = queue->count = 0;
    initialize_weights(model);

    pid_t pid = fork();
    if (pid == 0) {
        producer();
        exit(0);
    }

    for (int i = 0; i < NUM_PROCESSES; i++) {
        pid_t cpid = fork();
        if (cpid == 0) {
            consumer_loop();
            exit(0);
        }
    }

    wait(NULL); // wait for producer
    for (int i = 0; i < NUM_PROCESSES; i++) wait(NULL); // wait for consumers

    printf("\n== Done ==\nTotal tasks done: %d\n", *task_done_count);
    return 0;
}
