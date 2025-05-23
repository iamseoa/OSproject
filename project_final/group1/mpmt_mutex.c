#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>
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
#define NUM_INPUTS 40 
#define NUM_THREADS 2       
#define NUM_PROCESSES 2     
#define QUEUE_SIZE NUM_INPUTS

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
    float weights[FC1_OUT][(CONV_OUT / 2)*(CONV_OUT / 2)*CONV_DEPTH];
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
_Atomic int* producer_finished;
pthread_mutex_t* task_done_mutex;
pthread_mutex_t* print_mutex;

void initialize_weights(CNNModel* model) {
    int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
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
        for (int j = 0; j < idx; j++) sum += model->fc1.weights[i][j] * t->flat[j];
        t->fc1_out[i] = sum;
    }
    for (int i = 0; i < FC2_OUT; i++) {
        float sum = model->fc2.biases[i];
        for (int j = 0; j < FC1_OUT; j++) sum += model->fc2.weights[i][j] * t->fc1_out[j];
        t->fc2_out[i] = sum;
    }
}

void* producer(void* arg) {
    for (int i = 0; i < NUM_INPUTS; i++) {
        Task* t = &task_pool[i];
        initialize_input(t, i);
        pthread_mutex_lock(&queue->mutex);
        while (queue->count == QUEUE_SIZE)
            pthread_cond_wait(&queue->not_full, &queue->mutex);
        queue->buffer[queue->rear] = t;
        queue->rear = (queue->rear + 1) % QUEUE_SIZE;
        queue->count++;
        pthread_cond_signal(&queue->not_empty);
        pthread_mutex_unlock(&queue->mutex);
    }
    atomic_store(producer_finished, 1);
    pthread_cond_broadcast(&queue->not_empty);
    return NULL;
}

void* consumer(void* arg) {
    while (1) {
        pthread_mutex_lock(&queue->mutex);
        if (atomic_load(producer_finished) && queue->count == 0) {
            pthread_mutex_unlock(&queue->mutex);
            return NULL;
        }
        while (queue->count == 0) {
            if (atomic_load(producer_finished)) {
                pthread_mutex_unlock(&queue->mutex);
                return NULL;
            }
            pthread_cond_wait(&queue->not_empty, &queue->mutex);
        }
        Task* t = queue->buffer[queue->front];
        queue->front = (queue->front + 1) % QUEUE_SIZE;
        queue->count--;
        pthread_cond_signal(&queue->not_full);
        pthread_mutex_unlock(&queue->mutex);

        struct timespec main_start, main_end;
        struct rusage main_usage_start, main_usage_end;
    
        clock_gettime(CLOCK_MONOTONIC, &main_start);
        getrusage(RUSAGE_SELF, &main_usage_start);

        conv_relu_pool_fc(t);

        clock_gettime(CLOCK_MONOTONIC, &main_end);
        getrusage(RUSAGE_SELF, &main_usage_end);

        double user_usec = (main_usage_end.ru_utime.tv_sec - main_usage_start.ru_utime.tv_sec) * 1e6 +
                           (main_usage_end.ru_utime.tv_usec - main_usage_start.ru_utime.tv_usec);
        double sys_usec = (main_usage_end.ru_stime.tv_sec - main_usage_start.ru_stime.tv_sec) * 1e6 +
                          (main_usage_end.ru_stime.tv_usec - main_usage_start.ru_stime.tv_usec);
        double wall_msec = (main_end.tv_sec - main_start.tv_sec) * 1e3 +
                           (main_end.tv_nsec - main_start.tv_nsec) / 1e6;
        double cpu_util = 100.0 * (user_usec + sys_usec) / 1000.0 / wall_msec;

        pthread_mutex_lock(print_mutex);
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
        printf("\n");
        printf("== Resource Usage ==\n");
        printf("User CPU Time   : %.3f ms\n", user_usec / 1000.0);
        printf("System CPU Time : %.3f ms\n", sys_usec / 1000.0);
        printf("CPU Utilization : %.2f %%\n", cpu_util);
        printf("Wall Clock Time : %.3f ms\n\n", wall_msec);
        pthread_mutex_unlock(print_mutex);

        pthread_mutex_lock(task_done_mutex);
        (*task_done_count)++;
        pthread_mutex_unlock(task_done_mutex);
    }
}

int main() {
    model = mmap(NULL, sizeof(CNNModel), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    task_pool = mmap(NULL, sizeof(Task) * NUM_INPUTS, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    queue = mmap(NULL, sizeof(TaskQueue), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    task_done_count = mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    producer_finished = mmap(NULL, sizeof(_Atomic int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    task_done_mutex = mmap(NULL, sizeof(pthread_mutex_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    print_mutex = mmap(NULL, sizeof(pthread_mutex_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    *task_done_count = 0;
    atomic_store(producer_finished, 0);

    pthread_mutexattr_t mattr;
    pthread_mutexattr_init(&mattr);
    pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&queue->mutex, &mattr);
    pthread_mutex_init(task_done_mutex, &mattr);
    pthread_mutex_init(print_mutex, &mattr);

    pthread_condattr_t cattr;
    pthread_condattr_init(&cattr);
    pthread_condattr_setpshared(&cattr, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&queue->not_empty, &cattr);
    pthread_cond_init(&queue->not_full, &cattr);

    queue->front = queue->rear = queue->count = 0;
    initialize_weights(model);

    struct timespec wall_start, wall_end;
    struct rusage usage_self_start, usage_self_end, usage_child_end;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);
    getrusage(RUSAGE_SELF, &usage_self_start);

    pid_t producer_pid = fork();
    if (producer_pid == 0) {
        producer(NULL);
        exit(0);
    }

    pid_t workers[NUM_PROCESSES];
    for (int i = 0; i < NUM_PROCESSES; i++) {
        if ((workers[i] = fork()) == 0) {
            pthread_t threads[NUM_THREADS];
            for (int j = 0; j < NUM_THREADS; j++)
                pthread_create(&threads[j], NULL, consumer, NULL);
            for (int j = 0; j < NUM_THREADS; j++)
                pthread_join(threads[j], NULL);
            exit(0);
        }
    }

    waitpid(producer_pid, NULL, 0);
    for (int i = 0; i < NUM_PROCESSES; i++)
        waitpid(workers[i], NULL, 0);

    clock_gettime(CLOCK_MONOTONIC, &wall_end);
    getrusage(RUSAGE_SELF, &usage_self_end);
    getrusage(RUSAGE_CHILDREN, &usage_child_end);

    double user_usec = (usage_self_end.ru_utime.tv_sec - usage_self_start.ru_utime.tv_sec) * 1e6 +
                       (usage_self_end.ru_utime.tv_usec - usage_self_start.ru_utime.tv_usec) +
                       (usage_child_end.ru_utime.tv_sec * 1e6 + usage_child_end.ru_utime.tv_usec);
    double sys_usec = (usage_self_end.ru_stime.tv_sec - usage_self_start.ru_stime.tv_sec) * 1e6 +
                      (usage_self_end.ru_stime.tv_usec - usage_self_start.ru_stime.tv_usec) +
                      (usage_child_end.ru_stime.tv_sec * 1e6 + usage_child_end.ru_stime.tv_usec);
    double wall_msec = (wall_end.tv_sec - wall_start.tv_sec) * 1e3 +
                       (wall_end.tv_nsec - wall_start.tv_nsec) / 1e6;
    double cpu_util = 100.0 * (user_usec + sys_usec) / 1000.0 / wall_msec;

    printf("== Final Performance Metrics ==\n");
    printf("Wall Clock Time    : %.2f ms\n", wall_msec);
    printf("User CPU Time      : %.2f ms\n", user_usec / 1000.0);
    printf("System CPU Time    : %.2f ms\n", sys_usec / 1000.0);
    printf("CPU Utilization    : %.2f %%\n", cpu_util);
    printf("Total Tasks Done   : %d\n", *task_done_count);

    return 0;
}
