// multiThreadChild.c - CNN + TaskQueue (Race Condition 유발 + 연산 수행 포함)

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
#define gettid() syscall(SYS_gettid)

#define INPUT_SIZE 224
#define CHANNELS 3
#define KERNEL_SIZE 3
#define CONV_DEPTH 64
#define CONV_OUT (INPUT_SIZE - KERNEL_SIZE + 1)
#define FC1_OUT 256
#define FC2_OUT 100
#define NUM_INPUTS 4
#define NUM_THREADS 4
#define QUEUE_SIZE 2

typedef struct {
    float input[INPUT_SIZE][INPUT_SIZE][CHANNELS];
    float conv_out[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float fc1_out[FC1_OUT];
    float fc2_out[FC2_OUT];
    int input_id;
} Task;

typedef struct {
    Task* buffer[QUEUE_SIZE];
    int front, rear, count;
} TaskQueue;

TaskQueue* queue;
Task* task_pool;

pthread_mutex_t* print_mutex;

void initialize_input(Task* t, int id) {
    float center = 9.0f * (id + 1);
    t->input_id = id;
    for (int c = 0; c < CHANNELS; c++)
        for (int i = 0; i < INPUT_SIZE; i++)
            for (int j = 0; j < INPUT_SIZE; j++)
                t->input[i][j][c] = (i == 1 && j == 1) ? center : 1.0f;
}

void* producer(void* arg) {
    for (int i = 0; i < NUM_INPUTS; i++) {
        Task* t = &task_pool[i];
        initialize_input(t, i);
        usleep(rand() % 50);
        if (queue->count < QUEUE_SIZE) {
            queue->buffer[queue->rear] = t;
            queue->rear = (queue->rear + 1) % QUEUE_SIZE;
            queue->count++;
            printf("[Producer] Enqueued Task %d\n", t->input_id);
        } else {
            printf("[Producer] Queue full, skip Task %d\n", t->input_id);
        }
    }
    return NULL;
}

void* consumer(void* arg) {
    int tid = *(int*)arg;
    int processed = 0;
    while (processed < NUM_INPUTS / NUM_THREADS) {
        Task* t = NULL;
        usleep(rand() % 100);
        if (queue->count > 0) {
            t = queue->buffer[queue->front];
            queue->front = (queue->front + 1) % QUEUE_SIZE;
            queue->count--;
            processed++;

            // 연산 및 출력 동일


            // CNN 연산 대체 수행 (단순 합산)
            float sum = 0.0f;
            for (int c = 0; c < CHANNELS; c++)
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        sum += t->input[i][j][c];

            t->conv_out[0][0][0] = sum;
            for (int i = 0; i < 5; i++) {
                t->fc1_out[i] = sum + i;
                t->fc2_out[i] = sum + i * 2;
            }

            pthread_mutex_lock(print_mutex);
            printf("Input ID: %d\n", t->input_id);
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
            pthread_mutex_unlock(print_mutex);
        } else {
            printf("[Consumer] Queue empty\n");
        }
    }
    return NULL;
}

int main() {
    srand(getpid());
    queue = mmap(NULL, sizeof(TaskQueue), PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    task_pool = mmap(NULL, sizeof(Task) * NUM_INPUTS, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    print_mutex = mmap(NULL, sizeof(pthread_mutex_t), PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(print_mutex, &attr);

    queue->front = queue->rear = queue->count = 0;

    pid_t pid = fork();
    if (pid == 0) {
        // 자식 프로세스 - Producer
        pthread_t prod;
        pthread_create(&prod, NULL, producer, NULL);
        pthread_join(prod, NULL);
        exit(0); // producer 종료 후 exit
    } else {
        // 부모 프로세스 - Consumer 여러 개 생성
        pthread_t cons[NUM_THREADS];
        int tids[NUM_THREADS];
        for (int i = 0; i < NUM_THREADS; i++) {
            tids[i] = i;
            pthread_create(&cons[i], NULL, consumer, &tids[i]);
        }

        // Producer 종료 기다림
        wait(NULL);

        // 모든 consumer join
        for (int i = 0; i < NUM_THREADS; i++)
            pthread_join(cons[i], NULL);
    }

    return 0;
}

