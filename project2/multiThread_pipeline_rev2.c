#include "cnn_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <string.h>
#define gettid() syscall(SYS_gettid)

#define QUEUE_SIZE 16
#define FC1_IN ((CONV_OUT / 2) * (CONV_OUT / 2) * CONV_DEPTH)

typedef struct {
    float input[INPUT_SIZE][INPUT_SIZE][CHANNELS];
    float conv_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float relu_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float pooled_output[CONV_OUT / 2][CONV_OUT / 2][CONV_DEPTH];
    float flat[FC1_IN];
    float fc1_output[FC1_OUT];
    float fc2_output[FC2_OUT];
    int id;
} Sample;

Sample samples[NUM_INPUTS];
CNNModel model;
pthread_mutex_t print_mutex;

// Queue
typedef struct {
    Sample* items[QUEUE_SIZE];
    int front, rear, count;
    pthread_mutex_t mutex;
    pthread_cond_t not_empty, not_full;
} SampleQueue;

void init_queue(SampleQueue* q) {
    q->front = q->rear = q->count = 0;
    pthread_mutex_init(&q->mutex, NULL);
    pthread_cond_init(&q->not_empty, NULL);
    pthread_cond_init(&q->not_full, NULL);
}

void enqueue(SampleQueue* q, Sample* item) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == QUEUE_SIZE)
        pthread_cond_wait(&q->not_full, &q->mutex);
    q->items[q->rear] = item;
    q->rear = (q->rear + 1) % QUEUE_SIZE;
    q->count++;
    pthread_cond_signal(&q->not_empty);
    pthread_mutex_unlock(&q->mutex);
}

Sample* dequeue(SampleQueue* q) {
    pthread_mutex_lock(&q->mutex);
    while (q->count == 0)
        pthread_cond_wait(&q->not_empty, &q->mutex);
    Sample* item = q->items[q->front];
    q->front = (q->front + 1) % QUEUE_SIZE;
    q->count--;
    pthread_cond_signal(&q->not_full);
    pthread_mutex_unlock(&q->mutex);
    return item;
}

SampleQueue q1, q2, q3, q4;

void* conv_stage(void* arg) {
    for (int i = 0; i < NUM_INPUTS; i++) {
        conv_forward(&model, samples[i].input, samples[i].conv_output);
        enqueue(&q1, &samples[i]);
    }
    return NULL;
}

void* relu_stage(void* arg) {
    for (int i = 0; i < NUM_INPUTS; i++) {
        Sample* s = dequeue(&q1);
        relu_forward(s->conv_output, s->relu_output);
        enqueue(&q2, s);
    }
    return NULL;
}

void* maxpool_stage(void* arg) {
    for (int i = 0; i < NUM_INPUTS; i++) {
        Sample* s = dequeue(&q2);
        maxpool_forward(s->relu_output, s->pooled_output);
        enqueue(&q3, s);
    }
    return NULL;
}

void* flatten_fc1_stage(void* arg) {
    for (int i = 0; i < NUM_INPUTS; i++) {
        Sample* s = dequeue(&q3);
        flatten(s->pooled_output, s->flat);
        fc1_forward(&model, s->flat, s->fc1_output);
        enqueue(&q4, s);
    }
    return NULL;
}

void* fc2_output_stage(void* arg) {
    for (int i = 0; i < NUM_INPUTS; i++) {
        Sample* s = dequeue(&q4);
        fc2_forward(&model, s->fc1_output, s->fc2_output);

        pthread_mutex_lock(&print_mutex);
        printf("\n== Thread PID %d, TID %ld processing input %d ==\n", getpid(), gettid(), s->id);
        printf("Input Patch [0:3][0:3][0]:\n");
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) printf("%.1f ", s->input[x][y][0]);
            printf("\n");
        }
        printf("Conv Output [0:5][0][0] = ");
        for (int j = 0; j < 5; j++) printf("%.2f ", s->conv_output[j][0][0]);
        printf("\nfc1[0:5] = ");
        for (int j = 0; j < 5; j++) printf("%.2f ", s->fc1_output[j]);
        printf("\nfc2[0:5] = ");
        for (int j = 0; j < 5; j++) printf("%.2f ", s->fc2_output[j]);
        printf("\n");
        pthread_mutex_unlock(&print_mutex);
    }
    return NULL;
}

int main() {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    pthread_mutex_init(&print_mutex, NULL);
    initialize_weights(&model);

    #pragma omp parallel for
    for (int i = 0; i < NUM_INPUTS; i++) {
        initialize_input(samples[i].input, i);
        samples[i].id = i;
    }

    init_queue(&q1); init_queue(&q2); init_queue(&q3); init_queue(&q4);

    pthread_t convT, reluT, poolT, flatT, outT;
    pthread_create(&convT, NULL, conv_stage, NULL);
    pthread_create(&reluT, NULL, relu_stage, NULL);
    pthread_create(&poolT, NULL, maxpool_stage, NULL);
    pthread_create(&flatT, NULL, flatten_fc1_stage, NULL);
    pthread_create(&outT, NULL, fc2_output_stage, NULL);

    pthread_join(convT, NULL);
    pthread_join(reluT, NULL);
    pthread_join(poolT, NULL);
    pthread_join(flatT, NULL);
    pthread_join(outT, NULL);

    gettimeofday(&end, NULL);
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 +
                     (end.tv_usec - start.tv_usec) / 1000.0;

    printf("\n== Performance Metrics ==\n");
    printf("Elapsed time       : %.3f ms\n", elapsed);
    printf("Max RSS memory     : %ld KB\n", usage.ru_maxrss);
    printf("Voluntary context switches   : %ld\n", usage.ru_nvcsw);
    printf("Involuntary context switches : %ld\n", usage.ru_nivcsw);

    return 0;
}

