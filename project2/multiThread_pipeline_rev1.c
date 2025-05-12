#include "cnn_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

#define MAX_PRINT_LEN 512

typedef struct {
    float input[INPUT_SIZE][INPUT_SIZE][CHANNELS];
    float conv_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float relu_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float pooled_output[CONV_OUT / 2][CONV_OUT / 2][CONV_DEPTH];
    float flat[(CONV_OUT / 2) * (CONV_OUT / 2) * CONV_DEPTH];
    float fc1_output[FC1_OUT];
    float fc2_output[FC2_OUT];
    int id;
} Sample;

CNNModel model;
Sample samples[NUM_INPUTS];
pthread_mutex_t print_mutex;

void* pipeline_thread(void* arg) {
    Sample* s = (Sample*)arg;
    int id = s->id;

    conv_forward(&model, s->input, s->conv_output);
    relu_forward(s->conv_output, s->relu_output);
    maxpool_forward(s->relu_output, s->pooled_output);
    flatten(s->pooled_output, s->flat);
    fc1_forward(&model, s->flat, s->fc1_output);
    fc2_forward(&model, s->fc1_output, s->fc2_output);

    pthread_mutex_lock(&print_mutex);
    char buf[1024];
    int offset = 0;
    offset += snprintf(buf + offset, sizeof(buf) - offset,
        "\n== Thread PID %d, TID %ld processing input %d ==\n",
        getpid(), gettid(), id);
    offset += snprintf(buf + offset, sizeof(buf) - offset,
        "Input Patch [0:3][0:3][0]:\n");
    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) {
            offset += snprintf(buf + offset, sizeof(buf) - offset,
                "%.1f ", s->input[x][y][0]);
        }
        offset += snprintf(buf + offset, sizeof(buf) - offset, "\n");
    }
    offset += snprintf(buf + offset, sizeof(buf) - offset, "Conv Output [0:5][0][0] = ");
    for (int i = 0; i < 5; i++) offset += snprintf(buf + offset, sizeof(buf) - offset, "%.2f ", s->conv_output[i][0][0]);
    offset += snprintf(buf + offset, sizeof(buf) - offset, "\nfc1[0:5] = ");
    for (int i = 0; i < 5; i++) offset += snprintf(buf + offset, sizeof(buf) - offset, "%.2f ", s->fc1_output[i]);
    offset += snprintf(buf + offset, sizeof(buf) - offset, "\nfc2[0:5] = ");
    for (int i = 0; i < 5; i++) offset += snprintf(buf + offset, sizeof(buf) - offset, "%.2f ", s->fc2_output[i]);
    offset += snprintf(buf + offset, sizeof(buf) - offset, "\n");
    fputs(buf, stdout);
    pthread_mutex_unlock(&print_mutex);

    return NULL;
}

int main() {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    pthread_mutex_init(&print_mutex, NULL);
    initialize_weights(&model);
    pthread_t threads[NUM_INPUTS];

    // 병렬 입력 초기화
    #pragma omp parallel for
    for (int i = 0; i < NUM_INPUTS; i++) {
        initialize_input(samples[i].input, i);
        samples[i].id = i;
    }

    for (int i = 0; i < NUM_INPUTS; i++) {
        pthread_create(&threads[i], NULL, pipeline_thread, &samples[i]);
    }

    for (int i = 0; i < NUM_INPUTS; i++) pthread_join(threads[i], NULL);

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
