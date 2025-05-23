#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include "cnn_common.h"
#define gettid() syscall(SYS_gettid)

CNNModel model;
float inputs[NUM_INPUTS][INPUT_SIZE][INPUT_SIZE][CHANNELS];
pthread_mutex_t mutex;

void* process(void* arg) {
    int idx = *(int*)arg;
    pthread_mutex_lock(&mutex);

    printf("\n== Thread PID %d, TID %ld processing input %d ==\n", getpid(), gettid(), idx);

    float flat[(CONV_OUT / 2)*(CONV_OUT / 2)*CONV_DEPTH];
    conv_forward(&model, inputs[idx], model.conv_output);
    relu_forward(model.conv_output, model.relu_output);
    maxpool_forward(model.relu_output, model.pooled_output);
    flatten(model.pooled_output, flat);
    fc1_forward(&model, flat, model.fc1_output);
    fc2_forward(&model, model.fc1_output, model.fc2_output);

    printf("Input Patch [0:3][0:3][0]:\n");
    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) printf("%.1f ", inputs[idx][x][y][0]);
        printf("\n");
    }
    printf("Conv Output [0:5][0][0] = ");
    for (int i = 0; i < 5; i++) {
	    printf("%.2f ", model.conv_output[i][0][0]);
    }
    printf("\n");
    printf("fc1[0:5] = ");
    for (int j = 0; j < 5; j++) printf("%.2f ", model.fc1_output[j]);
    printf("\nfc2[0:5] = ");
    for (int j = 0; j < 5; j++) printf("%.2f ", model.fc2_output[j]);
    printf("\n");

    pthread_mutex_unlock(&mutex);
    return NULL;
}

int main() {
    pthread_mutex_init(&mutex, NULL);
    initialize_weights(&model);
    for (int i = 0; i < NUM_INPUTS; i++) initialize_input(inputs[i], i);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    pthread_t threads[NUM_INPUTS];
    int indices[NUM_INPUTS];
    for (int i = 0; i < NUM_INPUTS; i++) {
        indices[i] = i;
        pthread_create(&threads[i], NULL, process, &indices[i]);
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

    pthread_mutex_destroy(&mutex);
    return 0;
}
