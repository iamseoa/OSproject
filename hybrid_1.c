// ✅ 구조 1: 이미지 묶음 → 프로세스 → Conv2D 필터를 스레드로 병렬 처리
// 구조 간략 버전: 입력 9개를 3개의 프로세스에 3개씩 할당, 각 Conv2D는 4개의 필터 병렬 처리

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/resource.h>

#define NUM_INPUTS 9
#define FILTERS 4
#define IMG_PER_PROC 3

double input_streams[NUM_INPUTS][4];
double conv_output[FILTERS];

typedef struct {
    int filter_id;
    int input_idx;
} ThreadArg1;

pthread_mutex_t mutex[FILTERS];

void* conv_filter_thread(void* arg) {
    ThreadArg1* t_arg = (ThreadArg1*)arg;
    int f = t_arg->filter_id;
    int idx = t_arg->input_idx;

    pthread_mutex_lock(&mutex[f]);
    conv_output[f] = 0;
    for (int j = 0; j < 4; j++) {
        conv_output[f] += input_streams[idx][j] * (f + 1);
    }
    pthread_mutex_unlock(&mutex[f]);
    pthread_exit(NULL);
}

void print_resource_usage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    printf("\n=== Resource Usage ===\n");
    printf("Max RSS (memory): %ld KB\n", usage.ru_maxrss);
    printf("Voluntary Context Switches: %ld\n", usage.ru_nvcsw);
    printf("Involuntary Context Switches: %ld\n", usage.ru_nivcsw);
    printf("Minor Page Faults: %ld\n", usage.ru_minflt);
    printf("Major Page Faults: %ld\n", usage.ru_majflt);
}

void process_func(int start_idx) {
    for (int i = 0; i < IMG_PER_PROC; i++) {
        int idx = start_idx + i;
        printf("===== [PID %d] Finished input stream #%d =====\n", getpid(), idx + 1);

        pthread_t tids[FILTERS];
        ThreadArg1 args[FILTERS];

        for (int f = 0; f < FILTERS; f++) {
            args[f].filter_id = f;
            args[f].input_idx = idx;
            pthread_create(&tids[f], NULL, conv_filter_thread, &args[f]);
        }

        for (int f = 0; f < FILTERS; f++) pthread_join(tids[f], NULL);

        printf("Conv2D output sample: ");
        for (int f = 0; f < FILTERS; f++) printf("%.4f ", conv_output[f]);
        printf("\n");

        printf("FC1 output sample: ");
        for (int f = 0; f < FILTERS; f++) printf("%.4f ", conv_output[f] * 64);
        printf("\n");

        printf("FC2 output sample: ");
        for (int f = 0; f < FILTERS; f++) printf("%.4f ", conv_output[f] * 256);
        printf("\n");
    }
    print_resource_usage();
    exit(0);
}

int main() {
    for (int i = 0; i < NUM_INPUTS; i++) {
        for (int j = 0; j < 4; j++) {
            input_streams[i][j] = (double)((i + 1) * (j + 1));
        }
    }

    for (int i = 0; i < FILTERS; i++) pthread_mutex_init(&mutex[i], NULL);

    for (int i = 0; i < NUM_INPUTS / IMG_PER_PROC; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            process_func(i * IMG_PER_PROC);
        }
    }

    for (int i = 0; i < NUM_INPUTS / IMG_PER_PROC; i++) wait(NULL);

    return 0;
}

