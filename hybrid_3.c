// ✅ 구조 3: 이미지 1장당 프로세스 생성 → Conv/Pool/FC를 각 스레드로 병렬 처리
// 레이어 간 순서를 보장하기 위해 mutex + condition 변수 사용

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/resource.h>

#define NUM_INPUTS 9

double input_streams[NUM_INPUTS][4];

double conv_output[4];
double pool_output[4];
double fc1_output[4];
double fc2_output[4];

pthread_mutex_t sync_lock;
pthread_cond_t conv_done, pool_done;
int conv_ready = 0;
int pool_ready = 0;

int input_idx;

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

void* conv_layer(void* arg) {
    for (int f = 0; f < 4; f++) {
        conv_output[f] = 0;
        for (int j = 0; j < 4; j++) {
            conv_output[f] += input_streams[input_idx][j] * (f + 1);
        }
    }
    pthread_mutex_lock(&sync_lock);
    conv_ready = 1;
    pthread_cond_signal(&conv_done);
    pthread_mutex_unlock(&sync_lock);
    pthread_exit(NULL);
}

void* pool_layer(void* arg) {
    pthread_mutex_lock(&sync_lock);
    while (!conv_ready) pthread_cond_wait(&conv_done, &sync_lock);
    pthread_mutex_unlock(&sync_lock);

    for (int f = 0; f < 4; f++) {
        pool_output[f] = conv_output[f]; // 간단히 전달 (실제 풀링 생략)
    }

    pthread_mutex_lock(&sync_lock);
    pool_ready = 1;
    pthread_cond_signal(&pool_done);
    pthread_mutex_unlock(&sync_lock);
    pthread_exit(NULL);
}

void* fc_layer(void* arg) {
    pthread_mutex_lock(&sync_lock);
    while (!pool_ready) pthread_cond_wait(&pool_done, &sync_lock);
    pthread_mutex_unlock(&sync_lock);

    for (int f = 0; f < 4; f++) {
        fc1_output[f] = pool_output[f] * 64;
        fc2_output[f] = pool_output[f] * 256;
    }
    pthread_exit(NULL);
}

void run_cnn_pipeline(int idx) {
    input_idx = idx;
    pthread_t t_conv, t_pool, t_fc;
    pthread_mutex_init(&sync_lock, NULL);
    pthread_cond_init(&conv_done, NULL);
    pthread_cond_init(&pool_done, NULL);

    conv_ready = 0;
    pool_ready = 0;

    pthread_create(&t_conv, NULL, conv_layer, NULL);
    pthread_create(&t_pool, NULL, pool_layer, NULL);
    pthread_create(&t_fc, NULL, fc_layer, NULL);

    pthread_join(t_conv, NULL);
    pthread_join(t_pool, NULL);
    pthread_join(t_fc, NULL);

    printf("===== [PID %d] Finished input stream #%d =====\n", getpid(), idx + 1);

    printf("Conv2D output sample: ");
    for (int f = 0; f < 4; f++) printf("%.4f ", conv_output[f]);
    printf("\n");

    printf("FC1 output sample: ");
    for (int f = 0; f < 4; f++) printf("%.4f ", fc1_output[f]);
    printf("\n");

    printf("FC2 output sample: ");
    for (int f = 0; f < 4; f++) printf("%.4f ", fc2_output[f]);
    printf("\n");

    print_resource_usage();
    exit(0);
}

int main() {
    for (int i = 0; i < NUM_INPUTS; i++) {
        for (int j = 0; j < 4; j++) {
            input_streams[i][j] = (double)((i + 1) * (j + 1));
        }
    }

    for (int i = 0; i < NUM_INPUTS; i++) {
        pid_t pid = fork();
        if (pid == 0) run_cnn_pipeline(i);
    }

    for (int i = 0; i < NUM_INPUTS; i++) wait(NULL);
    return 0;
}
