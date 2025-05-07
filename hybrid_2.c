// ✅ 구조 2 (수정 버전): 작업 큐 공유 → 고정된 child process pool + 내부 스레드 풀
// sem_wait() 무한 대기 문제를 해결하기 위해 sem_init 값을 증가시키고, 큐 종료 시 break 처리 추가

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <semaphore.h>

#define NUM_INPUTS 9
#define NUM_PROCS 3
#define NUM_THREADS 2

int work_queue[NUM_INPUTS];
int queue_idx = 0;
double input_streams[NUM_INPUTS][4];
pthread_mutex_t queue_lock;
sem_t work_sem;

typedef struct {
    int thread_id;
    int proc_id;
} ThreadArg;

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

void* worker_thread(void* arg) {
    ThreadArg* t_arg = (ThreadArg*)arg;
    int tid = t_arg->thread_id;
    int pid = t_arg->proc_id;

    while (1) {
        sem_wait(&work_sem);

        pthread_mutex_lock(&queue_lock);
        if (queue_idx >= NUM_INPUTS) {
            pthread_mutex_unlock(&queue_lock);
            break;
        }
        int idx = work_queue[queue_idx++];
        pthread_mutex_unlock(&queue_lock);

        double conv_output[4];
        for (int f = 0; f < 4; f++) {
            conv_output[f] = 0;
            for (int j = 0; j < 4; j++) {
                conv_output[f] += input_streams[idx][j] * (f + 1);
            }
        }

        printf("===== [PID %d | TID %d] Finished input stream #%d =====\n", getpid(), tid, idx + 1);

        printf("Conv2D output sample: ");
        for (int f = 0; f < 4; f++) printf("%.4f ", conv_output[f]);
        printf("\n");

        printf("FC1 output sample: ");
        for (int f = 0; f < 4; f++) printf("%.4f ", conv_output[f] * 64);
        printf("\n");

        printf("FC2 output sample: ");
        for (int f = 0; f < 4; f++) printf("%.4f ", conv_output[f] * 256);
        printf("\n");
    }
    pthread_exit(NULL);
}

void process_worker(int proc_id) {
    pthread_t tids[NUM_THREADS];
    ThreadArg args[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].thread_id = i;
        args[i].proc_id = proc_id;
        pthread_create(&tids[i], NULL, worker_thread, &args[i]);
    }
    for (int i = 0; i < NUM_THREADS; i++) pthread_join(tids[i], NULL);

    print_resource_usage();
    exit(0);
}

int main() {
    for (int i = 0; i < NUM_INPUTS; i++) {
        for (int j = 0; j < 4; j++) {
            input_streams[i][j] = (double)((i + 1) * (j + 1));
        }
        work_queue[i] = i;
    }

    queue_idx = 0;
    pthread_mutex_init(&queue_lock, NULL);
    sem_init(&work_sem, 1, NUM_INPUTS + NUM_PROCS * NUM_THREADS);

    for (int i = 0; i < NUM_PROCS; i++) {
        pid_t pid = fork();
        if (pid == 0) process_worker(i);
    }

    for (int i = 0; i < NUM_PROCS; i++) wait(NULL);

    return 0;
}
