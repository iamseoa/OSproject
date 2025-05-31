#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <string.h>
#include <fcntl.h>
#include <math.h>
#include <sched.h>
#include <sys/mman.h>
#include <semaphore.h>

#define STACK_SIZE (16 * 1024 * 1024)
#define THREAD_COUNT 4
#define PROCESS_COUNT 4
#define TOTAL_TASKS 40
#define IMAGE_SIZE 224
#define CONV_DEPTH 64
#define KERNEL_SIZE 3
#define CONV_OUT (IMAGE_SIZE - KERNEL_SIZE + 1)
#define FC1_OUT 512
#define FC2_OUT 100
#define POOL_OUT (CONV_OUT / 2)
#define FLAT_SIZE (POOL_OUT * POOL_OUT * CONV_DEPTH)

typedef struct {
    float weights[CONV_DEPTH][3][KERNEL_SIZE][KERNEL_SIZE];
    float biases[CONV_DEPTH];
} ConvLayer;

typedef struct {
    float weights[FC1_OUT][FLAT_SIZE];
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
    int id;
    float input[IMAGE_SIZE][IMAGE_SIZE][3];
} InputData;

typedef struct {
    int input_id;
    float input[IMAGE_SIZE][IMAGE_SIZE][3];
    float conv_out[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float flat[FLAT_SIZE];
    float fc1_out[FC1_OUT];
    float fc2_out[FC2_OUT];
    float relu_out_sample[2][3][3];
} Task;

typedef struct {
    int id;
    int is_thread;  // 0 = process, 1 = thread
} WorkerArgs;

// Shared variables
CNNModel* model;
InputData* inputs;
int* current_index;
sem_t* input_sem;

void initialize_input(InputData* t, int id) {
    float center = 9.0f * (id + 1);
    t->id = id;
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < IMAGE_SIZE; i++)
            for (int j = 0; j < IMAGE_SIZE; j++)
                t->input[i][j][c] = (i == IMAGE_SIZE / 2 && j == IMAGE_SIZE / 2) ? center : 1.0f;
}

void initialize_weights(CNNModel* model) {
    int base_kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    for (int d = 0; d < CONV_DEPTH; d++) {
        model->conv.biases[d] = 1.0f;
        for (int c = 0; c < 3; c++) {
            float scale = 1.0f + 0.5f * c + 0.1f * d;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    model->conv.weights[d][c][i][j] = base_kernel[i][j] * scale;
        }
    }
    for (int i = 0; i < FC1_OUT; i++) {
        model->fc1.biases[i] = 1.0f;
        for (int j = 0; j < FLAT_SIZE; j++)
            model->fc1.weights[i][j] = ((j % (FLAT_SIZE / FC1_OUT)) == 0) ? 1.0f : 0.0f;
    }
    for (int i = 0; i < FC2_OUT; i++) {
        model->fc2.biases[i] = 1.0f;
        for (int j = 0; j < FC1_OUT; j++)
            model->fc2.weights[i][j] = ((j % (FC1_OUT / FC2_OUT)) == 0) ? 1.0f : 0.0f;
    }
}

void generate_inputs() {
    for (int i = 0; i < TOTAL_TASKS; i++) {
        initialize_input(&inputs[i], i);
    }
}

void cnn_inference(Task* t) {
    float* relu_out = malloc(sizeof(float) * CONV_OUT * CONV_OUT * CONV_DEPTH);
    float* pool_out = malloc(sizeof(float) * POOL_OUT * POOL_OUT * CONV_DEPTH);

    for (int d = 0; d < CONV_DEPTH; d++) {
        for (int i = 0; i < CONV_OUT; i++) {
            for (int j = 0; j < CONV_OUT; j++) {
                float sum = model->conv.biases[d];
                for (int c = 0; c < 3; c++)
                    for (int ki = 0; ki < KERNEL_SIZE; ki++)
                        for (int kj = 0; kj < KERNEL_SIZE; kj++)
                            sum += model->conv.weights[d][c][ki][kj] * t->input[i + ki][j + kj][c];
                relu_out[(d * CONV_OUT + i) * CONV_OUT + j] = fmaxf(sum, 0.0f);
            }
        }
    }

    for (int d = 0; d < CONV_DEPTH; d++) {
        for (int i = 0; i < POOL_OUT; i++) {
            for (int j = 0; j < POOL_OUT; j++) {
                float max_val = relu_out[(d * CONV_OUT + i * 2) * CONV_OUT + j * 2];
                for (int ki = 0; ki < 2; ki++)
                    for (int kj = 0; kj < 2; kj++) {
                        float val = relu_out[(d * CONV_OUT + i * 2 + ki) * CONV_OUT + j * 2 + kj];
                        if (val > max_val) max_val = val;
                    }
                pool_out[(d * POOL_OUT + i) * POOL_OUT + j] = max_val;
            }
        }
    }

    int idx = 0;
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < POOL_OUT; i++)
            for (int j = 0; j < POOL_OUT; j++)
                t->flat[idx++] = pool_out[(d * POOL_OUT + i) * POOL_OUT + j];

    for (int i = 0; i < FC1_OUT; i++) {
        float sum = model->fc1.biases[i];
        for (int j = 0; j < FLAT_SIZE; j++)
            sum += model->fc1.weights[i][j] * t->flat[j];
        t->fc1_out[i] = sum;
    }

    for (int i = 0; i < FC2_OUT; i++) {
        float sum = model->fc2.biases[i];
        for (int j = 0; j < FC1_OUT; j++)
            sum += model->fc2.weights[i][j] * t->fc1_out[j];
        t->fc2_out[i] = sum;
    }

    for (int d = 0; d < 2; d++)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                t->relu_out_sample[d][i][j] = relu_out[(d * CONV_OUT + i) * CONV_OUT + j];

    free(relu_out);
    free(pool_out);
}

void* runner(void* arg) {
    WorkerArgs* args = (WorkerArgs*)arg;
    while (1) {
        struct timeval start, end;
        struct rusage usage;

        sem_wait(input_sem);
        if (*current_index >= TOTAL_TASKS) {
            sem_post(input_sem);
            break;
        }
        int idx = (*current_index)++;
        sem_post(input_sem);

        Task* t = malloc(sizeof(Task));
        if (!t) {
            perror("malloc Task failed");
            exit(1);
        }

        gettimeofday(&start, NULL);
        t->input_id = inputs[idx].id;
        memcpy(t->input, inputs[idx].input, sizeof(t->input));

        cnn_inference(t);

        gettimeofday(&end, NULL);
        getrusage(args->is_thread ? RUSAGE_THREAD : RUSAGE_SELF, &usage);

        printf("\n[Runner ID %d | PID %d] Input ID: %d\n", args->id, getpid(), t->input_id);
        printf("fc1[0:5] = ");
        for (int i = 0; i < 5; i++) printf("%.2f ", t->fc1_out[i]);
        printf("\nfc2[0:5] = ");
        for (int i = 0; i < 5; i++) printf("%.2f ", t->fc2_out[i]);
        printf("\n");

        double wall_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                         (end.tv_usec - start.tv_usec) / 1000.0;
        double user_ms = usage.ru_utime.tv_sec * 1000.0 + usage.ru_utime.tv_usec / 1000.0;
        double sys_ms = usage.ru_stime.tv_sec * 1000.0 + usage.ru_stime.tv_usec / 1000.0;
        double cpu_util = (user_ms + sys_ms) / wall_ms * 100.0;

        printf("Wall: %.2f ms, CPU: %.2f%%\n", wall_ms, cpu_util);

        free(t);
    }

    if (!args->is_thread) exit(0);
    return NULL;
}

int main() {
    struct timeval total_start, total_end;
    gettimeofday(&total_start, NULL);

    // Shared memory
    inputs = mmap(NULL, sizeof(InputData) * TOTAL_TASKS,
                  PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    current_index = mmap(NULL, sizeof(int),
                         PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    input_sem = mmap(NULL, sizeof(sem_t),
                     PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    model = mmap(NULL, sizeof(CNNModel),
                 PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    if (inputs == MAP_FAILED || current_index == MAP_FAILED ||
        input_sem == MAP_FAILED || model == MAP_FAILED) {
        perror("mmap failed");
        exit(1);
    }

    *current_index = 0;
    sem_init(input_sem, 1, 1);

    initialize_weights(model);
    generate_inputs();

    pthread_t threads[THREAD_COUNT];
    WorkerArgs thread_args[THREAD_COUNT];

    for (int i = 0; i < PROCESS_COUNT; i++) {
        pid_t pid = fork();
        if (pid == 0) {
            WorkerArgs* proc_args = malloc(sizeof(WorkerArgs));
            proc_args->id = i;
            proc_args->is_thread = 0;
            runner(proc_args);  // 내부에서 exit
        }
    }

    for (int i = 0; i < THREAD_COUNT; i++) {
        thread_args[i].id = i;
        thread_args[i].is_thread = 1;
        pthread_create(&threads[i], NULL, runner, &thread_args[i]);
    }

    for (int i = 0; i < THREAD_COUNT; i++) pthread_join(threads[i], NULL);
    for (int i = 0; i < PROCESS_COUNT; i++) wait(NULL);

    gettimeofday(&total_end, NULL);
    double total_time = (total_end.tv_sec - total_start.tv_sec) * 1000.0 +
                        (total_end.tv_usec - total_start.tv_usec) / 1000.0;
    printf("\n== Done ==\nTotal Time: %.2f ms\n", total_time);

    return 0;
}
