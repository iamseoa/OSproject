#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <omp.h>
#define gettid() syscall(SYS_gettid)

#define NUM_INPUTS 4
#define INPUT_SIZE 32
#define CHANNELS 3
#define KERNEL_SIZE 3
#define CONV_OUT 30
#define CONV_DEPTH 16
#define FC1_OUT 256
#define FC2_OUT 100

typedef struct {
    float weights[CONV_DEPTH][CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    float biases[CONV_DEPTH];
} ConvLayer;

typedef struct {
    float weights[FC1_OUT][(CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH];
    float biases[FC1_OUT];
} FullyConnectedLayer1;

typedef struct {
    float weights[FC2_OUT][FC1_OUT];
    float biases[FC2_OUT];
} FullyConnectedLayer2;

typedef struct {
    float input[INPUT_SIZE][INPUT_SIZE][CHANNELS];
    float conv_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float relu_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float pooled_output[CONV_OUT/2][CONV_OUT/2][CONV_DEPTH];
    float flat[(CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH];
    float fc1_output[FC1_OUT];
    float fc2_output[FC2_OUT];
    int id;
} Sample;

ConvLayer conv;
FullyConnectedLayer1 fc1;
FullyConnectedLayer2 fc2;
Sample samples[NUM_INPUTS];

void initialize_input(Sample* s, int id) {
    s->id = id;
    for (int c = 0; c < CHANNELS; c++)
        for (int i = 0; i < INPUT_SIZE; i++)
            for (int j = 0; j < INPUT_SIZE; j++)
                s->input[i][j][c] = (i == 1 && j == 1) ? (float)(9 * (id + 1)) : 1.0f;
}

void initialize_weights() {
    int kernel[3][3] = {{1,2,1},{2,4,2},{1,2,1}};
    for (int d = 0; d < CONV_DEPTH; d++) {
        conv.biases[d] = 1.0f;
        for (int c = 0; c < CHANNELS; c++)
            for (int i = 0; i < KERNEL_SIZE; i++)
                for (int j = 0; j < KERNEL_SIZE; j++)
                    conv.weights[d][c][i][j] = kernel[i][j];
    }
    for (int i = 0; i < FC1_OUT; i++) {
        fc1.biases[i] = 1.0f;
        for (int j = 0; j < (CONV_OUT/2)*(CONV_OUT/2)*CONV_DEPTH; j++)
            fc1.weights[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    for (int i = 0; i < FC2_OUT; i++) {
        fc2.biases[i] = 1.0f;
        for (int j = 0; j < FC1_OUT; j++)
            fc2.weights[i][j] = (i == j) ? 1.0f : 0.0f;
    }
}

void* process_sample(void* arg) {
    Sample* s = (Sample*)arg;
    int id = s->id;

    // Conv Layer (im2col 없이 단순 병렬화)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < CONV_OUT; i++) {
        for (int j = 0; j < CONV_OUT; j++) {
            for (int d = 0; d < CONV_DEPTH; d++) {
                float sum = conv.biases[d];
                for (int c = 0; c < CHANNELS; c++)
                    for (int ki = 0; ki < KERNEL_SIZE; ki++)
                        for (int kj = 0; kj < KERNEL_SIZE; kj++)
                            sum += conv.weights[d][c][ki][kj] * s->input[i + ki][j + kj][c];
                s->conv_output[i][j][d] = sum;
                s->relu_output[i][j][d] = sum > 0 ? sum : 0;
            }
        }
    }

    // MaxPool + Flatten
    int idx = 0;
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT; i += 2)
            for (int j = 0; j < CONV_OUT; j += 2) {
                float maxval = s->relu_output[i][j][d];
                for (int di = 0; di < 2; di++)
                    for (int dj = 0; dj < 2; dj++) {
                        float val = s->relu_output[i + di][j + dj][d];
                        if (val > maxval) maxval = val;
                    }
                s->pooled_output[i/2][j/2][d] = maxval;
                s->flat[idx++] = maxval;
            }

    // FC1 (GEMM 구조)
    #pragma omp parallel for
    for (int i = 0; i < FC1_OUT; i++) {
        s->fc1_output[i] = fc1.biases[i];
        for (int j = 0; j < idx; j++)
            s->fc1_output[i] += fc1.weights[i][j] * s->flat[j];
    }

    // FC2 (GEMM 구조)
    #pragma omp parallel for
    for (int i = 0; i < FC2_OUT; i++) {
        s->fc2_output[i] = fc2.biases[i];
        for (int j = 0; j < FC1_OUT; j++)
            s->fc2_output[i] += fc2.weights[i][j] * s->fc1_output[j];
    }

    // 출력
    printf("\n== Thread PID %d, TID %ld processing input %d ==\n", getpid(), gettid(), id);
    printf("Input Patch [0:3][0:3][0]:\n");
    for (int x = 0; x < 3; x++) {
        for (int y = 0; y < 3; y++) printf("%.1f ", s->input[x][y][0]);
        printf("\n");
    }
    printf("Conv Output [0][0][0] = %.2f\n", s->conv_output[0][0][0]);
    printf("fc1[0:5] = ");
    for (int j = 0; j < 5; j++) printf("%.2f ", s->fc1_output[j]);
    printf("\nfc2[0:5] = ");
    for (int j = 0; j < 5; j++) printf("%.2f ", s->fc2_output[j]);
    printf("\n");

    return NULL;
}

int main() {
    struct timeval start, end;
    gettimeofday(&start, NULL);

    initialize_weights();
    for (int i = 0; i < NUM_INPUTS; i++)
        initialize_input(&samples[i], i);

    pthread_t threads[NUM_INPUTS];
    for (int i = 0; i < NUM_INPUTS; i++)
        pthread_create(&threads[i], NULL, process_sample, &samples[i]);
    for (int i = 0; i < NUM_INPUTS; i++)
        pthread_join(threads[i], NULL);

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
