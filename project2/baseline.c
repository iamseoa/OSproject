#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

#define INPUT_SIZE 64
#define KERNEL_SIZE 3
#define CONV_OUT (INPUT_SIZE - KERNEL_SIZE + 1)
#define CHANNELS 3
#define CONV_DEPTH 16
#define FC1_OUT 256
#define FC2_OUT 100
#define NUM_INPUTS 9

typedef struct {
    float weights[CONV_DEPTH][CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    float biases[CONV_DEPTH];
} ConvLayer;

typedef struct {
    float weights[FC1_OUT][(CONV_OUT / 2) * (CONV_OUT / 2) * CONV_DEPTH];
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
    float conv_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float relu_output[CONV_OUT][CONV_OUT][CONV_DEPTH];
    float pooled_output[CONV_OUT / 2][CONV_OUT / 2][CONV_DEPTH];
    float fc1_output[FC1_OUT];
    float fc2_output[FC2_OUT];
} CNNModel;

CNNModel model;
float inputs[NUM_INPUTS][INPUT_SIZE][INPUT_SIZE][CHANNELS];

void initialize_input(float input[INPUT_SIZE][INPUT_SIZE][CHANNELS], int input_id) {
    float base = 1.0f;
    float center = base * (input_id + 1) * 9.0f;
    for (int c = 0; c < CHANNELS; c++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            for (int j = 0; j < INPUT_SIZE; j++) {
                input[i][j][c] = (i == 1 && j == 1) ? center : base;
            }
        }
    }
}

void initialize_weights() {
    int kernel[3][3] = {{1,2,1},{2,4,2},{1,2,1}};
    for (int d = 0; d < CONV_DEPTH; d++) {
        model.conv.biases[d] = 1.0f;
        for (int c = 0; c < CHANNELS; c++)
            for (int i = 0; i < KERNEL_SIZE; i++)
                for (int j = 0; j < KERNEL_SIZE; j++)
                    model.conv.weights[d][c][i][j] = (float)kernel[i][j];
    }
    for (int i = 0; i < FC1_OUT; i++) {
        model.fc1.biases[i] = 1.0f;
        for (int j = 0; j < (CONV_OUT / 2) * (CONV_OUT / 2) * CONV_DEPTH; j++)
            model.fc1.weights[i][j] = (i == j) ? 1.0f : 0.0f;
    }
    for (int i = 0; i < FC2_OUT; i++) {
        model.fc2.biases[i] = 1.0f;
        for (int j = 0; j < FC1_OUT; j++)
            model.fc2.weights[i][j] = (i == j) ? 1.0f : 0.0f;
    }
}

void conv_forward(float input[INPUT_SIZE][INPUT_SIZE][CHANNELS], float output[CONV_OUT][CONV_OUT][CONV_DEPTH]) {
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT; i++)
            for (int j = 0; j < CONV_OUT; j++) {
                float sum = model.conv.biases[d];
                for (int c = 0; c < CHANNELS; c++)
                    for (int ki = 0; ki < KERNEL_SIZE; ki++)
                        for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                            int x = i + ki;
                            int y = j + kj;
                            sum += model.conv.weights[d][c][ki][kj] * input[x][y][c];
                        }
                output[i][j][d] = sum;
            }
}

void relu_forward(float in[CONV_OUT][CONV_OUT][CONV_DEPTH], float out[CONV_OUT][CONV_OUT][CONV_DEPTH]) {
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT; i++)
            for (int j = 0; j < CONV_OUT; j++)
                out[i][j][d] = (in[i][j][d] > 0) ? in[i][j][d] : 0;
}

void maxpool_forward(float in[CONV_OUT][CONV_OUT][CONV_DEPTH], float out[CONV_OUT / 2][CONV_OUT / 2][CONV_DEPTH]) {
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT; i += 2)
            for (int j = 0; j < CONV_OUT; j += 2) {
                float maxval = in[i][j][d];
                for (int di = 0; di < 2; di++)
                    for (int dj = 0; dj < 2; dj++) {
                        float val = in[i + di][j + dj][d];
                        if (val > maxval) maxval = val;
                    }
                out[i / 2][j / 2][d] = maxval;
            }
}

void flatten(float in[CONV_OUT / 2][CONV_OUT / 2][CONV_DEPTH], float out[]) {
    int idx = 0;
    for (int d = 0; d < CONV_DEPTH; d++)
        for (int i = 0; i < CONV_OUT / 2; i++)
            for (int j = 0; j < CONV_OUT / 2; j++)
                out[idx++] = in[i][j][d];
}

void fc1_forward(float input[], float output[FC1_OUT]) {
    for (int i = 0; i < FC1_OUT; i++) {
        output[i] = model.fc1.biases[i];
        for (int j = 0; j < (CONV_OUT / 2) * (CONV_OUT / 2) * CONV_DEPTH; j++)
            output[i] += model.fc1.weights[i][j] * input[j];
    }
}

void fc2_forward(float input[], float output[FC2_OUT]) {
    for (int i = 0; i < FC2_OUT; i++) {
        output[i] = model.fc2.biases[i];
        for (int j = 0; j < FC1_OUT; j++)
            output[i] += model.fc2.weights[i][j] * input[j];
    }
}

void print_resource_usage(struct timeval start, struct timeval end) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double elapsed = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;

    printf("\n== Performance Metrics ==\n");
    printf("Elapsed time       : %.3f ms\n", elapsed);
    printf("Max RSS memory     : %ld KB\n", usage.ru_maxrss);
    printf("Voluntary context switches   : %ld\n", usage.ru_nvcsw);
    printf("Involuntary context switches : %ld\n", usage.ru_nivcsw);
}

int main() {
    printf("== Running on PID %d, TID %ld ==\n", getpid(), (long)gettid());
    initialize_weights();
    for (int i = 0; i < NUM_INPUTS; i++)
        initialize_input(inputs[i], i);

    struct timeval start, end;
    gettimeofday(&start, NULL);

    for (int i = 0; i < NUM_INPUTS; i++) {
        printf("\n== Processing Input %d ==\n", i);

        conv_forward(inputs[i], model.conv_output);
        relu_forward(model.conv_output, model.relu_output);
        maxpool_forward(model.relu_output, model.pooled_output);

        float flat[(CONV_OUT / 2) * (CONV_OUT / 2) * CONV_DEPTH];
        flatten(model.pooled_output, flat);
        fc1_forward(flat, model.fc1_output);
        fc2_forward(model.fc1_output, model.fc2_output);

        printf("Input Patch [0:3][0:3][0]:\n");
        for (int x = 0; x < 3; x++) {
            for (int y = 0; y < 3; y++) {
                printf("%.1f ", inputs[i][x][y][0]);
            }
            printf("\n");
        }

        printf("Conv Output [0][0][0] = %.2f\n", model.conv_output[0][0][0]);
        printf("fc1[0:5] = ");
        for (int j = 0; j < 5; j++) printf("%.2f ", model.fc1_output[j]);
        printf("\nfc2[0:5] = ");
        for (int j = 0; j < 5; j++) printf("%.2f ", model.fc2_output[j]);
        printf("\n");
    }

    gettimeofday(&end, NULL);
    print_resource_usage(start, end);
    return 0;
}

