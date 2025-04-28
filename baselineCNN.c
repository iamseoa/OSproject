#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>
#include <sys/time.h>

#define NUM_INPUTS 9
#define CHANNELS 3
#define SIZE 256
#define VALID_SIZE (SIZE - 2)    // Conv2D valid 영역
#define POOL_SIZE (VALID_SIZE / 2)
#define FC1_SIZE 128
#define FC2_SIZE 20
#define FC1_INPUT_SIZE (16 * POOL_SIZE * POOL_SIZE)  // Flatten 벡터 길이

// 입력 스트림
double input_streams[NUM_INPUTS][CHANNELS][SIZE][SIZE] = {0};

void init_input_streams() {
    for (int n = 0; n < NUM_INPUTS; n++)
        for (int c = 0; c < CHANNELS; c++)
            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    input_streams[n][c][i][j] = (double)(n + 1);
}

double max(double a, double b) {
    return a > b ? a : b;
}

void process_input(double input[CHANNELS][SIZE][SIZE], int input_id) {
    // ======= 메모리 할당 =======
    double (*conv_output)[VALID_SIZE][VALID_SIZE] = malloc(sizeof(double) * 16 * VALID_SIZE * VALID_SIZE);
    double (*pool_output)[POOL_SIZE][POOL_SIZE] = malloc(sizeof(double) * 16 * POOL_SIZE * POOL_SIZE);
    double *flatten_output = malloc(sizeof(double) * FC1_INPUT_SIZE);
    double (*fc1_weights)[FC1_SIZE] = malloc(sizeof(double) * FC1_INPUT_SIZE * FC1_SIZE);
    double *fc1_bias = malloc(sizeof(double) * FC1_SIZE);
    double (*fc2_weights)[FC2_SIZE] = malloc(sizeof(double) * FC1_SIZE * FC2_SIZE);
    double *fc2_bias = malloc(sizeof(double) * FC2_SIZE);

    if (!conv_output || !pool_output || !flatten_output || !fc1_weights || !fc1_bias || !fc2_weights || !fc2_bias) {
        printf("Memory allocation failed!\n");
        exit(1);
    }

    // ======= Conv2D 필터 초기화 (홀짝 규칙) =======
    double conv_filter[16][CHANNELS][3][3];
    for (int f = 0; f < 16; f++)
        for (int c = 0; c < CHANNELS; c++)
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    conv_filter[f][c][i][j] = (i % 2 == 0) ? 0.5 : 0.0;

    // ======= Conv2D valid 연산 =======
    for (int f = 0; f < 16; f++) {
        for (int i = 1; i < SIZE-1; i++) {
            for (int j = 1; j < SIZE-1; j++) {
                double sum = 0.0;
                for (int c = 0; c < CHANNELS; c++)
                    for (int ki = -1; ki <= 1; ki++)
                        for (int kj = -1; kj <= 1; kj++)
                            sum += input[c][i+ki][j+kj] * conv_filter[f][c][ki+1][kj+1];
                conv_output[f][i-1][j-1] = sum;
            }
        }
    }

    // ======= ReLU =======
    for (int f = 0; f < 16; f++)
        for (int i = 0; i < VALID_SIZE; i++)
            for (int j = 0; j < VALID_SIZE; j++)
                if (conv_output[f][i][j] < 0) conv_output[f][i][j] = 0;

    // ======= MaxPooling 2x2 =======
    for (int f = 0; f < 16; f++)
        for (int i = 0; i < VALID_SIZE; i += 2)
            for (int j = 0; j < VALID_SIZE; j += 2) {
                double max_val = conv_output[f][i][j];
                max_val = max(max_val, conv_output[f][i+1][j]);
                max_val = max(max_val, conv_output[f][i][j+1]);
                max_val = max(max_val, conv_output[f][i+1][j+1]);
                pool_output[f][i/2][j/2] = max_val;
            }

    // ======= Flatten =======
    int idx = 0;
    for (int f = 0; f < 16; f++)
        for (int i = 0; i < POOL_SIZE; i++)
            for (int j = 0; j < POOL_SIZE; j++)
                flatten_output[idx++] = pool_output[f][i][j];

    // ======= FC1 초기화 (홀짝 규칙) =======
    for (int i = 0; i < FC1_INPUT_SIZE; i++)
        for (int j = 0; j < FC1_SIZE; j++)
            fc1_weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
    for (int j = 0; j < FC1_SIZE; j++)
        fc1_bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    // ======= FC1 연산 =======
    double fc1_output[FC1_SIZE] = {0};
    for (int j = 0; j < FC1_SIZE; j++) {
        for (int i = 0; i < FC1_INPUT_SIZE; i++)
            fc1_output[j] += flatten_output[i] * fc1_weights[i][j];
        fc1_output[j] += fc1_bias[j];
    }

    // ======= FC2 초기화 =======
    for (int i = 0; i < FC1_SIZE; i++)
        for (int j = 0; j < FC2_SIZE; j++)
            fc2_weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
    for (int j = 0; j < FC2_SIZE; j++)
        fc2_bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

    // ======= FC2 연산 =======
    double fc2_output[FC2_SIZE] = {0};
    for (int j = 0; j < FC2_SIZE; j++) {
        for (int i = 0; i < FC1_SIZE; i++)
            fc2_output[j] += fc1_output[i] * fc2_weights[i][j];
        fc2_output[j] += fc2_bias[j];
    }

    // ======= 출력 (Conv2D, FC1, FC2만 깔끔 출력) =======
    printf("\n===== Finished input stream #%d =====\n", input_id + 1);

    printf("Conv2D output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", conv_output[0][i][i]);
    printf("\n");

    printf("FC1 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc1_output[i]);
    printf("\n");

    printf("FC2 output sample: ");
    for (int i = 0; i < 5; i++) printf("%.5f ", fc2_output[i]);
    printf("\n");

    // 메모리 해제
    free(conv_output);
    free(pool_output);
    free(flatten_output);
    free(fc1_weights);
    free(fc1_bias);
    free(fc2_weights);
    free(fc2_bias);
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

int main() {
    init_input_streams();
    for (int idx = 0; idx < NUM_INPUTS; idx++) {
        process_input(input_streams[idx], idx);
    }
    print_resource_usage();
    return 0;
}
