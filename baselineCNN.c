#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <time.h>

#define NUM_INPUTS 9
#define CHANNELS 3
#define SIZE 256
#define FC1_SIZE 128
#define FC2_SIZE 20
#define FC1_INPUT_SIZE (16 * (SIZE/2) * (SIZE/2))  // Flatten한 크기

void print_resource_usage() {
    struct rusage usage;
    clock_t now = clock();
    double elapsed = (double)now / CLOCKS_PER_SEC;

    getrusage(RUSAGE_SELF, &usage);

    printf("\n=== Resource Usage ===\n");
    printf("Elapsed Real Time: %.6f seconds\n", elapsed);
    printf("User CPU Time: %.6f seconds\n", usage.ru_utime.tv_sec + usage.ru_utime.tv_usec/1e6);
    printf("System CPU Time: %.6f seconds\n", usage.ru_stime.tv_sec + usage.ru_stime.tv_usec/1e6);
    printf("Max RSS (memory): %ld KB\n", usage.ru_maxrss);
    printf("Voluntary Context Switches: %ld\n", usage.ru_nvcsw);
    printf("Involuntary Context Switches: %ld\n", usage.ru_nivcsw);
    printf("Minor Page Faults: %ld\n", usage.ru_minflt);
    printf("Major Page Faults: %ld\n", usage.ru_majflt);
}

// 입력 스트림
double input_streams[NUM_INPUTS][CHANNELS][SIZE][SIZE] = {0};

// 입력 스트림 초기화 (1~9)
void init_input_streams() {
    for (int n = 0; n < NUM_INPUTS; n++) {
        for (int c = 0; c < CHANNELS; c++) {
            for (int i = 0; i < SIZE; i++) {
                for (int j = 0; j < SIZE; j++) {
                    input_streams[n][c][i][j] = (double)(n + 1);  // 1~9로 입력값 설정
                }
            }
        }
    }
}

// 최대값 함수
double max(double a, double b) {
    return a > b ? a : b;
}

// Softmax 함수
void softmax(double* input, int size, double* output) {
    double max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Softmax 결과에서 argmax 찾기
int find_argmax(double* output, int size) {
    int max_idx = 0;
    double max_val = output[0];
    for (int i = 1; i < size; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            max_idx = i;
        }
    }
    return max_idx;
}

// 하나의 입력 스트림 처리
void process_input(double input[CHANNELS][SIZE][SIZE], int input_id) {
    // ======= 동적 메모리 할당 =======
    double (*conv_output)[SIZE][SIZE] = malloc(sizeof(double) * 16 * SIZE * SIZE);
    double (*pool_output)[SIZE/2][SIZE/2] = malloc(sizeof(double) * 16 * (SIZE/2) * (SIZE/2));
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
    for (int f = 0; f < 16; f++) {
        for (int c = 0; c < CHANNELS; c++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    conv_filter[f][c][i][j] = (i % 2 == 0) ? 0.5 : 0.0;
                }
            }
        }
    }

    // ======= Conv2D 연산 (padding 추가) =======
    for (int f = 0; f < 16; f++) {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                double sum = 0.0;
                for (int c = 0; c < CHANNELS; c++) {
                    for (int ki = -1; ki <= 1; ki++) {
                        for (int kj = -1; kj <= 1; kj++) {
                            int ni = i + ki;
                            int nj = j + kj;
                            double val = 0.0;
                            if (ni >= 0 && ni < SIZE && nj >= 0 && nj < SIZE) {
                                val = input[c][ni][nj];
                            }
                            sum += val * conv_filter[f][c][ki+1][kj+1];
                        }
                    }
                }
                conv_output[f][i][j] = sum;
            }
        }
    }

    // ======= ReLU =======
    for (int f = 0; f < 16; f++) {
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                if (conv_output[f][i][j] < 0)
                    conv_output[f][i][j] = 0;
            }
        }
    }

    // ======= MaxPooling =======
    for (int f = 0; f < 16; f++) {
        for (int i = 0; i < SIZE; i += 2) {
            for (int j = 0; j < SIZE; j += 2) {
                double max_val = conv_output[f][i][j];
                max_val = max(max_val, conv_output[f][i+1][j]);
                max_val = max(max_val, conv_output[f][i][j+1]);
                max_val = max(max_val, conv_output[f][i+1][j+1]);
                pool_output[f][i/2][j/2] = max_val;
            }
        }
    }

    // ======= Flatten =======
    int idx = 0;
    for (int f = 0; f < 16; f++) {
        for (int i = 0; i < SIZE/2; i++) {
            for (int j = 0; j < SIZE/2; j++) {
                flatten_output[idx++] = pool_output[f][i][j];
            }
        }
    }

    // ======= FC1 초기화 (홀짝 규칙) =======
    for (int i = 0; i < FC1_INPUT_SIZE; i++) {
        for (int j = 0; j < FC1_SIZE; j++) {
            fc1_weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
        }
    }
    for (int j = 0; j < FC1_SIZE; j++) {
        fc1_bias[j] = (j % 2 == 0) ? 0.5 : 0.0;
    }

    // ======= FC1 연산 =======
    double fc1_output[FC1_SIZE] = {0};
    for (int j = 0; j < FC1_SIZE; j++) {
        for (int i = 0; i < FC1_INPUT_SIZE; i++) {
            fc1_output[j] += flatten_output[i] * fc1_weights[i][j];
        }
        fc1_output[j] += fc1_bias[j];
    }

    // ======= FC2 초기화 (홀짝 규칙) =======
    for (int i = 0; i < FC1_SIZE; i++) {
        for (int j = 0; j < FC2_SIZE; j++) {
            fc2_weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
        }
    }
    for (int j = 0; j < FC2_SIZE; j++) {
        fc2_bias[j] = (j % 2 == 0) ? 0.5 : 0.0;
    }

    // ======= FC2 연산 =======
    double fc2_output[FC2_SIZE] = {0};
    for (int j = 0; j < FC2_SIZE; j++) {
        for (int i = 0; i < FC1_SIZE; i++) {
            fc2_output[j] += fc1_output[i] * fc2_weights[i][j];
        }
        fc2_output[j] += fc2_bias[j];
    }

    // ======= Softmax =======
    double softmax_output[FC2_SIZE];
    softmax(fc2_output, FC2_SIZE, softmax_output);

    // ======= 출력 =======
    printf("\nSoftmax output for input stream #%d:\n", input_id + 1);
    for (int i = 0; i < FC2_SIZE; i++) {
        printf("%.5f ", softmax_output[i]);
    }
    printf("\n");

    int predicted_class = find_argmax(softmax_output, FC2_SIZE);
    printf("Predicted class: %d (Input ID: %d)\n", predicted_class, input_id);

    // ======= 메모리 해제 =======
    free(conv_output);
    free(pool_output);
    free(flatten_output);
    free(fc1_weights);
    free(fc1_bias);
    free(fc2_weights);
    free(fc2_bias);
}

int main() {
    srand(0);
    init_input_streams();

    for (int idx = 0; idx < NUM_INPUTS; idx++) {
        process_input(input_streams[idx], idx);
    }
    print_resource_usage();

    return 0;
}
