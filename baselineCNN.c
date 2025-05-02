#include "layers.h"
#include <sys/resource.h>

#define NUM_INPUTS 9

double input_streams[NUM_INPUTS][3][SIZE][SIZE] = {0};

void init_input_streams() {
    for (int n = 0; n < NUM_INPUTS; n++)
        for (int c = 0; c < 3; c++)
            for (int i = 0; i < SIZE; i++)
                for (int j = 0; j < SIZE; j++)
                    input_streams[n][c][i][j] = (double)(n + 1);
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

    // 9개의 입력 스트림을 병렬로 처리하기 위해 9개의 스레드를 생성합니다
    pthread_t threads[NUM_INPUTS];

    // 각 스레드에 전달할 인덱스 구조체
    typedef struct {
        int idx;
    } ThreadArg;

    // 스레드가 처리할 작업 정의
    // 각 스레드는 하나의 입력 스트림을 CNN 레이어를 통해 처리합니다
    void* thread_worker(void* arg) {
        ThreadArg* t_arg = (ThreadArg*)arg;
        int idx = t_arg->idx;
        free(t_arg); // 전달된 구조체 메모리 해제

        // CNN 레이어들을 개별적으로 초기화하여 스레드 간 충돌 방지
        Conv2DLayer conv_layer;
        conv_layer.in_channels = 3;
        conv_layer.out_channels = 16;
        conv_layer.kernel_size = 3;

        // Conv2D 레이어 weight 초기화
        conv_layer.weights = (double****)malloc(sizeof(double***) * conv_layer.out_channels);
        for (int f = 0; f < conv_layer.out_channels; f++) {
            conv_layer.weights[f] = (double***)malloc(sizeof(double**) * conv_layer.in_channels);
            for (int c = 0; c < conv_layer.in_channels; c++) {
                conv_layer.weights[f][c] = (double**)malloc(sizeof(double*) * 3);
                for (int i = 0; i < 3; i++) {
                    conv_layer.weights[f][c][i] = (double*)malloc(sizeof(double) * 3);
                    for (int j = 0; j < 3; j++) {
                        conv_layer.weights[f][c][i][j] = (i % 2 == 0) ? 0.5 : 0.0;
                    }
                }
            }
        }

        MaxPool2DLayer pool_layer = {2};

        // FullyConnected1 레이어 초기화
        FullyConnected1Layer fc1_layer;
        fc1_layer.input_size = FC1_INPUT_SIZE;
        fc1_layer.output_size = FC1_SIZE;
        fc1_layer.weights = (double**)malloc(sizeof(double*) * FC1_INPUT_SIZE);
        for (int i = 0; i < FC1_INPUT_SIZE; i++) {
            fc1_layer.weights[i] = (double*)malloc(sizeof(double) * FC1_SIZE);
            for (int j = 0; j < FC1_SIZE; j++)
                fc1_layer.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
        }
        fc1_layer.bias = (double*)malloc(sizeof(double) * FC1_SIZE);
        for (int j = 0; j < FC1_SIZE; j++)
            fc1_layer.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

        // FullyConnected2 레이어 초기화
        FullyConnected2Layer fc2_layer;
        fc2_layer.input_size = FC1_SIZE;
        fc2_layer.output_size = FC2_SIZE;
        fc2_layer.weights = (double**)malloc(sizeof(double*) * FC1_SIZE);
        for (int i = 0; i < FC1_SIZE; i++) {
            fc2_layer.weights[i] = (double*)malloc(sizeof(double) * FC2_SIZE);
            for (int j = 0; j < FC2_SIZE; j++)
                fc2_layer.weights[i][j] = (i % 2 == 0) ? 0.5 : 0.0;
        }
        fc2_layer.bias = (double*)malloc(sizeof(double) * FC2_SIZE);
        for (int j = 0; j < FC2_SIZE; j++)
            fc2_layer.bias[j] = (j % 2 == 0) ? 0.5 : 0.0;

        // 출력 및 중간 결과 저장을 위한 버퍼
        double (*conv_output)[VALID_SIZE][VALID_SIZE] = malloc(sizeof(double) * 16 * VALID_SIZE * VALID_SIZE);
        double (*pool_output)[POOL_SIZE][POOL_SIZE] = malloc(sizeof(double) * 16 * POOL_SIZE * POOL_SIZE);
        double* flatten_output = (double*)malloc(sizeof(double) * FC1_INPUT_SIZE);
        double* fc1_output = (double*)malloc(sizeof(double) * FC1_SIZE);
        double* fc2_output = (double*)malloc(sizeof(double) * FC2_SIZE);

        // 입력 스트림 idx 번째에 대한 CNN 연산 수행
        printf("\n===== Finished input stream #%d =====\n", idx + 1);
        conv2d_forward(&conv_layer, input_streams[idx], conv_output);
        relu_forward(conv_output);
        maxpool2d_forward(&pool_layer, conv_output, pool_output);
        flatten_forward(pool_output, flatten_output);
        fc1_forward(&fc1_layer, flatten_output, fc1_output);
        fc2_forward(&fc2_layer, fc1_output, fc2_output);

        // 출력 결과 샘플
        printf("Conv2D output sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", conv_output[0][i][i]);
        printf("\n");
        printf("FC1 output sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", fc1_output[i]);
        printf("\n");
        printf("FC2 output sample: ");
        for (int i = 0; i < 5; i++) printf("%.5f ", fc2_output[i]);
        printf("\n");

        // 메모리 해제 (스레드 로컬)
        free(flatten_output); free(fc1_output); free(fc2_output);
        free(conv_output); free(pool_output);
        for (int f = 0; f < conv_layer.out_channels; f++) {
            for (int c = 0; c < conv_layer.in_channels; c++) {
                for (int i = 0; i < 3; i++) free(conv_layer.weights[f][c][i]);
                free(conv_layer.weights[f][c]);
            }
            free(conv_layer.weights[f]);
        }
        free(conv_layer.weights);
        for (int i = 0; i < fc1_layer.input_size; i++) free(fc1_layer.weights[i]);
        free(fc1_layer.weights); free(fc1_layer.bias);
        for (int i = 0; i < fc2_layer.input_size; i++) free(fc2_layer.weights[i]);
        free(fc2_layer.weights); free(fc2_layer.bias);

        return NULL;
    }

    // 각 입력 스트림에 대해 하나의 스레드를 생성하여 CNN 연산 병렬 수행
    for (int i = 0; i < NUM_INPUTS; i++) {
        ThreadArg* arg = malloc(sizeof(ThreadArg));
        arg->idx = i;
        pthread_create(&threads[i], NULL, thread_worker, arg);
    }

    // 모든 스레드가 종료될 때까지 대기
    for (int i = 0; i < NUM_INPUTS; i++) pthread_join(threads[i], NULL);

    print_resource_usage(); // 전체 리소스 사용량 출력

    return 0;
}
