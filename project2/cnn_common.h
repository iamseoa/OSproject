#ifndef CNN_COMMON_H
#define CNN_COMMON_H

#define INPUT_SIZE 224
#define CHANNELS 3
#define KERNEL_SIZE 3
#define CONV_DEPTH 16 
#define CONV_OUT (INPUT_SIZE - KERNEL_SIZE + 1)
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

void initialize_weights(CNNModel* model);
void initialize_input(float input[INPUT_SIZE][INPUT_SIZE][CHANNELS], int id);
void conv_forward(CNNModel* model, float input[INPUT_SIZE][INPUT_SIZE][CHANNELS], float output[CONV_OUT][CONV_OUT][CONV_DEPTH]);
void relu_forward(float in[CONV_OUT][CONV_OUT][CONV_DEPTH], float out[CONV_OUT][CONV_OUT][CONV_DEPTH]);
void maxpool_forward(float in[CONV_OUT][CONV_OUT][CONV_DEPTH], float out[CONV_OUT / 2][CONV_OUT / 2][CONV_DEPTH]);
void flatten(float in[CONV_OUT / 2][CONV_OUT / 2][CONV_DEPTH], float out[]);
void fc1_forward(CNNModel* model, float input[], float output[FC1_OUT]);
void fc2_forward(CNNModel* model, float input[], float output[FC2_OUT]);

#endif // CNN_COMMON_H
