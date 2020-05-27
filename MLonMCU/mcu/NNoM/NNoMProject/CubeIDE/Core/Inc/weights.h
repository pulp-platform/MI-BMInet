#include "nnom.h"

#define CONV2D_1_KERNEL_0 {5, 7, 4, -4, 3, 4, 12, 7, 1, -2, -6, 8, 2, -2, -5, 11, 11, 14, 15, 14, 34, 26, 37, 43, 31, 23, 19, 21, 23, 6, 17, 5, 4, 8, -6, -5, -10, -9, -12, -12, -12, -2, -8, -2, 1, 0, -3, 12, 7, 14, 16, 24, 44, 31, 34, 31, 30, 28, 21, 29, 28, 19, 26, 15, 7, 6, -13, -27, -25, -22, -16, -9, -7, 7, 1, 8, 10, 7, 3, -7, -25, -31, -32, -35, -38, -49, 4, -14, -13, -9, -21, -17, -23, -21, -20, -16, -19, -21, -17, -26, -22, -32, -29, -36, -42, -71, -90, -114, -98, -96, -75, -69, -55, -46, -52, -46, -34, -13, -15, 1, 8, 21, 14, 22, 12, 20, 21, 13, 13, 10, -3, -6, 7, 3, -3, 0, 8, 8, 9, 0, -4, -1, -3, -6, -6, 1, 5, -1, 7, 14, 8, 12, 11, 4, 25, 13, 10, 12, 4, 4, -2, -18, -9, -23, -28, -23, -29, -27, -32, -31, -21, -26, 4, -2, -7, -5, 4, 18, 25, 37, 32, 39, 16, 0, -20, -28, -40, -41, -22, -4, 25, 17, -13, -99, -109, -82, -77, -100, -95, -64, -52, -48, -24, 3, 28, 46, 57, 56, 57, 54, 32, 16, 15, -11, -37, -41, -35, -19, -17, -11, -5, 12, 16, 10, 22, 16, 3, 2, -3, -8, -11, -9, -16, -17, -38, -50, -45, -62, -100, -103, -79, -63, -55, -51, -66, -58, -53, -49, -30, -10, 11, 0, -1, 3, 11, 15, 28, 36, -11, -2, 1, 6, 19, 11, 8, 16, 17, 3, 7, 3, -17, -20, -25, -33, -49, -66, -75, -100, -119, -117, -101, -73, -46, -37, -24, -17, -2, -9, 5, 2, 0, 8, 7, 6, 0, 4, 11, 7, 13, 8, 8, 1, -10, -16, -16, -12, -11, -9, -8, -11, 2, 4, 9, 8, 12, 19, 29, 45, 41, 48, 55, 60, 58, 78, 98, 96, 91, 73, 55, 68, 70, 56, 49, 34, 20, 19, 11, 12, 17, 21, 8, 16, 7, 8}

#define CONV2D_1_KERNEL_0_SHIFT (6)

#define DEPTHWISE_CONV2D_1_DEPTHWISE_KERNEL_0 {13, -11, -4, 16, 12, -5, 23, -10, 2, -18, -15, -1, -16, 14, 29, -23, 2, -26, 2, 16, -28, 14, 10, -16, 12, -18, 17, 4, -5, 40, -27, -40, -20, -19, 12, 18, -17, 23, -1, -29, -6, -15, -16, -20, 26, 6, 28, -8, 12, -43, -25, -10, 54, -15, 17, 6, 10, -32, 12, -4, -50, -9, 0, -17, -19, -21, 19, -1, -24, 12, 0, -34, -8, -50, -44, -1, -13, -20, -6, 19, -25, -12, -38, -7, 27, -10, 2, -7, -35, -40, -10, -7, 15, -14, 25, 5, -14, -19, -10, -11, 15, -7, 8, -23, 6, -14, 7, -4, -6, 6, -7, -28, 3, -11, -5, -15, -4, 8, -1, -22, -10, 20, 13, -23, 0, 9, 29, -40, -11, -19, 5, -20, 10, -7, -1, -2, -14, 14, 9, -8, -21, 6, -4, -6, -28, -31, -8, 1, 37, -50, 41, 29, -18, -19, 13, 0, 8, 48, 2, -41, 3, -4, -31, 22, 11, -57, 8, 24, -16, -17, -6, 13, 30, -4, -13, -5, -19, 18, 67, 13, -21, 26, -2, -35, -3, -5, 7, -6, 10, 18, -17, -22, -2, -13, -13, 13, -12, 15, 1, -14, -64, 2, 0, 10, -26, 12, 9, 2, -22, -10, -36, -10, -12, -16, 5, 5, 6, -40, -41, -7, 1, -14, -40, 18, -5, 12, 7, -21, -2, -8, -4, -10, -12, -6, 5, -12, 15, 10, -18, -14, 2, 11, 12, -16, 28, 23, 7, -27, -37, -17, -10, 27, -7, -27, 3, 8, -26, -1, 2, 21, -10, -4, 3, -6, -12, -14, 16, 9, -19, 3, -38, -13, 18, -6, 18, -19, 23, 16, -62, -20, -26, -8, -1, 13, -13, -20, -19, 4, -27, 22, -2, 20, -5, -18, -14, 7, 12, 7, 26, 2, 10, 2, -32, -21, 9, 16, 23, 8, -19, -20, 32, -6, 0, 18, 9, 15, -1, 0, 28, -17, -3, 20, -13, -12, 29, 11, -3, -26, -12, 28, 3, -9, 23, 15, -31, -36, 3, 27, 26, -2, 10, 9, 0, -23, -7, 14, 17, 1, -34, -10, 10, -13, -7, 19, -37, -3, -48, -19, -12, 25, 21, 29, -10, 2, 56, 19, 2, -2, -11, 17, 15, 13, 14, 32, -2, -13, -16, 11, -36, 7, 13, -31, -11, 10, -15, 14, -20, 22, -10, -15, -3, -5, 3, 11, -27, 17, -12, -15, -8, 0, 11, 14, 9, 3, -16, 4, 28, -16, 0, 21, 1, 17, 16, 36, 7, -25, 3, 26, 0, 19, 10, 21, -8, -17, 18, 22, 7, 10, -16, 44, 14, -28, 28, 42, -26, 6, -7, 2, -6, 13, 11, 30, 8, -16, 15, -2, 36, -1, 26, 16, 5, 3, -41, -35, 34, 33, -34, 29, -23, 5, 18, 3, -34, -42, 19, 21, -8, -2, 6, -46, 23, 39, -1, 23, -7, 13, 3, -21, -4, -2, 2, 21, 13, -2, 8, 41, -16, -35, -25, 9, 17, -10, -5, 8, 10, -19, -10, 13, 8, 1, 27, -10, -17, -28, -13, 8, 56, -23, 13, -4, 36, -11, 29, 16, 14, -20, 5, -19, 14, 0, -35, 19, -42, -21, 5, -21, -48, -7, 22, 28, 34, 14, -2, 6, 21, -21, -8, 21, 5, 9, -18, 16, -23, -15, -10, 13, 29, 3, -42, 16, -5, -37, 8, 23, -19, -16, 14, -15, -16, 4, 0, 11, 19, -23, 5, -8, 5, -18, -5, 15, 8, -26, -3, 12, -15, 9, -19, 18, -8, 10, -5, 11, -38, -12, 17, 12, -16, -13, 22, -26, -1, 21, -4, 20, 14, 0, -18, -8, 18, 13, 3, 25, 0, 7, -4, 15, -14, 1}

#define DEPTHWISE_CONV2D_1_DEPTHWISE_KERNEL_0_SHIFT (7)

#define SEPARABLE_CONV2D_1_DEPTHWISE_KERNEL_0 {27, -16, -37, -21, -41, -28, -9, -3, 32, -70, 14, -59, 67, 35, 42, -19, 4, -29, -15, -11, -32, 0, 11, 1, -56, 37, 19, -35, 28, 23, 35, -23, -30, -22, 16, -5, -25, 6, 17, -21, -101, 125, 27, -22, 5, 11, 23, -22, -20, -16, 14, -2, -30, 23, 5, -32, -96, 108, 18, -31, -3, 6, 25, -9, -29, -14, 42, 14, -23, 28, 39, -38, -76, 86, 33, -19, -15, -3, 14, -10, -23, -26, 48, 6, -14, 8, 43, -39, -37, 34, 26, -11, -26, -4, 14, -14, -4, -8, 19, -3, -18, -11, 17, -16, 21, -23, 23, -8, -22, 7, 13, -24, 5, 12, -33, -12, -2, -36, -9, 4, 35, -33, 2, 2, -10, 37, 1, -3, 0, -6, -22, -20, 4, -53, -12, 2, 35, -49, -2, 5, 4, 5, -1, -19, 11, -7, -19, -22, 7, -86, -19, 15, 25, -36, -17, 3, -5, 13, -10, -8, 16, -9, -7, 1, -12, -76, -19, 28, 24, -15, -8, -2, 8, 31, 5, -12, 30, -18, -8, 11, 0, -35, -14, 16, 11, 2, 20, -19, 21, 39, 4, -9, 15, -32, 2, 24, -9, 10, 4, 18, 6, 3, 28, -11, 31, 33, 17, -7, -6, -34, 6, 6, -31, 34, 2, 20, 4, 12, 23, -31, 44, 38, 6, -20, -7, -24, -27, 8, -36, 19, 3, 5, 1, 4, 24, -41, 17, 32, 16, -30, 4, -2, -28, -21, -58, -44, -26, 13, 30, -34, 44, -62, 41, 38, 43, -17}

#define SEPARABLE_CONV2D_1_DEPTHWISE_KERNEL_0_SHIFT (6)

#define SEPARABLE_CONV2D_1_POINTWISE_KERNEL_0 {27, -16, -37, -21, -41, -28, -9, -3, 32, -70, 14, -59, 67, 35, 42, -19, 4, -29, -15, -11, -32, 0, 11, 1, -56, 37, 19, -35, 28, 23, 35, -23, -30, -22, 16, -5, -25, 6, 17, -21, -101, 125, 27, -22, 5, 11, 23, -22, -20, -16, 14, -2, -30, 23, 5, -32, -96, 108, 18, -31, -3, 6, 25, -9, -29, -14, 42, 14, -23, 28, 39, -38, -76, 86, 33, -19, -15, -3, 14, -10, -23, -26, 48, 6, -14, 8, 43, -39, -37, 34, 26, -11, -26, -4, 14, -14, -4, -8, 19, -3, -18, -11, 17, -16, 21, -23, 23, -8, -22, 7, 13, -24, 5, 12, -33, -12, -2, -36, -9, 4, 35, -33, 2, 2, -10, 37, 1, -3, 0, -6, -22, -20, 4, -53, -12, 2, 35, -49, -2, 5, 4, 5, -1, -19, 11, -7, -19, -22, 7, -86, -19, 15, 25, -36, -17, 3, -5, 13, -10, -8, 16, -9, -7, 1, -12, -76, -19, 28, 24, -15, -8, -2, 8, 31, 5, -12, 30, -18, -8, 11, 0, -35, -14, 16, 11, 2, 20, -19, 21, 39, 4, -9, 15, -32, 2, 24, -9, 10, 4, 18, 6, 3, 28, -11, 31, 33, 17, -7, -6, -34, 6, 6, -31, 34, 2, 20, 4, 12, 23, -31, 44, 38, 6, -20, -7, -24, -27, 8, -36, 19, 3, 5, 1, 4, 24, -41, 17, 32, 16, -30, 4, -2, -28, -21, -58, -44, -26, 13, 30, -34, 44, -62, 41, 38, 43, -17}

#define SEPARABLE_CONV2D_1_POINTWISE_KERNEL_0_SHIFT (6)

#define DENSE_KERNEL_0 {65, 59, -59, -55, 59, 66, -59, 32, -65, 68, -62, 68, -58, -38, 63, 65, -19, -26, 21, 22, -23, -28, -16, -46, 20, -9, 14, -12, 20, -23, -23, -27, -65, -59, 59, 55, -59, -66, 59, -32, 65, -68, 62, -68, 58, 38, -63, -65, 19, 26, -21, -22, 23, 28, 16, 46, -20, 9, -14, 12, -20, 23, 23, 27}

#define DENSE_KERNEL_0_SHIFT (10)

#define DENSE_BIAS_0 {67, -67}

#define DENSE_BIAS_0_SHIFT (9)



/* output enconding for each layer */
#define INPUT_1_OUTPUT_SHIFT 7
#define CONV2D_1_OUTPUT_SHIFT 4
#define BATCH_NORMALIZATION_1_OUTPUT_SHIFT 4
#define DEPTHWISE_CONV2D_1_OUTPUT_SHIFT 4
#define BATCH_NORMALIZATION_2_OUTPUT_SHIFT 4
#define ACTIVATION_1_OUTPUT_SHIFT 4
#define AVERAGE_POOLING2D_1_OUTPUT_SHIFT 4
#define DROPOUT_1_OUTPUT_SHIFT 4
#define SEPARABLE_CONV2D_1_OUTPUT_SHIFT 2
#define BATCH_NORMALIZATION_3_OUTPUT_SHIFT 2
#define ACTIVATION_2_OUTPUT_SHIFT 2
#define AVERAGE_POOLING2D_2_OUTPUT_SHIFT 2
#define DROPOUT_2_OUTPUT_SHIFT 2
#define FLATTEN_OUTPUT_SHIFT 2
#define DENSE_OUTPUT_SHIFT 4
#define SOFTMAX_OUTPUT_SHIFT 7

/* bias shift and output shift for each layer */
#define DENSE_OUTPUT_RSHIFT (FLATTEN_OUTPUT_SHIFT+DENSE_KERNEL_0_SHIFT-DENSE_OUTPUT_SHIFT)
#define DENSE_BIAS_LSHIFT   (FLATTEN_OUTPUT_SHIFT+DENSE_KERNEL_0_SHIFT-DENSE_BIAS_0_SHIFT)
#if DENSE_OUTPUT_RSHIFT < 0
#error DENSE_OUTPUT_RSHIFT must be bigger than 0
#endif
#if DENSE_BIAS_LSHIFT < 0
#error DENSE_BIAS_RSHIFT must be bigger than 0
#endif

/* weights for each layer */
static const int8_t conv2d_1_weights[] = CONV2D_1_KERNEL_0;
static const nnom_weight_t conv2d_1_w = { (const void*)conv2d_1_weights, CONV2D_1_OUTPUT_RSHIFT};
static const int8_t depthwise_conv2d_1_weights[] = DEPTHWISE_CONV2D_1_DEPTHWISE_KERNEL_0;
static const nnom_weight_t depthwise_conv2d_1_w = { (const void*)depthwise_conv2d_1_weights, DEPTHWISE_CONV2D_1_OUTPUT_RSHIFT};
static const int8_t separable_conv2d_1_weights[] = SEPARABLE_CONV2D_1_DEPTHWISE_KERNEL_0;
static const nnom_weight_t separable_conv2d_1_w = { (const void*)separable_conv2d_1_weights, SEPARABLE_CONV2D_1_OUTPUT_RSHIFT};
static const int8_t separable_conv2d_1_weights[] = SEPARABLE_CONV2D_1_POINTWISE_KERNEL_0;
static const nnom_weight_t separable_conv2d_1_w = { (const void*)separable_conv2d_1_weights, SEPARABLE_CONV2D_1_OUTPUT_RSHIFT};
static const int8_t dense_weights[] = DENSE_KERNEL_0;
static const nnom_weight_t dense_w = { (const void*)dense_weights, DENSE_OUTPUT_RSHIFT};
static const int8_t dense_bias[] = DENSE_BIAS_0;
static const nnom_bias_t dense_b = { (const void*)dense_bias, DENSE_BIAS_LSHIFT};

/* nnom model */
static int8_t nnom_input_data[2052];
static int8_t nnom_output_data[2];
static nnom_model_t* nnom_model_create(void)
{
	static nnom_model_t model;
	nnom_layer_t* layer[11];

	new_model(&model);

	layer[0] = Input(shape[], nnom_input_data);
	layer[1] = model.hook(Conv2D(8, kernel(1, 43), stride(1, 1), PADDING_SAME, &conv2d_1_w, &conv2d_1_b), layer[0]);
	layer[2] = model.hook(DW_Conv2D(1, kernel(38, 1), stride(1, 1), PADDING_VALID, &depthwise_conv2d_1_w, &depthwise_conv2d_1_b), layer[1]);
	layer[4] = model.hook(AvgPool(kernel(1, 3), stride(1, 3), PADDING_VALID), layer[3]);
	layer[5] = model.hook(Conv2D(16, kernel(1, 16), stride(1, 1), PADDING_SAME, &separable_conv2d_1_w, &separable_conv2d_1_b), layer[4]);
	layer[7] = model.hook(AvgPool(kernel(1, 8), stride(1, 8), PADDING_VALID), layer[6]);
	layer[8] = model.hook(Dense(2, &dense_w, &dense_b), layer[7]);
	layer[9] = model.hook(Softmax(), layer[8]);
	layer[10] = model.hook(Output(shape(2,1,1), nnom_output_data), layer[9]);
	model_compile(&model, layer[0], layer[10]);
	return &model;
}
