/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Tue May 26 17:33:01 2020
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2018 STMicroelectronics.
  * All rights reserved.
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */


#include "network.h"

#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "layers.h"

#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#define AI_TOOLS_VERSION_MAJOR 5
#define AI_TOOLS_VERSION_MINOR 0
#define AI_TOOLS_VERSION_MICRO 0


#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#define AI_TOOLS_API_VERSION_MAJOR 1
#define AI_TOOLS_API_VERSION_MINOR 3
#define AI_TOOLS_API_VERSION_MICRO 0

#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "4b6ab6a08d2f7a1ad6d0702a32d048cb"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     "(rev-5.0.0)"
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Tue May 26 17:33:01 2020"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

/**  Forward network declaration section  *************************************/
AI_STATIC ai_network AI_NET_OBJ_INSTANCE;


/**  Forward network array declarations  **************************************/
AI_STATIC ai_array separable_conv2d_1_conv2d_scratch0_array;   /* Array #0 */
AI_STATIC ai_array depthwise_conv2d_1_scratch0_array;   /* Array #1 */
AI_STATIC ai_array dense_bias_array;   /* Array #2 */
AI_STATIC ai_array dense_weights_array;   /* Array #3 */
AI_STATIC ai_array separable_conv2d_1_conv2d_bias_array;   /* Array #4 */
AI_STATIC ai_array separable_conv2d_1_conv2d_weights_array;   /* Array #5 */
AI_STATIC ai_array separable_conv2d_1_bias_array;   /* Array #6 */
AI_STATIC ai_array separable_conv2d_1_weights_array;   /* Array #7 */
AI_STATIC ai_array depthwise_conv2d_1_bias_array;   /* Array #8 */
AI_STATIC ai_array depthwise_conv2d_1_weights_array;   /* Array #9 */
AI_STATIC ai_array conv2d_1_bias_array;   /* Array #10 */
AI_STATIC ai_array conv2d_1_weights_array;   /* Array #11 */
AI_STATIC ai_array input_1_output_array;   /* Array #12 */
AI_STATIC ai_array conv2d_1_output_array;   /* Array #13 */
AI_STATIC ai_array depthwise_conv2d_1_output_array;   /* Array #14 */
AI_STATIC ai_array separable_conv2d_1_output_array;   /* Array #15 */
AI_STATIC ai_array separable_conv2d_1_conv2d_output_array;   /* Array #16 */
AI_STATIC ai_array dense_output_array;   /* Array #17 */
AI_STATIC ai_array softmax_output_array;   /* Array #18 */


/**  Forward network tensor declarations  *************************************/
AI_STATIC ai_tensor separable_conv2d_1_conv2d_scratch0;   /* Tensor #0 */
AI_STATIC ai_tensor depthwise_conv2d_1_scratch0;   /* Tensor #1 */
AI_STATIC ai_tensor dense_bias;   /* Tensor #2 */
AI_STATIC ai_tensor dense_weights;   /* Tensor #3 */
AI_STATIC ai_tensor separable_conv2d_1_conv2d_bias;   /* Tensor #4 */
AI_STATIC ai_tensor separable_conv2d_1_conv2d_weights;   /* Tensor #5 */
AI_STATIC ai_tensor separable_conv2d_1_bias;   /* Tensor #6 */
AI_STATIC ai_tensor separable_conv2d_1_weights;   /* Tensor #7 */
AI_STATIC ai_tensor depthwise_conv2d_1_bias;   /* Tensor #8 */
AI_STATIC ai_tensor depthwise_conv2d_1_weights;   /* Tensor #9 */
AI_STATIC ai_tensor conv2d_1_bias;   /* Tensor #10 */
AI_STATIC ai_tensor conv2d_1_weights;   /* Tensor #11 */
AI_STATIC ai_tensor input_1_output;   /* Tensor #12 */
AI_STATIC ai_tensor conv2d_1_output;   /* Tensor #13 */
AI_STATIC ai_tensor depthwise_conv2d_1_output;   /* Tensor #14 */
AI_STATIC ai_tensor separable_conv2d_1_output;   /* Tensor #15 */
AI_STATIC ai_tensor separable_conv2d_1_conv2d_output;   /* Tensor #16 */
AI_STATIC ai_tensor separable_conv2d_1_conv2d_output0;   /* Tensor #17 */
AI_STATIC ai_tensor dense_output;   /* Tensor #18 */
AI_STATIC ai_tensor softmax_output;   /* Tensor #19 */


/**  Forward network tensor chain declarations  *******************************/
AI_STATIC_CONST ai_tensor_chain conv2d_1_chain;   /* Chain #0 */
AI_STATIC_CONST ai_tensor_chain depthwise_conv2d_1_chain;   /* Chain #1 */
AI_STATIC_CONST ai_tensor_chain separable_conv2d_1_chain;   /* Chain #2 */
AI_STATIC_CONST ai_tensor_chain separable_conv2d_1_conv2d_chain;   /* Chain #3 */
AI_STATIC_CONST ai_tensor_chain dense_chain;   /* Chain #4 */
AI_STATIC_CONST ai_tensor_chain softmax_chain;   /* Chain #5 */


/**  Forward network layer declarations  **************************************/
AI_STATIC ai_layer_conv2d conv2d_1_layer; /* Layer #0 */
AI_STATIC ai_layer_conv2d_nl_pool depthwise_conv2d_1_layer; /* Layer #1 */
AI_STATIC ai_layer_conv2d separable_conv2d_1_layer; /* Layer #2 */
AI_STATIC ai_layer_conv2d_nl_pool separable_conv2d_1_conv2d_layer; /* Layer #3 */
AI_STATIC ai_layer_dense dense_layer; /* Layer #4 */
AI_STATIC ai_layer_nl softmax_layer; /* Layer #5 */


/**  Array declarations section  **********************************************/
AI_ARRAY_OBJ_DECLARE(
    separable_conv2d_1_conv2d_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 288,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    depthwise_conv2d_1_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 864,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 2,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 64,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    separable_conv2d_1_conv2d_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 16,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    separable_conv2d_1_conv2d_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 256,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    separable_conv2d_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 16,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    separable_conv2d_1_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 256,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    depthwise_conv2d_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 16,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    depthwise_conv2d_1_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 608,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 8,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_1_weights_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 344,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    input_1_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
    NULL, NULL, 2052,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    conv2d_1_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 16416,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    depthwise_conv2d_1_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 288,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    separable_conv2d_1_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 288,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    separable_conv2d_1_conv2d_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 32,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    dense_output_array, AI_ARRAY_FORMAT_FLOAT,
    NULL, NULL, 2,
     AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
    softmax_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
    NULL, NULL, 2,
     AI_STATIC)




/**  Tensor declarations section  *********************************************/
AI_TENSOR_OBJ_DECLARE(
  separable_conv2d_1_conv2d_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 18, 1), AI_STRIDE_INIT(4, 4, 4, 64, 1152),
  1, &separable_conv2d_1_conv2d_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_1_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 54, 1), AI_STRIDE_INIT(4, 4, 4, 64, 3456),
  1, &depthwise_conv2d_1_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &dense_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 32, 2, 1, 1), AI_STRIDE_INIT(4, 4, 128, 256, 256),
  1, &dense_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  separable_conv2d_1_conv2d_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &separable_conv2d_1_conv2d_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  separable_conv2d_1_conv2d_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 16, 1, 1, 16), AI_STRIDE_INIT(4, 4, 64, 64, 64),
  1, &separable_conv2d_1_conv2d_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  separable_conv2d_1_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &separable_conv2d_1_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  separable_conv2d_1_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 1, 16), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &separable_conv2d_1_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_1_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &depthwise_conv2d_1_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_1_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1, 38, 16), AI_STRIDE_INIT(4, 4, 4, 4, 152),
  1, &depthwise_conv2d_1_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_1_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 43, 1, 8), AI_STRIDE_INIT(4, 4, 4, 172, 172),
  1, &conv2d_1_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  input_1_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1, 54, 38), AI_STRIDE_INIT(4, 4, 4, 4, 216),
  1, &input_1_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 54, 38), AI_STRIDE_INIT(4, 4, 4, 32, 1728),
  1, &conv2d_1_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  depthwise_conv2d_1_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 18, 1), AI_STRIDE_INIT(4, 4, 4, 64, 1152),
  1, &depthwise_conv2d_1_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  separable_conv2d_1_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 18, 1), AI_STRIDE_INIT(4, 4, 4, 64, 1152),
  1, &separable_conv2d_1_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  separable_conv2d_1_conv2d_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 2, 1), AI_STRIDE_INIT(4, 4, 4, 64, 128),
  1, &separable_conv2d_1_conv2d_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  separable_conv2d_1_conv2d_output0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &separable_conv2d_1_conv2d_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &dense_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  softmax_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 2, 1, 1), AI_STRIDE_INIT(4, 4, 4, 8, 8),
  1, &softmax_output_array, NULL)


/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&input_1_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_1_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 1,
  CONV2D_TYPE,
  conv2d, forward_conv2d,
  &AI_NET_OBJ_INSTANCE, &depthwise_conv2d_1_layer, AI_STATIC,
  .tensors = &conv2d_1_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 21, 0, 21), 
)


AI_STATIC_CONST ai_float depthwise_conv2d_1_nl_params_data[] = { 1.0 };
AI_ARRAY_OBJ_DECLARE(
    depthwise_conv2d_1_nl_params, AI_ARRAY_FORMAT_FLOAT, depthwise_conv2d_1_nl_params_data,
    depthwise_conv2d_1_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  depthwise_conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_1_output),
  AI_TENSOR_LIST_ENTRY(&depthwise_conv2d_1_output),
  AI_TENSOR_LIST_ENTRY(&depthwise_conv2d_1_weights, &depthwise_conv2d_1_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&depthwise_conv2d_1_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  depthwise_conv2d_1_layer, 3,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &AI_NET_OBJ_INSTANCE, &separable_conv2d_1_layer, AI_STATIC,
  .tensors = &depthwise_conv2d_1_chain, 
  .groups = 8, 
  .nl_params = &depthwise_conv2d_1_nl_params, 
  .nl_func = nl_func_elu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(3, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(3, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_ap_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  separable_conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&depthwise_conv2d_1_output),
  AI_TENSOR_LIST_ENTRY(&separable_conv2d_1_output),
  AI_TENSOR_LIST_ENTRY(&separable_conv2d_1_weights, &separable_conv2d_1_bias, NULL),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  separable_conv2d_1_layer, 8,
  CONV2D_TYPE,
  conv2d, forward_conv2d,
  &AI_NET_OBJ_INSTANCE, &separable_conv2d_1_conv2d_layer, AI_STATIC,
  .tensors = &separable_conv2d_1_chain, 
  .groups = 16, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 7, 0, 8), 
)


AI_STATIC_CONST ai_float separable_conv2d_1_conv2d_nl_params_data[] = { 1.0 };
AI_ARRAY_OBJ_DECLARE(
    separable_conv2d_1_conv2d_nl_params, AI_ARRAY_FORMAT_FLOAT, separable_conv2d_1_conv2d_nl_params_data,
    separable_conv2d_1_conv2d_nl_params_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  separable_conv2d_1_conv2d_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&separable_conv2d_1_output),
  AI_TENSOR_LIST_ENTRY(&separable_conv2d_1_conv2d_output),
  AI_TENSOR_LIST_ENTRY(&separable_conv2d_1_conv2d_weights, &separable_conv2d_1_conv2d_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&separable_conv2d_1_conv2d_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  separable_conv2d_1_conv2d_layer, 8,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &AI_NET_OBJ_INSTANCE, &dense_layer, AI_STATIC,
  .tensors = &separable_conv2d_1_conv2d_chain, 
  .groups = 1, 
  .nl_params = &separable_conv2d_1_conv2d_nl_params, 
  .nl_func = nl_func_elu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(8, 1), 
  .pool_stride = AI_SHAPE_2D_INIT(8, 1), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_ap_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&separable_conv2d_1_conv2d_output0),
  AI_TENSOR_LIST_ENTRY(&dense_output),
  AI_TENSOR_LIST_ENTRY(&dense_weights, &dense_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_layer, 14,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &softmax_layer, AI_STATIC,
  .tensors = &dense_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  softmax_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&dense_output),
  AI_TENSOR_LIST_ENTRY(&softmax_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  softmax_layer, 15,
  NL_TYPE,
  nl, forward_sm,
  &AI_NET_OBJ_INSTANCE, &softmax_layer, AI_STATIC,
  .tensors = &softmax_chain, 
)


AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 6344, 1,
                     NULL),
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 70272, 1,
                     NULL),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_1_output),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &softmax_output),
  &conv2d_1_layer, 0, NULL)



AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_buffer* activation_buffer)
{
  AI_ASSERT(net_ctx &&  activation_buffer && activation_buffer->data)

  ai_ptr activations = AI_PTR(AI_PTR_ALIGN(activation_buffer->data, 4));
  AI_ASSERT(activations)
  AI_UNUSED(net_ctx)

  {
    /* Updating activations (byte) offsets */
    separable_conv2d_1_conv2d_scratch0_array.data = AI_PTR(activations + 1152);
    separable_conv2d_1_conv2d_scratch0_array.data_start = AI_PTR(activations + 1152);
    depthwise_conv2d_1_scratch0_array.data = AI_PTR(activations + 65664);
    depthwise_conv2d_1_scratch0_array.data_start = AI_PTR(activations + 65664);
    input_1_output_array.data = AI_PTR(NULL);
    input_1_output_array.data_start = AI_PTR(NULL);
    conv2d_1_output_array.data = AI_PTR(activations + 0);
    conv2d_1_output_array.data_start = AI_PTR(activations + 0);
    depthwise_conv2d_1_output_array.data = AI_PTR(activations + 69120);
    depthwise_conv2d_1_output_array.data_start = AI_PTR(activations + 69120);
    separable_conv2d_1_output_array.data = AI_PTR(activations + 0);
    separable_conv2d_1_output_array.data_start = AI_PTR(activations + 0);
    separable_conv2d_1_conv2d_output_array.data = AI_PTR(activations + 2304);
    separable_conv2d_1_conv2d_output_array.data_start = AI_PTR(activations + 2304);
    dense_output_array.data = AI_PTR(activations + 0);
    dense_output_array.data_start = AI_PTR(activations + 0);
    softmax_output_array.data = AI_PTR(NULL);
    softmax_output_array.data_start = AI_PTR(NULL);
    
  }
  return true;
}



AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_buffer* weights_buffer)
{
  AI_ASSERT(net_ctx &&  weights_buffer && weights_buffer->data)

  ai_ptr weights = AI_PTR(weights_buffer->data);
  AI_ASSERT(weights)
  AI_UNUSED(net_ctx)

  {
    /* Updating weights (byte) offsets */
    
    dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_bias_array.data = AI_PTR(weights + 6336);
    dense_bias_array.data_start = AI_PTR(weights + 6336);
    dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_weights_array.data = AI_PTR(weights + 6080);
    dense_weights_array.data_start = AI_PTR(weights + 6080);
    separable_conv2d_1_conv2d_bias_array.format |= AI_FMT_FLAG_CONST;
    separable_conv2d_1_conv2d_bias_array.data = AI_PTR(weights + 6016);
    separable_conv2d_1_conv2d_bias_array.data_start = AI_PTR(weights + 6016);
    separable_conv2d_1_conv2d_weights_array.format |= AI_FMT_FLAG_CONST;
    separable_conv2d_1_conv2d_weights_array.data = AI_PTR(weights + 4992);
    separable_conv2d_1_conv2d_weights_array.data_start = AI_PTR(weights + 4992);
    separable_conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    separable_conv2d_1_bias_array.data = AI_PTR(weights + 4928);
    separable_conv2d_1_bias_array.data_start = AI_PTR(weights + 4928);
    separable_conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    separable_conv2d_1_weights_array.data = AI_PTR(weights + 3904);
    separable_conv2d_1_weights_array.data_start = AI_PTR(weights + 3904);
    depthwise_conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    depthwise_conv2d_1_bias_array.data = AI_PTR(weights + 3840);
    depthwise_conv2d_1_bias_array.data_start = AI_PTR(weights + 3840);
    depthwise_conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    depthwise_conv2d_1_weights_array.data = AI_PTR(weights + 1408);
    depthwise_conv2d_1_weights_array.data_start = AI_PTR(weights + 1408);
    conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_bias_array.data = AI_PTR(weights + 1376);
    conv2d_1_bias_array.data_start = AI_PTR(weights + 1376);
    conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_weights_array.data = AI_PTR(weights + 0);
    conv2d_1_weights_array.data_start = AI_PTR(weights + 0);
  }

  return true;
}


/**  PUBLIC APIs SECTION  *****************************************************/

AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if ( report && net_ctx )
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = {AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR,
                            AI_TOOLS_API_VERSION_MICRO, 0x0},

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 761862,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .activations       = AI_STRUCT_INIT,
      .params            = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if ( !ai_platform_api_get_network_report(network, &r) ) return false;

    *report = r;
    return true;
  }

  return false;
}

AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if ( !net_ctx ) return false;

  ai_bool ok = true;
  ok &= network_configure_weights(net_ctx, &params->params);
  ok &= network_configure_activations(net_ctx, &params->activations);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}

#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

