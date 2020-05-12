#include <stdio.h>
#include <math.h>
#include "conv2d.h"

__device__ float apply_filter(float** image, int x, int y, float** filter, int filter_size)
{
    float result = 0.f;

    for (int i = 0; i < filter_size; i++)
    {
        for (int j = 0; j < filter_size; j++)
        {
            result += image[y + i][x + j] * filter[i][j];
        }
    }

    return result;
}

__global__ void conv_2d(float **image_input, float **image_output, int image_width, int image_height, float **filter, int filter_size)
{
    // padding = 0, stride = 1
    int out_width = image_width - filter_size + 1;
    int out_height = image_height - filter_size + 1;

    float division = ((float)out_height / blockDim.x);
    int start_y = division * threadIdx.x;
    int end_y = division * (threadIdx.x + 1);

    for (int y = start_y; y < end_y; y++)
    {
        for (int x = 0; x < out_width; x++)
        {
            image_output[y][x] = apply_filter(image_input, x, y, filter, filter_size);
        }
    }
}

float** allocCudaMemory(int x, int y) 
{
    float** mem;
    cudaMallocManaged(&mem, y * sizeof(float*));

    for (int i = 0; i < y; i++) {
        cudaMallocManaged(&mem[i], x * sizeof(float));
    }

    return mem;
}

float** runx(int** image, int width, int height)
{
    // load image

    // Cuda goes here...
    int img_x = width, img_y = height;
    int filter_size = 3;
    int img_x_out = img_x - filter_size + 1;
    int img_y_out = img_y - filter_size + 1;

    float **image_input;
    float **image_output;
    float **filter;

    image_input = allocCudaMemory(img_x, img_y);
    image_output = allocCudaMemory(img_x_out, img_y_out);
    filter = allocCudaMemory(filter_size, filter_size);

    for (int i = 0; i < img_x; i++)
    {
        for (int j = 0; j < img_y; j++)
        {
            image_input[i][j] = (float)image[i][j] / 255.0;
        }
    }

    for (int i = 0; i < filter_size; i++)
    {
        for (int j = 0; j < filter_size; j++)
        {
            filter[i][j] = 1.0f;
        }
    }

    int blockSize = 2;
    int numBlocks = 1;

    conv_2d<<<numBlocks, blockSize>>>(image_input, image_output, img_x, img_y, filter, filter_size);
    fflush(stdout);

    cudaDeviceSynchronize();

    return image_output;
}