#include <stdio.h>
#include <math.h>

__device__ float **initialize_2d(int height, int width)
{
    float **array = (float **)malloc(height * sizeof(float *));

    for (int i = 0; i < height; i++)
    {
        array[i] = (float *)malloc(height * sizeof(float));
    }

    return array;
}

__device__ void free_2d(float **array, int height, int width)
{
    for (int i = 0; i < height; i++)
    {
        free(array[i]);
    }

    free(array);
}

__global__ void conv_2d(float **image, int image_width, int image_height, float **filter, int filter_size)
{
    // padding = 0, stride = 1
    int out_width = image_width - filter_size + 1;
    int out_height = image_height - filter_size + 1;

    float **out_image = initialize_2d(out_height, out_width);

    for (int i = 0; i < out_height; i++)
    {
        for (int j = 0; j < out_width; j++)
        {
            float result = 0.f;

            for (int i_f = 0; i_f < filter_size; i_f++)
            {
                for (int j_f = 0; j_f < filter_size; j_f++)
                {
                    result += image[i][j] * filter[i_f][j_f];
                }
            }

            out_image[i][j] = result;
        }
    }

    for (int i = 0; i < out_width; i++)
    {
        for (int j = 0; j < out_height; j++)
        {
            printf("%.2f ", out_image[i][j]);
        }

        printf("\n");
    }

    free_2d(out_image, out_height, out_width);
}

int main()
{
    int img_x = 5, img_y = 5;
    float **image;

    int filter_size = 3;
    float **filter;

    cudaMallocManaged(&image, img_y * sizeof(float *));
    cudaMallocManaged(&filter, filter_size * sizeof(float *));

    // initialize
    for (int i = 0; i < img_x; i++)
    {
        cudaMallocManaged(&image[i], img_x * sizeof(float));
        for (int j = 0; j < img_y; j++)
        {
            image[i][j] = 1.0f;
        }
    }

    for (int i = 0; i < filter_size; i++)
    {
        cudaMallocManaged(&filter[i], filter_size * sizeof(float));
        for (int j = 0; j < filter_size; j++)
        {
            filter[i][j] = 2.0f;
        }
    }

    int blockSize = 1;
    int numBlocks = 1;

    conv_2d<<<numBlocks, blockSize>>>(image, img_x, img_y, filter, filter_size);
    fflush(stdout);

    cudaDeviceSynchronize();

    cudaFree(image);
    cudaFree(filter);

    return 0;
}