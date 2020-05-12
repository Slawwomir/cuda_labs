#include <stdio.h>
#include <CImg.h>
#include "conv2d.h"

using namespace std;
using namespace cimg_library;

int** allocMemory(int x, int y) 
{
    int** mem = (int**) malloc(y * sizeof(int*));

    for (int i = 0; i < y; i++) {
        mem[i] = (int*) malloc(x * sizeof(int));
    }

    return mem;
}

int main()
{
    CImg<unsigned char> src("test.png");
    int width = src.width();
    int height = src.height();

    int** red_layer = allocMemory(width, height);

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            printf("red_layer[%d][%d] = %d", r, c, (int)src(c, r, 0, 0));
            // red_layer[r][c] = (int)src(c, r, 0, 0);
            printf(" ok...\n");
        }
    }

    printf("Running cuda...");
    float** image = runx(red_layer, width, height);

    for (int r = 0; r < height - 2; r++) {
        for (int c = 0; c < width - 2; c++) {
            printf("image[%d][%d] == %.2f \n", r, c, image[r][c]);
        }
    }

    return 0;
}