#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int kernelCenter = (l.size - 1) / 2;

    for (int i = 0; i < out.rows; i++) {
        for (int j = 0; j < out.cols; j++) {
            int channel = j/(outw*outh);
            // extract the "row and col" we are outputting to
            int outRow = (j%(outw*outh))/outw;
            int outCol = (j%(outw*outh))%outw;
            // and the top-left corner of what we are scanning
            int miniRow = l.stride*outRow - kernelCenter;
            int miniCol = l.stride*outCol - kernelCenter;
            int capRow = miniRow + l.size;
            int capCol = miniCol + l.size;
            if (miniRow < 0) {
                miniRow = 0;
            }
            if (miniCol < 0) {
                miniCol = 0;
            }
            if (capRow > l.height) {
                capRow = l.height;
            }
            if (capCol > l.width) {
                capCol = l.width;
            }
            // get the maximum value
            float maxValue = in.data[i*in.cols 
                                     + channel*l.width*l.height 
                                     + miniRow*l.width 
                                     + miniCol];
            for (int scanRow = miniRow; scanRow < capRow; scanRow++) {
                for (int scanCol = miniCol; scanCol < capCol; scanCol++) {
                    float scanValue = in.data[i*in.cols 
                                              + channel*l.width*l.height 
                                              + scanRow*l.width 
                                              + scanCol];
                    if (scanValue > maxValue) {
                        maxValue = scanValue;
                    }
                }
            }
            out.data[i*out.rows + j] = maxValue;
        }
    }

    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.



    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

