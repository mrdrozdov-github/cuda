#pragma once

#include <stdio.h>
#define CUDA(call) do {                                 \
        cudaError_t e = (call);                         \
        if (e == cudaSuccess) break;                    \
        fprintf(stderr, __FILE__":%d: %s (%d)\n",       \
                __LINE__, cudaGetErrorString(e), e);    \
        exit(1);                                        \
    } while (0)


inline unsigned divup(unsigned n, unsigned div)
{
    return (n + div - 1) / div;
}
