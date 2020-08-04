/**
THC is one of the low-level tensor libraries for PyTorch
TH = TorcH
THC = TorcH Cuda
THCS = TorcH Cuda Sparse (now defunct)
THCUNN = TorcH CUda Neural Network (see cunn)
THNN = TorcH Neural Network (now defunct)
THS = TorcH Sparse (now defunct)
*/
#include <THC/THC.h>
#include <stdio.h>
#include "nms_cuda_kernel.h"

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

int nms_cuda(THCudaIntTensor *keep_out, THCudaTensor *boxes_host,
		     THCudaIntTensor *num_out, float nms_overlap_thresh) {

	nms_cuda_compute(THCudaIntTensor_data(state, keep_out), 
		         THCudaIntTensor_data(state, num_out), 
      	                 THCudaTensor_data(state, boxes_host), 
		         THCudaTensor_size(state, boxes_host, 0),
		         THCudaTensor_size(state, boxes_host, 1),
		         nms_overlap_thresh);

	return 1;
}
