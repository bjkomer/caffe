#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/softlif_layer.hpp"

namespace caffe {

__global__ const float tau_ref = 0.004;
__global__ const float tau_rc = 0.02;
__global__ const float v_th = 1.0; //voltage threshold
__global__ const float y = 0.5; //smoothing parameter

template <typename Dtype>
__global__ void SoftLIFForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = 1. / (tau_ref + tau_rc * log(1. + v_th/(y*log(1. + exp((in[index]-v_th)/y)))));
  }
}

template <typename Dtype>
void SoftLIFLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftLIFForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void SoftLIFBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype x = out_data[index] - v_th;
    out_diff[index] = in_diff[index] * (tau_rc * v_th * exp(x/y) / (y*y*(exp(x/y)+1.) * log(exp(x/y)+1.) * (v_th/(y*log(exp(x/y)+1.))+1.) * pow((tau_rc*log(v_th/(y*log(exp(x/y)+1.))+1.) + tau_ref),2)));
  }
}

template <typename Dtype>
void SoftLIFLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftLIFBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftLIFLayer);


}  // namespace caffe
