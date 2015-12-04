#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/softlif_layer.hpp"

namespace caffe {

const float tau_ref = 0.3;//0.004;
const float tau_rc = 0.2;//0.02;
const float v_th = 1.0; //voltage threshold
const float y = 100.0; //smoothing parameter

template <typename Dtype>
inline Dtype softlif(Dtype x) {
  return 1. / (tau_ref + tau_rc * log(1. + v_th/(y*log(1. + exp((x-v_th)/y)))));
}

template <typename Dtype>
inline Dtype d_softlif(Dtype x) {
  return tau_rc * v_th * exp((x-v_th)/y) / (y*y*(exp((x-v_th)/y)+1.) * pow(log(exp((x-v_th)/y)+1.),2) * (v_th/(y*log(exp((x-v_th)/y)+1.))+1.) * pow((tau_rc*log(v_th/(y*log(exp((x-v_th)/y)+1.))+1.) + tau_ref),2));
}

template <typename Dtype>
void SoftLIFLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = softlif(bottom_data[i]);
  }
}

template <typename Dtype>
void SoftLIFLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    //const Dtype* top_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * d_softlif(top_data[i]); //TODO: is it top or bottom data??
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftLIFLayer);
#endif

INSTANTIATE_CLASS(SoftLIFLayer);


}  // namespace caffe
