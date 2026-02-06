#ifndef TYPES_H
#define TYPES_H

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nn {

static bool random_seeded = false;
inline float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}
inline float sigmoid_derivative(float x) {
  float s = sigmoid(x);
  return s * (1.0f - s);
}
inline float relu(float x) {
  return std::max(0.0f, x);
}
inline float relu_derivative(float x) {
  return x > 0.0f ? 1.0f : 0.0f;
}
inline float tanh_activation(float x) {
  return std::tanh(x);
}
inline float tanh_derivative(float x) {
  float t = tanh_activation(x);
  return 1.0f - t * t;
}
inline float linear_activation(float x) {
  return x;
}
inline float linear_derivative(float /*x*/) {
  return 1.0f;
}
enum class ActivationType {
  SIGMOID, RELU, TANH, LINEAR
};
enum class LossTypeEnum {
  MSE,
  CROSS_ENTROPY_BINARY,
  CROSS_ENTROPY_MULTI
};
class LossType {
 public:
  enum class Type {
    MSE = LossTypeEnum::MSE,
    CROSS_ENTROPY_BINARY =
      LossTypeEnum::CROSS_ENTROPY_BINARY,
    CROSS_ENTROPY_MULTI =
      LossTypeEnum::CROSS_ENTROPY_MULTI
  };
 private:
  Type type_;
 public:
  LossType() : type_(Type::MSE) {}
  explicit LossType(Type t) : type_(t) {}
  LossType(const LossType& o) : type_(o.type_) {}
  LossType& operator=(const LossType& o) {
    type_ = o.type_;
    return *this;
  }
  LossType& operator=(Type t) {
    type_ = t;
    return *this;
  }
  bool operator==(Type t) const {
    return type_ == t;
  }
  bool operator!=(Type t) const {
    return type_ != t;
  }
  Type get() const { return type_; }
  int toInt() const {
    return static_cast<int>(type_);
  }
  static LossType fromInt(int v) {
    return LossType(static_cast<Type>(v));
  }
};
using Tensor1D = std::vector<float>;
using Tensor2D = std::vector<Tensor1D>;
using Tensor3D = std::vector<Tensor2D>;
using Tensor4D = std::vector<Tensor3D>;
using ActivationFunc = float (*)(float);
using DerivativeFunc = float (*)(float);
using LossFunc = float (*)(const Tensor1D&,
                           const Tensor1D&);
using LossDerivFunc = Tensor1D (*)(const Tensor1D&,
                                    const Tensor1D&);
ActivationFunc get_activation(ActivationType type);
DerivativeFunc get_derivative(ActivationType type);
ActivationType get_activation_from_string(
  const std::string& str);
enum class LayerType {
  DENSE = 0,
  CONV = 1,
  FLATTEN = 2,
  RNN = 3,
  LSTM = 4
};
} // namespace nn
#endif // TYPES_H
