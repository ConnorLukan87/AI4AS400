// types.cpp
#include "types.h"

namespace nn {

ActivationFunc get_activation(ActivationType type) {
  switch (type) {
    case ActivationType::SIGMOID: return sigmoid;
    case ActivationType::RELU: return relu;
    case ActivationType::TANH: return tanh_activation;
    case ActivationType::LINEAR: return linear_activation;
  }
  throw std::invalid_argument("Unknown activation type");
}

DerivativeFunc get_derivative(ActivationType type) {
  switch (type) {
    case ActivationType::SIGMOID: return sigmoid_derivative;
    case ActivationType::RELU: return relu_derivative;
    case ActivationType::TANH: return tanh_derivative;
    case ActivationType::LINEAR: return linear_derivative;
  }
  throw std::invalid_argument("Unknown derivative type");
}

ActivationType get_activation_from_string(const std::string& str) {
  std::string lower = str;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  if (lower == "sigmoid") return ActivationType::SIGMOID;
  if (lower == "relu") return ActivationType::RELU;
  if (lower == "tanh") return ActivationType::TANH;
  if (lower == "linear") return ActivationType::LINEAR;
  throw std::invalid_argument("Unknown activation string");
}

} // namespace nn