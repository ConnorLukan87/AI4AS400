// layer.cpp
#include "layer.h"

namespace nn {

Tensor2D matmul(const Tensor2D& A, const Tensor2D& B) {
  int m = A.size();
  int k = A[0].size();
  int n = B[0].size();
  if (k != B.size()) throw std::invalid_argument("Matrix dimensions mismatch");
  Tensor2D C(m, Tensor1D(n, 0.0f));
#ifdef USE_IBM_XL
  // Use IBM XL MMA built-in functions for acceleration
  // Example placeholder - implement tiling with __mma_xvf32gerpp etc.
  // Require Power10 and compiler support. For general sizes, tile into 4x4 blocks.
  // Note: This requires including appropriate headers like <vecintrin.h> or similar.
  // For simplicity, fall back to standard if not fully implemented.
#else
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int p = 0; p < k; ++p) {
        C[i][j] += A[i][p] * B[p][j];
      }
    }
  }
#endif
  return C;
}

Tensor1D matvec(const Tensor2D& A, const Tensor1D& x) {
  int m = A.size();
  int k = A[0].size();
  if (k != x.size()) throw std::invalid_argument("Dimensions mismatch");
  Tensor1D y(m, 0.0f);
#ifdef USE_IBM_XL
  // Similar placeholder for IBM XL MMA acceleration
#else
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < k; ++j) {
      y[i] += A[i][j] * x[j];
    }
  }
#endif
  return y;
}

Layer::Layer(ActivationType act_type) : activation_type(act_type), act(get_activation(act_type)), deriv(get_derivative(act_type)) {}

Tensor1D Layer::forward1D(const Tensor1D& input) { return {}; }

Tensor2D Layer::forward_seq(const Tensor2D& input) { return {}; }

Tensor3D Layer::forward3D(const Tensor3D& input) { return {}; }

Tensor1D Layer::flatten_forward(const Tensor3D& input) { return {}; }

void Layer::compute_deltas1D(const Tensor1D& grad_output, const Tensor1D& z_val) {}

void Layer::compute_deltas_seq(const Tensor2D& grad_output, const Tensor2D& z_val) {}

void Layer::compute_deltas3D(const Tensor3D& grad_output, const Tensor3D& z_val) {}

Tensor1D Layer::get_grad_input1D() { return {}; }

Tensor2D Layer::get_grad_input_seq() { return {}; }

Tensor3D Layer::get_grad_input3D() { return {}; }

Tensor1D Layer::get_outputs1D() const { return {}; }

Tensor2D Layer::get_outputs_seq() const { return {}; }

Tensor3D Layer::get_outputs3D() const { return {}; }

Tensor1D Layer::get_z1D() const { return {}; }

Tensor2D Layer::get_z_seq() const { return {}; }

Tensor3D Layer::get_z3D() const { return {}; }

Tensor1D Layer::get_deltas1D() const { return {}; }

Tensor2D Layer::get_deltas_seq() const { return {}; }

Tensor3D Layer::get_deltas3D() const { return {}; }

void Layer::update_params(const Tensor1D& prev_outputs, float lr, int t, float beta1, float beta2, float epsilon) {}

void Layer::update_params_seq(const Tensor2D& prev_outputs, float lr, int t, float beta1, float beta2, float epsilon) {}

void Layer::update_params3D(const Tensor3D& prev_outputs, float lr, int t, float beta1, float beta2, float epsilon) {}

DenseLayer::DenseLayer(int in, int out, ActivationType act_type) : Layer(act_type), input_size(in), output_size(out), weights(out, Tensor1D(in, 0.0f)), biases(out, 0.0f), outputs(out, 0.0f), deltas(out, 0.0f), z(out, 0.0f), m_weights(out, Tensor1D(in, 0.0f)), v_weights(out, Tensor1D(in, 0.0f)), m_biases(out, 0.0f), v_biases(out, 0.0f) {

  if (!random_seeded) {

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    random_seeded = true;

  }

  float scale = std::sqrt(2.0f / in);

  for (auto& row : weights) {

    for (auto& w : row) {

      w = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;

    }

  }

}

Tensor1D DenseLayer::forward1D(const Tensor1D& input) {

  z = matvec(weights, input);

  for (size_t i = 0; i < z.size(); ++i) {

    z[i] += biases[i];

  }

  outputs = z;

  for (auto& o : outputs) {

    o = act(o);

  }

  return outputs;

}

void DenseLayer::compute_deltas1D(const Tensor1D& grad_output, const Tensor1D& z_val) {

  deltas.resize(output_size);

  for (int i = 0; i < output_size; ++i) {

    deltas[i] = grad_output[i] * deriv(z_val[i]);

  }

}

Tensor1D DenseLayer::get_grad_input1D() {

  Tensor1D grad_input(input_size, 0.0f);

  for (int i = 0; i < input_size; ++i) {

    for (int j = 0; j < output_size; ++j) {

      grad_input[i] += deltas[j] * weights[j][i];

    }

  }

  return grad_input;

}

void DenseLayer::update_params(const Tensor1D& prev_outputs, float lr, int t, float beta1, float beta2, float epsilon) {

  Tensor2D grad_weights(output_size, Tensor1D(input_size, 0.0f));

  Tensor1D grad_biases(output_size, 0.0f);

  for (int j = 0; j < output_size; ++j) {

    for (int i = 0; i < input_size; ++i) {

      grad_weights[j][i] = deltas[j] * prev_outputs[i];

    }

    grad_biases[j] = deltas[j];

  }

  for (int j = 0; j < output_size; ++j) {

    for (int i = 0; i < input_size; ++i) {

      m_weights[j][i] = beta1 * m_weights[j][i] + (1 - beta1) * grad_weights[j][i];

      v_weights[j][i] = beta2 * v_weights[j][i] + (1 - beta2) * grad_weights[j][i] * grad_weights[j][i];

      float m_hat = m_weights[j][i] / (1 - std::pow(beta1, t));

      float v_hat = v_weights[j][i] / (1 - std::pow(beta2, t));

      weights[j][i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);

    }

    m_biases[j] = beta1 * m_biases[j] + (1 - beta1) * grad_biases[j];

    v_biases[j] = beta2 * v_biases[j] + (1 - beta2) * grad_biases[j] * grad_biases[j];

    float m_hat = m_biases[j] / (1 - std::pow(beta1, t));

    float v_hat = v_biases[j] / (1 - std::pow(beta2, t));

    biases[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);

  }

}

Tensor1D DenseLayer::get_outputs1D() const { return outputs; }

Tensor1D DenseLayer::get_z1D() const { return z; }

Tensor1D DenseLayer::get_deltas1D() const { return deltas; }

void DenseLayer::save(std::ostream& os) const {

  os << input_size << " " << output_size << " " << static_cast<int>(activation_type) << "\n";

  for (const auto& row : weights) {

    for (float w : row) os << w << " ";

    os << "\n";

  }

  for (float b : biases) os << b << " ";

  os << "\n";

}

void DenseLayer::load(std::istream& is) {

  int act_int;

  is >> input_size >> output_size >> act_int;

  activation_type = static_cast<ActivationType>(act_int);

  act = get_activation(activation_type);

  deriv = get_derivative(activation_type);

  weights.resize(output_size, Tensor1D(input_size));

  for (auto& row : weights) {

    for (auto& w : row) is >> w;

  }

  biases.resize(output_size);

  for (auto& b : biases) is >> b;

  outputs.resize(output_size);

  deltas.resize(output_size);

  z.resize(output_size);

  m_weights.resize(output_size, Tensor1D(input_size, 0.0f));

  v_weights.resize(output_size, Tensor1D(input_size, 0.0f));

  m_biases.resize(output_size, 0.0f);

  v_biases.resize(output_size, 0.0f);

}

LayerType DenseLayer::get_layer_type() const { return LayerType::DENSE; }

ConvLayer::ConvLayer(int in_ch, int out_ch, int k_size, int str, ActivationType act_type) : Layer(act_type), in_channels(in_ch), out_channels(out_ch), kernel_size(k_size), stride(str), weights(out_ch, Tensor3D(in_ch, Tensor2D(k_size, Tensor1D(k_size, 0.0f)))), biases(out_ch, 0.0f), m_weights(out_ch, Tensor3D(in_ch, Tensor2D(k_size, Tensor1D(k_size, 0.0f)))), v_weights(out_ch, Tensor3D(in_ch, Tensor2D(k_size, Tensor1D(k_size, 0.0f)))), m_biases(out_ch, 0.0f), v_biases(out_ch, 0.0f) {

  if (!random_seeded) {

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    random_seeded = true;

  }

  float scale = std::sqrt(2.0f / (in_ch * k_size * k_size));

  for (auto& out : weights) {

    for (auto& in : out) {

      for (auto& row : in) {

        for (auto& w : row) {

          w = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;

        }

      }

    }

  }

}

Tensor3D ConvLayer::forward3D(const Tensor3D& input) {

  int in_h = input[0].size();

  int in_w = input[0][0].size();

  int out_h = (in_h - kernel_size) / stride + 1;

  int out_w = (in_w - kernel_size) / stride + 1;

  Tensor3D out(out_channels, Tensor2D(out_h, Tensor1D(out_w, 0.0f)));

  z = out;

  for (int oc = 0; oc < out_channels; ++oc) {

    for (int oh = 0; oh < out_h; ++oh) {

      for (int ow = 0; ow < out_w; ++ow) {

        float sum = biases[oc];

        for (int ic = 0; ic < in_channels; ++ic) {

          for (int kh = 0; kh < kernel_size; ++kh) {

            for (int kw = 0; kw < kernel_size; ++kw) {

              int ih = oh * stride + kh;

              int iw = ow * stride + kw;

              if (ih < in_h && iw < in_w) {

                sum += input[ic][ih][iw] * weights[oc][ic][kh][kw];

              }

            }

          }

        }

        z[oc][oh][ow] = sum;

        out[oc][oh][ow] = act(sum);

      }

    }

  }

  outputs = out;

  return out;

}

void ConvLayer::compute_deltas3D(const Tensor3D& grad_output, const Tensor3D& z_val) {

  deltas = grad_output;

  for (int oc = 0; oc < out_channels; ++oc) {

    for (int oh = 0; oh < z_val[0].size(); ++oh) {

      for (int ow = 0; ow < z_val[0][0].size(); ++ow) {

        deltas[oc][oh][ow] *= deriv(z_val[oc][oh][ow]);

      }

    }

  }

}

Tensor3D ConvLayer::get_grad_input3D() {

  int out_h = deltas[0].size();

  int out_w = deltas[0][0].size();

  int in_h = (out_h - 1) * stride + kernel_size;

  int in_w = (out_w - 1) * stride + kernel_size;

  Tensor3D grad_input(in_channels, Tensor2D(in_h, Tensor1D(in_w, 0.0f));

  for (int ic = 0; ic < in_channels; ++ic) {

    for (int ih = 0; ih < in_h; ++ih) {

      for (int iw = 0; iw < in_w; ++iw) {

        float sum = 0.0f;

        for (int oc = 0; oc < out_channels; ++oc) {

          for (int kh = 0; kh < kernel_size; ++kh) {

            for (int kw = 0; kw < kernel_size; ++kw) {

              int oh = (ih - kh) / stride;

              int ow = (iw - kw) / stride;

              if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w && (ih - kh) % stride == 0 && (iw - kw) % stride == 0) {

                sum += deltas[oc][oh][ow] * weights[oc][ic][kh][kw];

              }

            }

          }

        }

        grad_input[ic][ih][iw] = sum;

      }

    }

  }

  return grad_input;

}

void ConvLayer::update_params3D(const Tensor3D& prev_input, float lr, int t, float beta1, float beta2, float epsilon) {

  int out_h = deltas[0].size();

  int out_w = deltas[0][0].size();

  Tensor4D grad_weights(out_channels, Tensor3D(in_channels, Tensor2D(kernel_size, Tensor1D(kernel_size, 0.0f))));

  Tensor1D grad_biases(out_channels, 0.0f);

  for (int oc = 0; oc < out_channels; ++oc) {

    for (int oh = 0; oh < out_h; ++oh) {

      for (int ow = 0; ow < out_w; ++ow) {

        float d = deltas[oc][oh][ow];

        grad_biases[oc] += d;

        for (int ic = 0; ic < in_channels; ++ic) {

          for (int kh = 0; kh < kernel_size; ++kh) {

            for (int kw = 0; kw < kernel_size; ++kw) {

              int ih = oh * stride + kh;

              int iw = ow * stride + kw;

              if (ih < prev_input[0].size() && iw < prev_input[0][0].size()) {

                grad_weights[oc][ic][kh][kw] += d * prev_input[ic][ih][iw];

              }

            }

          }

        }

      }

    }

  }

  for (int oc = 0; oc < out_channels; ++oc) {

    for (int ic = 0; ic < in_channels; ++ic) {

      for (int kh = 0; kh < kernel_size; ++kh) {

        for (int kw = 0; kw < kernel_size; ++kw) {

          float g = grad_weights[oc][ic][kh][kw];

          m_weights[oc][ic][kh][kw] = beta1 * m_weights[oc][ic][kh][kw] + (1 - beta1) * g;

          v_weights[oc][ic][kh][kw] = beta2 * v_weights[oc][ic][kh][kw] + (1 - beta2) * g * g;

          float m_hat = m_weights[oc][ic][kh][kw] / (1 - std::pow(beta1, t));

          float v_hat = v_weights[oc][ic][kh][kw] / (1 - std::pow(beta2, t));

          weights[oc][ic][kh][kw] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);

        }

      }

    }

    float g = grad_biases[oc];

    m_biases[oc] = beta1 * m_biases[oc] + (1 - beta1) * g;

    v_biases[oc] = beta2 * v_biases[oc] + (1 - beta2) * g * g;

    float m_hat = m_biases[oc] / (1 - std::pow(beta1, t));

    float v_hat = v_biases[oc] / (1 - std::pow(beta2, t));

    biases[oc] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);

  }

}

Tensor3D ConvLayer::get_outputs3D() const { return outputs; }

Tensor3D ConvLayer::get_z3D() const { return z; }

Tensor3D ConvLayer::get_deltas3D() const { return deltas; }

void ConvLayer::save(std::ostream& os) const {

  os << in_channels << " " << out_channels << " " << kernel_size << " " << stride << " " << static_cast<int>(activation_type) << "\n";

  for (const auto& oc : weights) {

    for (const auto& ic : oc) {

      for (const auto& row : ic) {

        for (float w : row) os << w << " ";

        os << "\n";

      }

    }

  }

  for (float b : biases) os << b << " ";

  os << "\n";

}

void ConvLayer::load(std::istream& is) {

  int act_int;

  is >> in_channels >> out_channels >> kernel_size >> stride >> act_int;

  activation_type = static_cast<ActivationType>(act_int);

  act = get_activation(activation_type);

  deriv = get_derivative(activation_type);

  weights.resize(out_channels, Tensor3D(in_channels, Tensor2D(kernel_size, Tensor1D(kernel_size))));

  for (auto& oc : weights) {

    for (auto& ic : oc) {

      for (auto& row : ic) {

        for (auto& w : row) is >> w;

      }

    }

  }

  biases.resize(out_channels);

  for (auto& b : biases) is >> b;

  m_weights.resize(out_channels, Tensor3D(in_channels, Tensor2D(kernel_size, Tensor1D(kernel_size, 0.0f))));

  v_weights.resize(out_channels, Tensor3D(in_channels, Tensor2D(kernel_size, Tensor1D(kernel_size, 0.0f))));

  m_biases.resize(out_channels, 0.0f);

  v_biases.resize(out_channels, 0.0f);

}

LayerType ConvLayer::get_layer_type() const { return LayerType::CONV; }

FlattenLayer::FlattenLayer() : Layer(ActivationType::LINEAR), input_channels(0), input_h(0), input_w(0) {}

Tensor1D FlattenLayer::flatten_forward(const Tensor3D& input) {

  input_channels = input.size();

  input_h = input[0].size();

  input_w = input[0][0].size();

  int flat_size = input_channels * input_h * input_w;

  Tensor1D flat(flat_size);

  int idx = 0;

  for (const auto& ch : input) {

    for (const auto& row : ch) {

      for (float val : row) {

        flat[idx++] = val;

      }

    }

  }

  outputs = flat;

  deltas.resize(flat_size, 0.0f);

  return flat;

}

void FlattenLayer::compute_deltas1D(const Tensor1D& grad_output, const Tensor1D& z_val) {

  deltas = grad_output;

}

Tensor3D FlattenLayer::get_grad_input3D() {

  Tensor3D grad_input(input_channels, Tensor2D(input_h, Tensor1D(input_w, 0.0f)));

  int idx = 0;

  for (int c = 0; c < input_channels; ++c) {

    for (int h = 0; h < input_h; ++h) {

      for (int w = 0; w < input_w; ++w) {

        grad_input[c][h][w] = deltas[idx++];

      }

    }

  }

  return grad_input;

}

Tensor1D FlattenLayer::get_outputs1D() const { return outputs; }

Tensor1D FlattenLayer::get_deltas1D() const { return deltas; }

void FlattenLayer::save(std::ostream& os) const {

  os << static_cast<int>(activation_type) << "\n";

}

void FlattenLayer::load(std::istream& is) {

  int act_int;

  is >> act_int;

  activation_type = static_cast<ActivationType>(act_int);

  act = get_activation(activation_type);

  deriv = get_derivative(activation_type);

}

LayerType FlattenLayer::get_layer_type() const { return LayerType::FLATTEN; }

RNNLayer::RNNLayer(int in, int hid, ActivationType act_type) : Layer(act_type), input_size(in), hidden_size(hid), Wxh(hid, Tensor1D(in, 0.0f)), Whh(hid, Tensor1D(hid, 0.0f)), bh(hid, 0.0f), m_Wxh(hid, Tensor1D(in, 0.0f)), v_Wxh(hid, Tensor1D(in, 0.0f)), m_Whh(hid, Tensor1D(hid, 0.0f)), v_Whh(hid, Tensor1D(hid, 0.0f)), m_bh(hid, 0.0f), v_bh(hid, 0.0f) {

  if (!random_seeded) {

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    random_seeded = true;

  }

  float scale = std::sqrt(1.0f / hid);

  for (auto& row : Wxh) {

    for (auto& w : row) {

      w = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;

    }

  }

  for (auto& row : Whh) {

    for (auto& w : row) {

      w = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;

    }

  }

}

Tensor2D RNNLayer::forward_seq(const Tensor2D& inputs) {

  int seq_len = inputs.size();

  Tensor2D h(seq_len + 1, Tensor1D(hidden_size, 0.0f));

  z.resize(seq_len, Tensor1D(hidden_size));

  outputs.resize(seq_len, Tensor1D(hidden_size));

  for (int t = 0; t < seq_len; ++t) {

    Tensor1D wx = matvec(Wxh, inputs[t]);

    Tensor1D wh = matvec(Whh, h[t]);

    for (int i = 0; i < hidden_size; ++i) {

      z[t][i] = wx[i] + wh[i] + bh[i];

    }

    h[t+1] = z[t];

    for (auto& val : h[t+1]) {

      val = act(val);

    }

    outputs[t] = h[t+1];

  }

  return outputs;

}

void RNNLayer::compute_deltas_seq(const Tensor2D& grad_output, const Tensor2D& z_val) {

  int seq_len = grad_output.size();

  deltas.resize(seq_len, Tensor1D(hidden_size, 0.0f));

  Tensor1D grad_h(hidden_size, 0.0f);

  for (int t = seq_len - 1; t >= 0; --t) {

    for (int i = 0; i < hidden_size; ++i) {

      grad_h[i] += grad_output[t][i];

    }

    for (int i = 0; i < hidden_size; ++i) {

      deltas[t][i] = grad_h[i] * deriv(z_val[t][i]);

    }

    Tensor1D grad_h_prev(hidden_size, 0.0f);

  for (int i = 0; i < hidden_size; ++i) {

      for (int j = 0; j < hidden_size; ++j) {

        grad_h_prev[i] += deltas[t][j] * Whh[j][i];

      }

    }

    grad_h = grad_h_prev;

  }

}

Tensor2D RNNLayer::get_grad_input_seq() {

  int seq_len = deltas.size();

  Tensor2D grad_input(seq_len, Tensor1D(input_size, 0.0f));

  for (int t = 0; t < seq_len; ++t) {

    for (int i = 0; i < input_size; ++i) {

      for (int j = 0; j < hidden_size; ++j) {

        grad_input[t][i] += deltas[t][j] * Wxh[j][i];

      }

    }

  }

  return grad_input;

}

void RNNLayer::update_params_seq(const Tensor2D& inputs, float lr, int t, float beta1, float beta2, float epsilon) {

  int seq_len = inputs.size();

  Tensor2D grad_Wxh(hidden_size, Tensor1D(input_size, 0.0f));

  Tensor2D grad_Whh(hidden_size, Tensor1D(hidden_size, 0.0f));

  Tensor1D grad_bh(hidden_size, 0.0f);

  for (int s = 0; s < seq_len; ++s) {

    Tensor1D h_prev = (s == 0) ? Tensor1D(hidden_size, 0.0f) : outputs[s-1];

    for (int j = 0; j < hidden_size; ++j) {

      for (int i = 0; i < input_size; ++i) {

        grad_Wxh[j][i] += deltas[s][j] * inputs[s][i];

      }

      for (int i = 0; i < hidden_size; ++i) {

        grad_Whh[j][i] += deltas[s][j] * h_prev[i];

      }

      grad_bh[j] += deltas[s][j];

    }

  }

  for (int j = 0; j < hidden_size; ++j) {

    for (int i = 0; i < input_size; ++i) {

      float g = grad_Wxh[j][i];

      m_Wxh[j][i] = beta1 * m_Wxh[j][i] + (1 - beta1) * g;

      v_Wxh[j][i] = beta2 * v_Wxh[j][i] + (1 - beta2) * g * g;

      float m_hat = m_Wxh[j][i] / (1 - std::pow(beta1, t));

      float v_hat = v_Wxh[j][i] / (1 - std::pow(beta2, t));

      Wxh[j][i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);

    }

    for (int i = 0; i < hidden_size; ++i) {

      float g = grad_Whh[j][i];

      m_Whh[j][i] = beta1 * m_Whh[j][i] + (1 - beta1) * g;

      v_Whh[j][i] = beta2 * v_Whh[j][i] + (1 - beta2) * g * g;

      float m_hat = m_Whh[j][i] / (1 - std::pow(beta1, t));

      float v_hat = v_Whh[j][i] / (1 - std::pow(beta2, t));

      Whh[j][i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);

    }

    float g = grad_bh[j];

    m_bh[j] = beta1 * m_bh[j] + (1 - beta1) * g;

    v_bh[j] = beta2 * v_bh[j] + (1 - beta2) * g * g;

    float m_hat = m_bh[j] / (1 - std::pow(beta1, t));

    float v_hat = v_bh[j] / (1 - std::pow(beta2, t));

    bh[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);

  }

}

Tensor2D RNNLayer::get_outputs_seq() const { return outputs; }

Tensor2D RNNLayer::get_z_seq() const { return z; }

Tensor2D RNNLayer::get_deltas_seq() const { return deltas; }

void RNNLayer::save(std::ostream& os) const {

  os << input_size << " " << hidden_size << " " << static_cast<int>(activation_type) << "\n";

  for (const auto& row : Wxh) {

    for (float w : row) os << w << " ";

    os << "\n";

  }

  for (const auto& row : Whh) {

    for (float w : row) os << w << " ";

    os << "\n";

  }

  for (float b : bh) os << b << " ";

  os << "\n";

}

void RNNLayer::load(std::istream& is) {

  int act_int;

  is >> input_size >> hidden_size >> act_int;

  activation_type = static_cast<ActivationType>(act_int);

  act = get_activation(activation_type);

  deriv = get_derivative(activation_type);

  Wxh.resize(hidden_size, Tensor1D(input_size));

  for (auto& row : Wxh) {

    for (auto& w : row) is >> w;

  }

  Whh.resize(hidden_size, Tensor1D(hidden_size));

  for (auto& row : Whh) {

    for (auto& w : row) is >> w;

  }

  bh.resize(hidden_size);

  for (auto& b : bh) is >> b;

  m_Wxh.resize(hidden_size, Tensor1D(input_size, 0.0f));

  v_Wxh.resize(hidden_size, Tensor1D(input_size, 0.0f));

  m_Whh.resize(hidden_size, Tensor1D(hidden_size, 0.0f));

  v_Whh.resize(hidden_size, Tensor1D(hidden_size, 0.0f));

  m_bh.resize(hidden_size, 0.0f);

  v_bh.resize(hidden_size, 0.0f);

}

LayerType RNNLayer::get_layer_type() const { return LayerType::RNN; }

LSTMLayer::LSTMLayer(int in, int hid, ActivationType act_type) : Layer(act_type), input_size(in), hidden_size(hid), Wf(hidden_size, Tensor1D(input_size, 0.0f)), Wi(hidden_size, Tensor1D(input_size, 0.0f)), Wc(hidden_size, Tensor1D(input_size, 0.0f)), Wo(hidden_size, Tensor1D(input_size, 0.0f)), Uf(hidden_size, Tensor1D(hidden_size, 0.0f)), Ui(hidden_size, Tensor1D(hidden_size, 0.0f)), Uc(hidden_size, Tensor1D(hidden_size, 0.0f)), Uo(hidden_size, Tensor1D(hidden_size, 0.0f)), bf(hidden_size, 1.0f), bi(hidden_size, 0.0f), bc(hidden_size, 0.0f), bo(hidden_size, 0.0f), m_Wf(hidden_size, Tensor1D(input_size, 0.0f)), v_Wf(hidden_size, Tensor1D(input_size, 0.0f)), m_Uf(hidden_size, Tensor1D(hidden_size, 0.0f)), v_Uf(hidden_size, Tensor1D(hidden_size, 0.0f)), m_Wi(hidden_size, Tensor1D(input_size, 0.0f)), v_Wi(hidden_size, Tensor1D(input_size, 0.0f)), m_Ui(hidden_size, Tensor1D(hidden_size, 0.0f)), v_Ui(hidden_size, Tensor1D(hidden_size, 0.0f)), m_Wc(hidden_size, Tensor1D(input_size, 0.0f)), v_Wc(hidden_size, Tensor1D(input_size, 0.0f)), m_Uc(hidden_size, Tensor1D(hidden_size, 0.0f)), v_Uc(hidden_size, Tensor1D(hidden_size, 0.0f)), m_Wo(hidden_size, Tensor1D(input_size, 0.0f)), v_Wo(hidden_size, Tensor1D(input_size, 0.0f)), m_Uo(hidden_size, Tensor1D(hidden_size, 0.0f)), v_Uo(hidden_size, Tensor1D(hidden_size, 0.0f)), m_bf(hidden_size, 0.0f), v_bf(hidden_size, 0.0f), m_bi(hidden_size, 0.0f), v_bi(hidden_size, 0.0f), m_bc(hidden_size, 0.0f), v_bc(hidden_size, 0.0f), m_bo(hidden_size, 0.0f), v_bo(hidden_size, 0.0f) {

  float scale = std::sqrt(1.0f / hidden_size);

  auto init = [scale](Tensor2D& W) {

    for (auto& row : W) {

      for (auto& w : row) {

        w = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;

      }

    }

  };

  init(Wf);

  init(Wi);

  init(Wc);

  init(Wo);

  init(Uf);

  init(Ui);

  init(Uc);

  init(Uo);

}

Tensor2D LSTMLayer::forward_seq(const Tensor2D& inputs) {

  int seq_len = inputs.size();

  outputs.resize(seq_len, Tensor1D(hidden_size, 0.0f));

  cells.resize(seq_len + 1, Tensor1D(hidden_size, 0.0f));

  z_f.resize(seq_len, Tensor1D(hidden_size));

  z_i.resize(seq_len, Tensor1D(hidden_size));

  z_c.resize(seq_len, Tensor1D(hidden_size));

  z_o.resize(seq_len, Tensor1D(hidden_size));

  Tensor1D h_prev(hidden_size, 0.0f);

  Tensor1D c_prev(hidden_size, 0.0f);

  for (int t = 0; t < seq_len; ++t) {

    z_f[t] = matvec(Wf, inputs[t]) + matvec(Uf, h_prev) + bf;

    z_i[t] = matvec(Wi, inputs[t]) + matvec(Ui, h_prev) + bi;

    z_c[t] = matvec(Wc, inputs[t]) + matvec(Uc, h_prev) + bc;

    z_o[t] = matvec(Wo, inputs[t]) + matvec(Uo, h_prev) + bo;

    Tensor1D f = z_f[t];

    for (auto& val : f) val = sigmoid(val);

    Tensor1D i = z_i[t];

    for (auto& val : i) val = sigmoid(val);

    Tensor1D cand = z_c[t];

    for (auto& val : cand) val = tanh_activation(val);

    Tensor1D o = z_o[t];

    for (auto& val : o) val = sigmoid(val);

    Tensor1D cell(hidden_size);

    for (int j = 0; j < hidden_size; ++j) {

      cell[j] = f[j] * c_prev[j] + i[j] * cand[j];

    }

    Tensor1D h = cell;

    for (auto& val : h) val = tanh_activation(val);

    for (int j = 0; j < hidden_size; ++j) {

      h[j] *= o[j];

    }

    outputs[t] = h;

    cells[t+1] = cell;

    h_prev = h;

    c_prev = cell;

  }

  return outputs;

}

void LSTMLayer::compute_deltas_seq(const Tensor2D& grad_output, const Tensor2D& /*z_val*/) {

  int seq_len = grad_output.size();

  deltas_h.resize(seq_len, Tensor1D(hidden_size, 0.0f));

  deltas_c.resize(seq_len, Tensor1D(hidden_size, 0.0f));

  Tensor1D grad_h(hidden_size, 0.0f);

  Tensor1D grad_c(hidden_size, 0.0f);

  for (int t = seq_len - 1; t >= 0; --t) {

    for (int i = 0; i < hidden_size; ++i) {

      grad_h[i] += grad_output[t][i];

    }

    deltas_h[t] = grad_h;

    Tensor1D tanh_c(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      tanh_c[i] = tanh_activation(cells[t+1][i]);

    }

    Tensor1D grad_o(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      grad_o[i] = grad_h[i] * tanh_c[i];

    }

    Tensor1D grad_tanh_c(hidden_size);

    Tensor1D o(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      o[i] = sigmoid(z_o[t][i]);

      grad_tanh_c[i] = grad_h[i] * o[i];

    }

    Tensor1D grad_c_this(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      grad_c_this[i] = grad_tanh_c[i] * tanh_derivative(cells[t+1][i]) + grad_c[i];

    }

    deltas_c[t] = grad_c_this;

    Tensor1D grad_f(hidden_size);

    Tensor1D f(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      f[i] = sigmoid(z_f[t][i]);

      grad_f[i] = grad_c_this[i] * cells[t][i];

    }

    Tensor1D grad_i(hidden_size);

    Tensor1D i(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      i[i] = sigmoid(z_i[t][i]);

      grad_i[i] = grad_c_this[i] * tanh_activation(z_c[t][i]);

    }

    Tensor1D grad_cand(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      grad_cand[i] = grad_c_this[i] * i[i];

    }

    Tensor1D grad_c_prev(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      grad_c_prev[i] = grad_c_this[i] * f[i];

    }

    for (int i = 0; i < hidden_size; ++i) {

      grad_f[i] *= f[i] * (1 - f[i]);

      grad_i[i] *= i[i] * (1 - i[i]);

      grad_o[i] *= o[i] * (1 - o[i]);

      grad_cand[i] *= tanh_derivative(z_c[t][i]);

    }

    Tensor1D grad_h_prev(hidden_size, 0.0f);

    for (int i = 0; i < hidden_size; ++i) {

      for (int j = 0; j < hidden_size; ++j) {

        grad_h_prev[i] += grad_f[j] * Uf[j][i];

        grad_h_prev[i] += grad_i[j] * Ui[j][i];

        grad_h_prev[i] += grad_cand[j] * Uc[j][i];

        grad_h_prev[i] += grad_o[j] * Uo[j][i];

      }

    }

    grad_h = grad_h_prev;

    grad_c = grad_c_prev;

  }

}

Tensor2D LSTMLayer::get_grad_input_seq() {

  int seq_len = deltas_h.size();

  Tensor2D grad_input(seq_len, Tensor1D(input_size, 0.0f));

  for (int t = 0; t < seq_len; ++t) {

    Tensor1D grad_h = deltas_h[t];

    Tensor1D grad_c_this = deltas_c[t];

    Tensor1D tanh_c(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      tanh_c[i] = tanh_activation(cells[t+1][i]);

    }

    Tensor1D grad_o(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      grad_o[i] = grad_h[i] * tanh_c[i];

    }

    Tensor1D grad_tanh_c(hidden_size);

    Tensor1D o(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      o[i] = sigmoid(z_o[t][i]);

      grad_tanh_c[i] = grad_h[i] * o[i];

    }

    Tensor1D grad_f(hidden_size);

    Tensor1D f(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      f[i] = sigmoid(z_f[t][i]);

      grad_f[i] = grad_c_this[i] * cells[t][i];

    }

    Tensor1D grad_i(hidden_size);

    Tensor1D i(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      i[i] = sigmoid(z_i[t][i]);

      grad_i[i] = grad_c_this[i] * tanh_activation(z_c[t][i]);

    }

    Tensor1D grad_cand(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      grad_cand[i] = grad_c_this[i] * i[i];

    }

    for (int i = 0; i < hidden_size; ++i) {

      grad_f[i] *= f[i] * (1 - f[i]);

      grad_i[i] *= i[i] * (1 - i[i]);

      grad_o[i] *= o[i] * (1 - o[i]);

      grad_cand[i] *= tanh_derivative(z_c[t][i]);

    }

    for (int k = 0; k < input_size; ++k) {

      for (int j = 0; j < hidden_size; ++j) {

        grad_input[t][k] += grad_f[j] * Wf[j][k];

        grad_input[t][k] += grad_i[j] * Wi[j][k];

        grad_input[t][k] += grad_cand[j] * Wc[j][k];

        grad_input[t][k] += grad_o[j] * Wo[j][k];

      }

    }

  }

  return grad_input;

}

void LSTMLayer::update_params_seq(const Tensor2D& inputs, float lr, int t, float beta1, float beta2, float epsilon) {

  int seq_len = inputs.size();

  Tensor2D grad_Wf(hidden_size, Tensor1D(input_size, 0.0f));

  Tensor2D grad_Wi(hidden_size, Tensor1D(input_size, 0.0f));

  Tensor2D grad_Wc(hidden_size, Tensor1D(input_size, 0.0f));

  Tensor2D grad_Wo(hidden_size, Tensor1D(input_size, 0.0f));

  Tensor2D grad_Uf(hidden_size, Tensor1D(hidden_size, 0.0f));

  Tensor2D grad_Ui(hidden_size, Tensor1D(hidden_size, 0.0f));

  Tensor2D grad_Uc(hidden_size, Tensor1D(hidden_size, 0.0f));

  Tensor2D grad_Uo(hidden_size, Tensor1D(hidden_size, 0.0f));

  Tensor1D grad_bf(hidden_size, 0.0f);

  Tensor1D grad_bi(hidden_size, 0.0f);

  Tensor1D grad_bc(hidden_size, 0.0f);

  Tensor1D grad_bo(hidden_size, 0.0f);

  for (int s = 0; s < seq_len; ++s) {

    Tensor1D h_prev = (s == 0) ? Tensor1D(hidden_size, 0.0f) : outputs[s-1];

    Tensor1D grad_h = deltas_h[s];

    Tensor1D grad_c_this = deltas_c[s];

    Tensor1D tanh_c(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      tanh_c[i] = tanh_activation(cells[s+1][i]);

    }

    Tensor1D grad_o(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      grad_o[i] = grad_h[i] * tanh_c[i];

    }

    Tensor1D grad_tanh_c(hidden_size);

    Tensor1D o(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      o[i] = sigmoid(z_o[s][i]);

      grad_tanh_c[i] = grad_h[i] * o[i];

    }

    Tensor1D grad_f(hidden_size);

    Tensor1D f(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      f[i] = sigmoid(z_f[s][i]);

      grad_f[i] = grad_c_this[i] * cells[s][i];

    }

    Tensor1D grad_i(hidden_size);

    Tensor1D i(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      i[i] = sigmoid(z_i[s][i]);

      grad_i[i] = grad_c_this[i] * tanh_activation(z_c[s][i]);

    }

    Tensor1D grad_cand(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {

      grad_cand[i] = grad_c_this[i] * i[i];

    }

    for (int i = 0; i < hidden_size; ++i) {

      grad_f[i] *= f[i] * (1 - f[i]);

      grad_i[i] *= i[i] * (1 - i[i]);

      grad_o[i] *= o[i] * (1 - o[i]);

      grad_cand[i] *= tanh_derivative(z_c[s][i]);

    }

    for (int j = 0; j < hidden_size; ++j) {

      for (int k = 0; k < input_size; ++k) {

        grad_Wf[j][k] += grad_f[j] * inputs[s][k];

        grad_Wi[j][k] += grad_i[j] * inputs[s][k];

        grad_Wc[j][k] += grad_cand[j] * inputs[s][k];

        grad_Wo[j][k] += grad_o[j] * inputs[s][k];

      }

      for (int k = 0; k < hidden_size; ++k) {

        grad_Uf[j][k] += grad_f[j] * h_prev[k];

        grad_Ui[j][k] += grad_i[j] * h_prev[k];

        grad_Uc[j][k] += grad_cand[j] * h_prev[k];

        grad_Uo[j][k] += grad_o[j] * h_prev[k];

      }

      grad_bf[j] += grad_f[j];

      grad_bi[j] += grad_i[j];

      grad_bc[j] += grad_cand[j];

      grad_bo[j] += grad_o[j];

    }

  }

  auto update = [beta1, beta2, epsilon, lr, t](Tensor2D& W, Tensor2D& m, Tensor2D& v, const Tensor2D& g) {

    for (int j = 0; j < W.size(); ++j) {

      for (int i = 0; i < W[0].size(); ++i) {

        float gg = g[j][i];

        m[j][i] = beta1 * m[j][i] + (1 - beta1) * gg;

        v[j][i] = beta2 * v[j][i] + (1 - beta2) * gg * gg;

        float m_hat = m[j][i] / (1 - std::pow(beta1, t));

        float v_hat = v[j][i] / (1 - std::pow(beta2, t));

        W[j][i] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);

      }

    }

  };

  update(Wf, m_Wf, v_Wf, grad_Wf);

  update(Wi, m_Wi, v_Wi, grad_Wi);

  update(Wc, m_Wc, v_Wc, grad_Wc);

  update(Wo, m_Wo, v_Wo, grad_Wo);

  update(Uf, m_Uf, v_Uf, grad_Uf);

  update(Ui, m_Ui, v_Ui, grad_Ui);

  update(Uc, m_Uc, v_Uc, grad_Uc);

  update(Uo, m_Uo, v_Uo, grad_Uo);

  auto update_v = [beta1, beta2, epsilon, lr, t](Tensor1D& b, Tensor1D& m, Tensor1D& v, const Tensor1D& g) {

    for (int j = 0; j < b.size(); ++j) {

      float gg = g[j];

      m[j] = beta1 * m[j] + (1 - beta1) * gg;

      v[j] = beta2 * v[j] + (1 - beta2) * gg * gg;

      float m_hat = m[j] / (1 - std::pow(beta1, t));

      float v_hat = v[j] / (1 - std::pow(beta2, t));

      b[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);

    }

  };

  update_v(bf, m_bf, v_bf, grad_bf);

  update_v(bi, m_bi, v_bi, grad_bi);

  update_v(bc, m_bc, v_bc, grad_bc);

  update_v(bo, m_bo, v_bo, grad_bo);

}

Tensor2D LSTMLayer::get_outputs_seq() const { return outputs; }

Tensor2D LSTMLayer::get_z_seq() const { return z_o; } // Arbitrary, since not used

Tensor2D LSTMLayer::get_deltas_seq() const { return deltas_h; }

void LSTMLayer::save(std::ostream& os) const {

  os << input_size << " " << hidden_size << " " << static_cast<int>(activation_type) << "\n";

  auto save_mat = [&os](const Tensor2D& M) {

    for (const auto& row : M) {

      for (float v : row) os << v << " ";

      os << "\n";

    }

  };

  save_mat(Wf);

  save_mat(Wi);

  save_mat(Wc);

  save_mat(Wo);

  save_mat(Uf);

  save_mat(Ui);

  save_mat(Uc);

  save_mat(Uo);

  auto save_vec = [&os](const Tensor1D& v) {

    for (float b : v) os << b << " ";

    os << "\n";

  };

  save_vec(bf);

  save_vec(bi);

  save_vec(bc);

  save_vec(bo);

}

void LSTMLayer::load(std::istream& is) {

  int act_int;

  is >> input_size >> hidden_size >> act_int;

  activation_type = static_cast<ActivationType>(act_int);

  act = get_activation(activation_type);

  deriv = get_derivative(activation_type);

  auto load_mat = [&is](Tensor2D& M) {

    for (auto& row : M) {

      for (auto& v : row) is >> v;

    }

  };

  load_mat(Wf);

  load_mat(Wi);

  load_mat(Wc);

  load_mat(Wo);

  load_mat(Uf);

  load_mat(Ui);

  load_mat(Uc);

  load_mat(Uo);

  auto load_vec = [&is](Tensor1D& v) {

    for (auto& b : v) is >> b;

  };

  load_vec(bf);

  load_vec(bi);

  load_vec(bc);

  load_vec(bo);

  // Reset Adam

  m_Wf = Tensor2D(hidden_size, Tensor1D(input_size, 0.0f));

  // Similarly for others

}

LayerType LSTMLayer::get_layer_type() const { return LayerType::LSTM; }

} // namespace nn
