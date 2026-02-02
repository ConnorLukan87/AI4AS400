// network.cpp
#include "network.h"

namespace nn {

NeuralNetwork::NeuralNetwork() : loss_type(LossType::Type::MSE), custom_loss(nullptr), custom_loss_deriv(nullptr) {}

void NeuralNetwork::set_loss(LossType type) {

  loss_type = type;

}

void NeuralNetwork::set_custom_loss(LossFunc loss, LossDerivFunc deriv) {

  custom_loss = loss;

  custom_loss_deriv = deriv;

}

void NeuralNetwork::add_layer(int in_size, int out_size, ActivationType act) {

  layers.emplace_back(in_size, out_size, act);

}

Tensor1D NeuralNetwork::predict(const Tensor1D& inputs) {

  Tensor1D out = inputs;

  for (auto& l : layers) {

    out = l.forward1D(out);

  }

  return out;

}

float NeuralNetwork::compute_loss(const Tensor1D& output, const Tensor1D& target) {

  if (custom_loss) return custom_loss(output, target);

  if (loss_type == LossType::Type::MSE) {

    float sum = 0.0f;

    for (size_t i = 0; i < output.size(); ++i) {

      float d = output[i] - target[i];

      sum += d * d;

    }

    return sum / output.size();

  } else if (loss_type == LossType::Type::CROSS_ENTROPY_BINARY) {

    float sum = 0.0f;

    for (size_t i = 0; i < output.size(); ++i) {

      sum += target[i] * std::log(output[i] + 1e-10f) + (1 - target[i]) * std::log(1 - output[i] + 1e-10f);

    }

    return -sum / output.size();

  } else if (loss_type == LossType::Type::CROSS_ENTROPY_MULTI) {

    float sum = 0.0f;

    for (size_t i = 0; i < output.size(); ++i) {

      sum += target[i] * std::log(output[i] + 1e-10f);

    }

    return -sum;

  }

  throw std::invalid_argument("Unknown loss type");

}

Tensor1D NeuralNetwork::compute_loss_deriv(const Tensor1D& output, const Tensor1D& target) {

  if (custom_loss_deriv) return custom_loss_deriv(output, target);

  Tensor1D deriv(output.size());

  if (loss_type == LossType::Type::MSE) {

    for (size_t i = 0; i < output.size(); ++i) {

      deriv[i] = 2.0f * (output[i] - target[i]) / output.size();

    }

  } else if (loss_type == LossType::Type::CROSS_ENTROPY_BINARY) {

    for (size_t i = 0; i < output.size(); ++i) {

      deriv[i] = (output[i] - target[i]) / (output[i] * (1 - output[i]) + 1e-10f);

    }

  } else if (loss_type == LossType::Type::CROSS_ENTROPY_MULTI) {

    for (size_t i = 0; i < output.size(); ++i) {

      deriv[i] = output[i] - target[i];

    }

  }

  return deriv;

}

void NeuralNetwork::train(const std::vector<Tensor1D>& X, const std::vector<Tensor1D>& y, float lr, int epochs, float beta1, float beta2, float epsilon) {

  int n = X.size();

  for (int e = 1; e <= epochs; ++e) {

    for (int i = 0; i < n; ++i) {

      Tensor1D out = predict(X[i]);

      Tensor1D grad_out = compute_loss_deriv(out, y[i]);

      Tensor1D grad = grad_out;

      for (int l = layers.size() - 1; l >= 0; --l) {

        layers[l].compute_deltas1D(grad, layers[l].get_z1D());

        grad = layers[l].get_grad_input1D();

      }

      Tensor1D prev = X[i];

      for (int l = 0; l < layers.size(); ++l) {

        layers[l].update_params(prev, lr, e, beta1, beta2, epsilon);

        prev = layers[l].get_outputs1D();

      }

    }

  }

}

void NeuralNetwork::save(const std::string& filename) const {

  std::ofstream os(filename);

  os << layers.size() << "\n";

  for (const auto& l : layers) {

    l.save(os);

  }

}

void NeuralNetwork::load(const std::string& filename) {

  std::ifstream is(filename);

  size_t num;

  is >> num;

  layers.clear();

  for (size_t i = 0; i < num; ++i) {

    layers.emplace_back(0, 0, ActivationType::LINEAR);

    layers.back().load(is);

  }

}

CNN::CNN() : loss_type(LossType::Type::MSE), custom_loss(nullptr), custom_loss_deriv(nullptr) {}

CNN::~CNN() {

  clear_layers();

}

void CNN::set_loss(LossType type) {

  loss_type = type;

}

void CNN::set_custom_loss(LossFunc loss, LossDerivFunc deriv) {

  custom_loss = loss;

  custom_loss_deriv = deriv;

}

void CNN::add_conv_layer(int in_ch, int out_ch, int k_size, int str, ActivationType act) {

  layers.push_back(new ConvLayer(in_ch, out_ch, k_size, str, act));

  layer_types.push_back(LayerType::CONV);

}

void CNN::add_flatten_layer() {

  layers.push_back(new FlattenLayer());

  layer_types.push_back(LayerType::FLATTEN);

}

void CNN::add_dense_layer(int in_size, int out_size, ActivationType act) {

  layers.push_back(new DenseLayer(in_size, out_size, act));

  layer_types.push_back(LayerType::DENSE);

}

void CNN::add_rnn_layer(int in_size, int hid_size, ActivationType act) {

  layers.push_back(new RNNLayer(in_size, hid_size, act));

  layer_types.push_back(LayerType::RNN);

}

void CNN::add_lstm_layer(int in_size, int hid_size, ActivationType act) {

  layers.push_back(new LSTMLayer(in_size, hid_size, act));

  layer_types.push_back(LayerType::LSTM);

}

Tensor1D CNN::predict(const Tensor3D& input) {

  Tensor3D current3 = input;

  Tensor1D current1;

  Tensor2D current2;

  for (size_t i = 0; i < layers.size(); ++i) {

    LayerType type = layer_types[i];

    if (type == LayerType::CONV) {

      current3 = layers[i]->forward3D(current3);

    } else if (type == LayerType::FLATTEN) {

      current1 = layers[i]->flatten_forward(current3);

    } else if (type == LayerType::DENSE) {

      current1 = layers[i]->forward1D(current1);

    } else if (type == LayerType::RNN || type == LayerType::LSTM) {

      // Assume input for RNN is seq, but since input is 3D, perhaps reshape, but placeholder

      throw std::runtime_error("RNN/LSTM not supported in this predict method");

    }

  }

  return current1;

}

float CNN::calculate_loss(const Tensor1D& output, const Tensor1D& target) {

  return NeuralNetwork().compute_loss(output, target); // Reuse

}

Tensor1D CNN::calculate_loss_deriv(const Tensor1D& output, const Tensor1D& target) {

  return NeuralNetwork().compute_loss_deriv(output, target); // Reuse

}

void CNN::train(const std::vector<Tensor3D>& X, const std::vector<Tensor1D>& y, float lr, int epochs, float beta1, float beta2, float epsilon) {

  int n = X.size();

  for (int e = 1; e <= epochs; ++e) {

    for (int i = 0; i < n; ++i) {

      Tensor1D out = predict(X[i]);

      Tensor1D grad_out = calculate_loss_deriv(out, y[i]);

      Tensor1D grad1 = grad_out;

      Tensor3D grad3;

      Tensor2D grad2;

      for (int l = layers.size() - 1; l >= 0; --l) {

        LayerType type = layer_types[l];

        if (type == LayerType::DENSE) {

          layers[l]->compute_deltas1D(grad1, layers[l]->get_z1D());

          grad1 = layers[l]->get_grad_input1D();

        } else if (type == LayerType::FLATTEN) {

          layers[l]->compute_deltas1D(grad1, layers[l]->get_z1D());

          grad3 = layers[l]->get_grad_input3D();

        } else if (type == LayerType::CONV) {

          layers[l]->compute_deltas3D(grad3, layers[l]->get_z3D());

          grad3 = layers[l]->get_grad_input3D();

        } else if (type == LayerType::RNN || type == LayerType::LSTM) {

          // Placeholder

          layers[l]->compute_deltas_seq(grad2, layers[l]->get_z_seq());

          grad2 = layers[l]->get_grad_input_seq();

        }

      }

      Tensor3D prev3 = X[i];

      Tensor1D prev1;

      Tensor2D prev2;

      for (int l = 0; l < layers.size(); ++l) {

        LayerType type = layer_types[l];

        if (type == LayerType::CONV) {

          layers[l]->update_params3D(prev3, lr, e, beta1, beta2, epsilon);

          prev3 = layers[l]->get_outputs3D();

        } else if (type == LayerType::FLATTEN) {

          // No update

          prev1 = layers[l]->get_outputs1D();

        } else if (type == LayerType::DENSE) {

          layers[l]->update_params(prev1, lr, e, beta1, beta2, epsilon);

          prev1 = layers[l]->get_outputs1D();

        } else if (type == LayerType::RNN || type == LayerType::LSTM) {

          layers[l]->update_params_seq(prev2, lr, e, beta1, beta2, epsilon);

          prev2 = layers[l]->get_outputs_seq();

        }

      }

    }

  }

}

void CNN::save(const std::string& filename) const {

  std::ofstream os(filename);

  os << layers.size() << "\n";

  for (auto t : layer_types) os << static_cast<int>(t) << " ";

  os << "\n";

  for (const auto& l : layers) {

    l->save(os);

  }

}

void CNN::load(const std::string& filename) {

  std::ifstream is(filename);

  size_t num;

  is >> num;

  layer_types.resize(num);

  for (auto& t : layer_types) {

    int int_t;

    is >> int_t;

    t = static_cast<LayerType>(int_t);

  }

  clear_layers();

  for (size_t i = 0; i < num; ++i) {

    switch (layer_types[i]) {

      case LayerType::DENSE:

        layers.push_back(new DenseLayer(0, 0, ActivationType::LINEAR));

        break;

      case LayerType::CONV:

        layers.push_back(new ConvLayer(0, 0, 0, 0, ActivationType::LINEAR));

        break;

      case LayerType::FLATTEN:

        layers.push_back(new FlattenLayer());

        break;

      case LayerType::RNN:

        layers.push_back(new RNNLayer(0, 0, ActivationType::LINEAR));

        break;

      case LayerType::LSTM:

        layers.push_back(new LSTMLayer(0, 0, ActivationType::LINEAR));

        break;

    }

    layers.back()->load(is);

  }

}

YOLO::YOLO(int gs, int nb, int nc, const std::vector<Tensor1D>& anch) : CNN(), grid_size(gs), num_boxes(nb), num_classes(nc), anchors(anch) {}

float YOLO::yolo_loss(const Tensor1D& output, const Tensor1D& target) {

  // Implement YOLO-specific loss (bbox, objectness, class)

  // Placeholder

  return compute_loss(output, target); // Fall back

}

std::vector<std::vector<float>> YOLO::non_max_suppression(const Tensor1D& pred, float iou_thres, float conf_thres) {

  // Parse pred to boxes (x,y,w,h,conf,classes)

  // Sort by conf, suppress high IOU

  // Placeholder

  return {};

}

std::vector<std::vector<float>> YOLO::predict_objects(const Tensor3D& input, float conf_thres, float iou_thres) {

  Tensor1D pred = predict(input);

  return non_max_suppression(pred, iou_thres, conf_thres);

}

Tree::Tree(int md, float l, float g) : max_depth(md), lambda(l), gamma(g), root(nullptr) {}

Tree::~Tree() {

  delete root;

}

void Tree::fit(const std::vector<Tensor1D>& X, const Tensor1D& gradients, const Tensor1D& hessians) {

  std::vector<int> indices(X.size());

  std::iota(indices.begin(), indices.end(), 0);

  root = build(X, gradients, hessians, indices, 0);

}

Node* Tree::build(const std::vector<Tensor1D>& X, const Tensor1D& gradients, const Tensor1D& hessians, const std::vector<int>& indices, int depth) {

  if (depth >= max_depth || indices.size() <= 1) {

    Node* leaf = new Node();

    leaf->is_leaf = true;

    float g_sum = 0.0f;

    float h_sum = 0.0f;

    for (int i : indices) {

      g_sum += gradients[i];

      h_sum += hessians[i];

    }

    leaf->value = -g_sum / (h_sum + lambda);

    return leaf;

  }

  float g_sum = 0.0f;

  float h_sum = 0.0f;

  for (int i : indices) {

    g_sum += gradients[i];

    h_sum += hessians[i];

  }

  float score = g_sum * g_sum / (h_sum + lambda);

  float best_gain = -std::numeric_limits<float>::infinity();

  int best_feature = -1;

  float best_thres = 0.0f;

  int num_features = X[0].size();

  for (int f = 0; f < num_features; ++f) {

    std::vector<int> sorted = indices;

    std::sort(sorted.begin(), sorted.end(), [&X, f](int a, int b) { return X[a][f] < X[b][f]; });

    float g_left = 0.0f;

    float h_left = 0.0f;

    for (size_t j = 0; j < sorted.size() - 1; ++j) {

      g_left += gradients[sorted[j]];

      h_left += hessians[sorted[j]];

      float g_right = g_sum - g_left;

      float h_right = h_sum - h_left;

      float gain = g_left * g_left / (h_left + lambda) + g_right * g_right / (h_right + lambda) - score - gamma;

      if (gain > best_gain) {

        best_gain = gain;

        best_feature = f;

        best_thres = (X[sorted[j]][f] + X[sorted[j+1]][f]) / 2.0f;

      }

    }

  }

  if (best_gain <= 0 || best_feature == -1) {

    Node* leaf = new Node();

    leaf->is_leaf = true;

    float g_sum = 0.0f;

    float h_sum = 0.0f;

    for (int i : indices) {

      g_sum += gradients[i];

      h_sum += hessians[i];

    }

    leaf->value = -g_sum / (h_sum + lambda);

    return leaf;

  }

  std::vector<int> left, right;

  for (int i : indices) {

    if (X[i][best_feature] <= best_thres) left.push_back(i);

    else right.push_back(i);

  }

  Node* node = new Node();

  node->feature = best_feature;

  node->threshold = best_thres;

  node->left = build(X, gradients, hessians, left, depth + 1);

  node->right = build(X, gradients, hessians, right, depth + 1);

  return node;

}

float Tree::predict(const Tensor1D& x) const {

  return predict_node(root, x);

}

float Tree::predict_node(Node* node, const Tensor1D& x) const {

  if (node->is_leaf) return node->value;

  if (x[node->feature] <= node->threshold) return predict_node(node->left, x);

  return predict_node(node->right, x);

}

void Tree::save(std::ostream& os) const {

  save_node(os, root);

}

void Tree::save_node(std::ostream& os, Node* node) const {

  if (node->is_leaf) {

    os << "leaf " << node->value << "\n";

    return;

  }

  os << "node " << node->feature << " " << node->threshold << "\n";

  save_node(os, node->left);

  save_node(os, node->right);

}

void Tree::load(std::istream& is) {

  root = load_node(is);

}

Node* Tree::load_node(std::istream& is) {

  std::string type;

  is >> type;

  Node* node = new Node();

  if (type == "leaf") {

    is >> node->value;

    node->is_leaf = true;

  } else {

    is >> node->feature >> node->threshold;

    node->left = load_node(is);

    node->right = load_node(is);

  }

  return node;

}

XGBoost::XGBoost() {}

XGBoost::~XGBoost() {

  clear_trees();

}

void XGBoost::set_loss(LossType type) {

  loss_type = type;

}

void XGBoost::set_custom_loss(LossFunc loss, LossDerivFunc deriv) {

  custom_loss = loss;

  custom_loss_deriv = deriv;

}

void XGBoost::train(const std::vector<Tensor1D>& X, const std::vector<Tensor1D>& y) {

  int n = X.size();

  Tensor1D pred(n, 0.5f);

  for (int round = 0; round < num_boost_round; ++round) {

    Tensor1D gradients(n, 0.0f);

    Tensor1D hessians(n, 0.0f);

    for (int i = 0; i < n; ++i) {

      Tensor1D output(1, pred[i]);

      Tensor1D target = y[i];

      Tensor1D deriv = compute_loss_deriv(output, target);

      gradients[i] = deriv[0];

      // Hessians

      if (loss_type == LossType::Type::MSE) {

        hessians[i] = 2.0f;

      } else if (loss_type == LossType::Type::CROSS_ENTROPY_BINARY) {

        hessians[i] = pred[i] * (1 - pred[i]);

      } // Add others

    }

    Tree* tree = new Tree(max_depth, lambda, gamma);

    tree->fit(X, gradients, hessians);

    trees.push_back(tree);

    for (int i = 0; i < n; ++i) {

      pred[i] += learning_rate * tree->predict(X[i]);

    }

  }

}

Tensor1D XGBoost::predict(const Tensor1D& x) const {

  float p = 0.5f;

  for (auto t : trees) p += learning_rate * t->predict(x);

  Tensor1D out(1, p);

  return out;

}

float XGBoost::compute_loss(const Tensor1D& output, const Tensor1D& target) const {

  if (custom_loss) return custom_loss(output, target);

  // Similar to NeuralNetwork

  return 0.0f; // Placeholder

}

void XGBoost::save(const std::string& filename) const {

  std::ofstream os(filename);

  os << trees.size() << "\n";

  for (auto t : trees) t->save(os);

}

void XGBoost::load(const std::string& filename) {

  std::ifstream is(filename);

  size_t num;

  is >> num;

  clear_trees();

  for (size_t i = 0; i < num; ++i) {

    Tree* t = new Tree(0, 0.0f, 0.0f);

    t->load(is);

    trees.push_back(t);

  }

}

void XGBoost::clear_trees() {

  for (auto t : trees) delete t;

  trees.clear();

}

} // namespace nn