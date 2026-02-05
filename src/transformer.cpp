// transformer.cpp
#include "transformer.h"

namespace nn {

LayerNorm::LayerNorm(int d, float eps) : dim(d), gamma(d, 1.0f), beta(d, 0.0f), epsilon(eps), normalized(d, 0.0f), input_cache(d, 0.0f), mean_cache(0.0f), var_cache(0.0f), m_gamma(d, 0.0f), v_gamma(d, 0.0f), m_beta(d, 0.0f), v_beta(d, 0.0f) {}

Tensor1D LayerNorm::forward(const Tensor1D& x) {

  input_cache = x;

  float mean = 0.0f;

  for (float val : x) mean += val;

  mean /= dim;

  mean_cache = mean;

  float var = 0.0f;

  for (float val : x) var += (val - mean) * (val - mean);

  var /= dim;

  var_cache = var;

  Tensor1D norm(dim);

  float std = std::sqrt(var + epsilon);

  for (int i = 0; i < dim; ++i) {

    norm[i] = (x[i] - mean) / std * gamma[i] + beta[i];

  }

  normalized = norm;

  return norm;

}

Tensor1D LayerNorm::backward(const Tensor1D& grad_out) {

  Tensor1D grad_x(dim, 0.0f);

  float std = std::sqrt(var_cache + epsilon);

  Tensor1D dnorm(dim);

  for (int i = 0; i < dim; ++i) {

    dnorm[i] = grad_out[i] * gamma[i];

  }

  float dvar = 0.0f;

  for (int i = 0; i < dim; ++i) {

    dvar += dnorm[i] * (input_cache[i] - mean_cache) * -0.5f / (std * std * std);

  }

  float dmean = 0.0f;

  for (int i = 0; i < dim; ++i) {

    dmean += dnorm[i] * -1.0f / std;

  }

  float sum_temp = 0.0f;

  for (float val : input_cache) sum_temp += val - mean_cache;

  dmean += dvar * -2.0f / dim * sum_temp;

  for (int i = 0; i < dim; ++i) {

    grad_x[i] = dnorm[i] / std + dvar * 2.0f / dim * (input_cache[i] - mean_cache) + dmean / dim;

  }

  return grad_x;

}

void LayerNorm::save(std::ostream& os) const {

  for (float g : gamma) os << g << " ";

  os << "\n";

  for (float b : beta) os << b << " ";

  os << "\n";

}

void LayerNorm::load(std::istream& is) {

  for (auto& g : gamma) is >> g;

  for (auto& b : beta) is >> b;

}

Embedding::Embedding(int vocab, int dim) : vocab_size(vocab), embed_dim(dim), weights(vocab, Tensor1D(dim, 0.0f)), m_weights(vocab, Tensor1D(dim, 0.0f)), v_weights(vocab, Tensor1D(dim, 0.0f)) {

  if (!random_seeded) {

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    random_seeded = true;

  }

  float scale = std::sqrt(1.0f / dim);

  for (auto& row : weights) {

    for (auto& w : row) {

      w = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;

    }

  }

}

Tensor1D Embedding::forward(int token_id) const {

  if (token_id < 0 || token_id >= vocab_size) throw std::invalid_argument("Invalid token ID");

  return weights[token_id];

}

Tensor2D Embedding::forward_seq(const std::vector<int>& tokens) const {

  Tensor2D emb(tokens.size(), Tensor1D(embed_dim));

  for (size_t i = 0; i < tokens.size(); ++i) {

    emb[i] = forward(tokens[i]);

  }

  return emb;

}

void Embedding::save(std::ostream& os) const {

  for (const auto& row : weights) {

    for (float w : row) os << w << " ";

    os << "\n";

  }

}

void Embedding::load(std::istream& is) {

  for (auto& row : weights) {

    for (auto& w : row) is >> w;

  }

}

RoPE::RoPE(int d, int max_len, float base) : dim(d), max_seq_len(max_len), cos_cache(max_len, Tensor1D(d, 0.0f)), sin_cache(max_len, Tensor1D(d, 0.0f)) {

  for (int pos = 0; pos < max_len; ++pos) {

    for (int i = 0; i < d; i += 2) {

      float theta = pos / std::pow(base, static_cast<float>(i) / d);

      cos_cache[pos][i] = std::cos(theta);

      sin_cache[pos][i] = std::sin(theta);

      if (i + 1 < d) {

        cos_cache[pos][i + 1] = std::cos(theta);

        sin_cache[pos][i + 1] = std::sin(theta);

      }

    }

  }

}

Tensor1D RoPE::apply(const Tensor1D& x, int pos) const {

  if (pos >= max_seq_len) throw std::invalid_argument("Position exceeds max seq len");

  Tensor1D rotated = x;

  for (int i = 0; i < dim; i += 2) {

    float x1 = x[i];

    float x2 = (i + 1 < dim) ? x[i + 1] : 0.0f;

    rotated[i] = x1 * cos_cache[pos][i] - x2 * sin_cache[pos][i];

    if (i + 1 < dim) {

      rotated[i + 1] = x1 * sin_cache[pos][i] + x2 * cos_cache[pos][i];

    }

  }

  return rotated;

}

Tensor2D Attention::scaled_dot_product(const Tensor2D& Q, const Tensor2D& K, const Tensor2D& V, bool causal_mask) {

  int seq = Q.size();

  int d = Q[0].size();

  Tensor2D KT(K[0].size(), Tensor1D(K.size()));

  for (int i = 0; i < K.size(); ++i) {

    for (int j = 0; j < K[0].size(); ++j) {

      KT[j][i] = K[i][j];

    }

  }

  Tensor2D scores = matmul(Q, KT);

  float scale = 1.0f / std::sqrt(static_cast<float>(d));

  for (auto& row : scores) {

    for (auto& s : row) s *= scale;

  }

  if (causal_mask) {

    for (int i = 0; i < seq; ++i) {

      for (int j = i + 1; j < seq; ++j) {

        scores[i][j] = -std::numeric_limits<float>::infinity();

      }

    }

  }

  // Softmax per row

  for (int i = 0; i < seq; ++i) {

    float max_val = *std::max_element(scores[i].begin(), scores[i].end());

    float sum = 0.0f;

    for (auto& s : scores[i]) {

      s = std::exp(s - max_val);

      sum += s;

    }

    for (auto& s : scores[i]) s /= sum;

  }

  Tensor2D attn = matmul(scores, V);

  return attn;

}

MultiHeadAttention::MultiHeadAttention(int dim, int heads, int max_seq, bool bias) : embed_dim(dim), num_heads(heads), head_dim(dim / heads), W_q(dim, Tensor1D(dim, 0.0f)), W_k(dim, Tensor1D(dim, 0.0f)), W_v(dim, Tensor1D(dim, 0.0f)), W_o(dim, Tensor1D(dim, 0.0f)), use_bias(bias), b_q(dim, 0.0f), b_k(dim, 0.0f), b_v(dim, 0.0f), b_o(dim, 0.0f), rope(new RoPE(dim, max_seq)), k_cache(max_seq, Tensor1D(dim, 0.0f)), v_cache(max_seq, Tensor1D(dim, 0.0f)), cache_len(0), m_Wq(dim, Tensor1D(dim, 0.0f)), v_Wq(dim, Tensor1D(dim, 0.0f)), m_Wk(dim, Tensor1D(dim, 0.0f)), v_Wk(dim, Tensor1D(dim, 0.0f)), m_Wv(dim, Tensor1D(dim, 0.0f)), v_Wv(dim, Tensor1D(dim, 0.0f)), m_Wo(dim, Tensor1D(dim, 0.0f)), v_Wo(dim, Tensor1D(dim, 0.0f)) {

  if (dim % heads != 0) throw std::invalid_argument("dim not divisible by heads");

  init_weight(W_q, dim, dim, 1.0f / std::sqrt(static_cast<float>(dim)));

  init_weight(W_k, dim, dim, 1.0f / std::sqrt(static_cast<float>(dim)));

  init_weight(W_v, dim, dim, 1.0f / std::sqrt(static_cast<float>(dim)));

  init_weight(W_o, dim, dim, 1.0f / std::sqrt(static_cast<float>(dim)));

  init_adam(m_Wq, v_Wq, dim, dim);

  // Similar for others

  if (use_bias) {

    // Initialize biases if needed

  }

}

void MultiHeadAttention::init_weight(Tensor2D& W, int rows, int cols, float scale) {

  W.resize(rows, Tensor1D(cols, 0.0f));

  if (!random_seeded) {

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    random_seeded = true;

  }

  for (auto& row : W) {

    for (auto& w : row) {

      w = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;

    }

  }

}

void MultiHeadAttention::init_adam(Tensor2D& m, Tensor2D& v, int rows, int cols) {

  m.resize(rows, Tensor1D(cols, 0.0f));

  v.resize(rows, Tensor1D(cols, 0.0f));

}

void MultiHeadAttention::add_bias(Tensor2D& X, const Tensor1D& b) {

  int seq = X.size();

  for (int s = 0; s < seq; ++s) {

    for (int i = 0; i < b.size(); ++i) {

      X[s][i] += b[i];

    }

  }

}

Tensor2D MultiHeadAttention::forward(const Tensor2D& x) {

  int seq = x.size();

  Tensor2D Q = matmul(x, transpose(W_q));

  if (use_bias) add_bias(Q, b_q);

  Tensor2D K = matmul(x, transpose(W_k));

  if (use_bias) add_bias(K, b_k);

  Tensor2D V = matmul(x, transpose(W_v));

  if (use_bias) add_bias(V, b_v);

  // Apply RoPE

  for (int s = 0; s < seq; ++s) {

    Q[s] = rope->apply(Q[s], s);

    K[s] = rope->apply(K[s], s);

  }

  // Split to heads

  Tensor3D Q_heads(num_heads, Tensor2D(seq, Tensor1D(head_dim, 0.0f)));

  // Fill Q_heads, K_heads, V_heads

  for (int h = 0; h < num_heads; ++h) {

    for (int s = 0; s < seq; ++s) {

      for (int i = 0; i < head_dim; ++i) {

        Q_heads[h][s][i] = Q[s][h * head_dim + i];

        // Similar for K, V

      }

    }

  }

  // Attention per head

  Tensor3D attn_heads(num_heads, Tensor2D(seq, Tensor1D(head_dim, 0.0f)));

  for (int h = 0; h < num_heads; ++h) {

    attn_heads[h] = Attention::scaled_dot_product(Q_heads[h], K_heads[h], V_heads[h], true); // causal

  }

  // Concat

  Tensor2D attn(seq, Tensor1D(embed_dim, 0.0f));

  for (int s = 0; s < seq; ++s) {

    for (int h = 0; h < num_heads; ++h) {

      for (int i = 0; i < head_dim; ++i) {

        attn[s][h * head_dim + i] = attn_heads[h][s][i];

      }

    }

  }

  Tensor2D out = matmul(attn, transpose(W_o));

  if (use_bias) add_bias(out, b_o);

  // Update cache

  k_cache.resize(seq);

  v_cache.resize(seq);

  for (int s = 0; s < seq; ++s) {

    k_cache[s] = K[s];

    v_cache[s] = V[s];

  }

  cache_len = seq;

  return out;

}

Tensor1D MultiHeadAttention::forward_cached(const Tensor1D& x, int pos) {

  // Similar to forward, but for single token, using cache for K, V

  Tensor1D Q = matvec(transpose(W_q), x);

  if (use_bias) for (int i = 0; i < embed_dim; ++i) Q[i] += b_q[i];

  Tensor1D K = matvec(transpose(W_k), x);

  if (use_bias) for (int i = 0; i < embed_dim; ++i) K[i] += b_k[i];

  Tensor1D V = matvec(transpose(W_v), x);

  if (use_bias) for (int i = 0; i < embed_dim; ++i) V[i] += b_v[i];

  Q = rope->apply(Q, pos);

  K = rope->apply(K, pos);

  k_cache[pos] = K;

  v_cache[pos] = V;

  cache_len = pos + 1;

  // Split Q to heads (single seq)

  Tensor3D Q_heads(num_heads, Tensor2D(1, Tensor1D(head_dim, 0.0f)));

  for (int h = 0; h < num_heads; ++h) {

    for (int i = 0; i < head_dim; ++i) {

      Q_heads[h][0][i] = Q[h * head_dim + i];

    }

  }

  // K_heads, V_heads from cache

  Tensor3D K_heads(num_heads, Tensor2D(cache_len, Tensor1D(head_dim, 0.0f)));

  Tensor3D V_heads(num_heads, Tensor2D(cache_len, Tensor1D(head_dim, 0.0f)));

  for (int h = 0; h < num_heads; ++h) {

    for (int s = 0; s < cache_len; ++s) {

      for (int i = 0; i < head_dim; ++i) {

        K_heads[h][s][i] = k_cache[s][h * head_dim + i];

        V_heads[h][s][i] = v_cache[s][h * head_dim + i];

      }

    }

  }

  // Attention

  Tensor3D attn_heads(num_heads, Tensor2D(1, Tensor1D(head_dim, 0.0f)));

  for (int h = 0; h < num_heads; ++h) {

    attn_heads[h] = Attention::scaled_dot_product(Q_heads[h], K_heads[h], V_heads[h], true);

  }

  // Concat

  Tensor1D attn(embed_dim, 0.0f);

  for (int h = 0; h < num_heads; ++h) {

    for (int i = 0; i < head_dim; ++i) {

      attn[h * head_dim + i] = attn_heads[h][0][i];

    }

  }

  Tensor1D out = matvec(transpose(W_o), attn);

  if (use_bias) for (int i = 0; i < embed_dim; ++i) out[i] += b_o[i];

  return out;

}

void MultiHeadAttention::clear_cache() {

  cache_len = 0;

}

void MultiHeadAttention::save(std::ostream& os) const {

  save_matrix(os, W_q);

  save_matrix(os, W_k);

  save_matrix(os, W_v);

  save_matrix(os, W_o);

  if (use_bias) {

    save_vector(os, b_q);

    save_vector(os, b_k);

    save_vector(os, b_v);

    save_vector(os, b_o);

  }

}

void MultiHeadAttention::load(std::istream& is) {

  load_matrix(is, W_q);

  load_matrix(is, W_k);

  load_matrix(is, W_v);

  load_matrix(is, W_o);

  if (use_bias) {

    load_vector(is, b_q);

    load_vector(is, b_k);

    load_vector(is, b_v);

    load_vector(is, b_o);

  }

  // Reset Adam if needed

}

void MultiHeadAttention::save_matrix(std::ostream& os, const Tensor2D& M) const {

  for (const auto& row : M) {

    for (float v : row) os << v << " ";

    os << "\n";

  }

}

void MultiHeadAttention::load_matrix(std::istream& is, Tensor2D& M) {

  for (auto& row : M) {

    for (auto& v : row) is >> v;

  }

}

void MultiHeadAttention::save_vector(std::ostream& os, const Tensor1D& v) const {

  for (float val : v) os << val << " ";

  os << "\n";

}

void MultiHeadAttention::load_vector(std::istream& is, Tensor1D& v) {

  for (auto& val : v) is >> val;

}

FeedForward::FeedForward(int dim, int hidden) : embed_dim(dim), hidden_dim(hidden), W_gate(hidden, Tensor1D(dim, 0.0f)), W_up(hidden, Tensor1D(dim, 0.0f)), W_down(dim, Tensor1D(hidden, 0.0f)), m_Wgate(hidden, Tensor1D(dim, 0.0f)), v_Wgate(hidden, Tensor1D(dim, 0.0f)), m_Wup(hidden, Tensor1D(dim, 0.0f)), v_Wup(hidden, Tensor1D(dim, 0.0f)), m_Wdown(dim, Tensor1D(hidden, 0.0f)), v_Wdown(dim, Tensor1D(hidden, 0.0f)) {

  init_weight(W_gate, hidden, dim, 1.0f / std::sqrt(static_cast<float>(dim)));

  init_weight(W_up, hidden, dim, 1.0f / std::sqrt(static_cast<float>(dim)));

  init_weight(W_down, dim, hidden, 1.0f / std::sqrt(static_cast<float>(hidden)));

  init_adam(m_Wgate, v_Wgate, hidden, dim);

  init_adam(m_Wup, v_Wup, hidden, dim);

  init_adam(m_Wdown, v_Wdown, dim, hidden);

}

Tensor1D FeedForward::forward(const Tensor1D& x) {

  Tensor1D gate = matvec(W_gate, x);

  for (auto& val : gate) val = val * sigmoid(val); // SiLU

  Tensor1D up = matvec(W_up, x);

  Tensor1D hidden(hidden_dim);

  for (int i = 0; i < hidden_dim; ++i) {

    hidden[i] = gate[i] * up[i];

  }

  Tensor1D out = matvec(W_down, hidden);

  return out;

}

void FeedForward::save(std::ostream& os) const {

  save_matrix(os, W_gate);

  save_matrix(os, W_up);

  save_matrix(os, W_down);

}

void FeedForward::load(std::istream& is) {

  load_matrix(is, W_gate);

  load_matrix(is, W_up);

  load_matrix(is, W_down);

}

TransformerBlock::TransformerBlock(int dim, int heads, int hidden, int max_seq) : embed_dim(dim), num_heads(heads), hidden_dim(hidden), ln1(dim), ln2(dim), attn(dim, heads, max_seq), ffn(dim, hidden) {}

Tensor2D TransformerBlock::forward(const Tensor2D& x) {

  int seq = x.size();

  Tensor2D norm1(seq, Tensor1D(embed_dim));

  for (int s = 0; s < seq; ++s) {

    norm1[s] = ln1.forward(x[s]);

  }

  Tensor2D attn_out = attn.forward(norm1);

  Tensor2D x1(seq, Tensor1D(embed_dim));

  for (int s = 0; s < seq; ++s) {

    for (int i = 0; i < embed_dim; ++i) {

      x1[s][i] = x[s][i] + attn_out[s][i];

    }

  }

  Tensor2D norm2(seq, Tensor1D(embed_dim));

  for (int s = 0; s < seq; ++s) {

    norm2[s] = ln2.forward(x1[s]);

  }

  Tensor2D ff(seq, Tensor1D(embed_dim));

  for (int s = 0; s < seq; ++s) {

    ff[s] = ffn.forward(norm2[s]);

  }

  Tensor2D out(seq, Tensor1D(embed_dim));

  for (int s = 0; s < seq; ++s) {

    for (int i = 0; i < embed_dim; ++i) {

      out[s][i] = x1[s][i] + ff[s][i];

    }

  }

  return out;

}

Tensor1D TransformerBlock::forward_cached(const Tensor1D& x, int pos) {

  Tensor1D norm1 = ln1.forward(x);

  Tensor1D attn_out = attn.forward_cached(norm1, pos);

  Tensor1D x1(embed_dim);

  for (int i = 0; i < embed_dim; ++i) {

    x1[i] = x[i] + attn_out[i];

  }

  Tensor1D norm2 = ln2.forward(x1);

  Tensor1D ff = ffn.forward(norm2);

  Tensor1D out(embed_dim);

  for (int i = 0; i < embed_dim; ++i) {

    out[i] = x1[i] + ff[i];

  }

  return out;

}

void TransformerBlock::clear_cache() {

  attn.clear_cache();

}

void TransformerBlock::save(std::ostream& os) const {

  ln1.save(os);

  ln2.save(os);

  attn.save(os);

  ffn.save(os);

}

void TransformerBlock::load(std::istream& is) {

  ln1.load(is);

  ln2.load(is);

  attn.load(is);

  ffn.load(is);

}

TransformerLLM::TransformerLLM(int vocab, int dim, int n_layers, int heads, int hidden, int max_seq) : vocab_size(vocab), embed_dim(dim), num_layers(n_layers), num_heads(heads), hidden_dim(hidden), max_seq_len(max_seq), token_embed(new Embedding(vocab, dim)), final_norm(new LayerNorm(dim)), lm_head(vocab, Tensor1D(dim, 0.0f)) {

  for (int l = 0; l < num_layers; ++l) {

    layers.push_back(new TransformerBlock(dim, heads, hidden, max_seq));

  }

  // Initialize lm_head

  float scale = 1.0f / std::sqrt(static_cast<float>(dim));

  init_weight(lm_head, vocab, dim, scale);

}

TransformerLLM::~TransformerLLM() {

  delete token_embed;

  delete final_norm;

  for (auto l : layers) delete l;

}

void TransformerLLM::clear_cache() {

  for (auto l : layers) l->clear_cache();

}

Tensor2D TransformerLLM::forward(const std::vector<int>& tokens) {

  Tensor2D emb = token_embed->forward_seq(tokens);

  Tensor2D x = emb;

  for (auto l : layers) {

    x = l->forward(x);

  }

  int seq = x.size();

  Tensor2D norm(seq, Tensor1D(embed_dim));

  for (int s = 0; s < seq; ++s) {

    norm[s] = final_norm->forward(x[s]);

  }

  return norm;

}

int TransformerLLM::generate_next(int token, int pos, float temperature) {

  Tensor1D emb = token_embed->forward(token);

  Tensor1D x = emb;

  for (auto l : layers) {

    x = l->forward_cached(x, pos);

  }

  Tensor1D norm = final_norm->forward(x);

  Tensor1D logits = matvec(lm_head, norm);

  if (temperature == 0.0f) {

    auto max_it = std::max_element(logits.begin(), logits.end());

    return std::distance(logits.begin(), max_it);

  }

  float max_val = *std::max_element(logits.begin(), logits.end());

  float sum = 0.0f;

  for (auto& l : logits) {

    l = std::exp((l - max_val) / temperature);

    sum += l;

  }

  for (auto& l : logits) l /= sum;

  float r = static_cast<float>(std::rand()) / RAND_MAX;

  float cum = 0.0f;

  for (int i = 0; i < vocab_size; ++i) {

    cum += logits[i];

    if (r < cum) return i;

  }

  return vocab_size - 1;

}

std::vector<int> TransformerLLM::generate(const std::vector<int>& prompt, int max_new_tokens, float temperature) {

  clear_cache();

  std::vector<int> gen = prompt;

  // Build cache with prompt

  for (int i = 0; i < prompt.size() - 1; ++i) {

    generate_next(prompt[i], i, 0.0f);

  }

  int next = prompt.back();

  int pos = prompt.size() - 1;

  for (int i = 0; i < max_new_tokens; ++i) {

    next = generate_next(next, pos++, temperature);

    gen.push_back(next);

    // Stop if EOS, assume 0 is EOS

    if (next == 0) break;

  }

  return gen;

}

void TransformerLLM::save(const std::string& filename) const {

  std::ofstream os(filename, std::ios::binary);

  os << vocab_size << " " << embed_dim << " " << num_layers << " " << num_heads << " " << hidden_dim << " " << max_seq_len << "\n";

  token_embed->save(os);

  for (auto l : layers) l->save(os);

  final_norm->save(os);

  save_matrix(os, lm_head);

}

void TransformerLLM::load(const std::string& filename) {

  std::ifstream is(filename, std::ios::binary);

  is >> vocab_size >> embed_dim >> num_layers >> num_heads >> hidden_dim >> max_seq_len;

  token_embed->load(is);

  for (auto l : layers) l->load(is);

  final_norm->load(is);

  load_matrix(is, lm_head);

}

void TransformerLLM::load_gguf(const std::string& filename) {

  // Placeholder for GGUF loading

  throw std::runtime_error("GGUF loading not implemented in this version");

}

} // namespace nn