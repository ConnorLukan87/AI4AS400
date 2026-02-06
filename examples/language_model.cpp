#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <random>

#include "types.h"
#include "layer.h"
#include "network.h"
#include "transformer.h"

using namespace std;
using namespace nn;
namespace fs = std::filesystem;

const int PAD_TOKEN = 0;
const int UNK_TOKEN = 1;

vector<string> tokenize(const string& text) {
    vector<string> tokens;
    stringstream ss(text);
    string token;
    while (ss >> token) {
        transform(token.begin(), token.end(), token.begin(), ::tolower);
        token.erase(remove_if(token.begin(), token.end(), ::ispunct), token.end());
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

vector<int> sequenceize(const string& text, const unordered_map<string, int>& vocab, int max_len) {
    vector<int> seq;
    auto tokens = tokenize(text);
    for (const auto& t : tokens) {
        auto it = vocab.find(t);
        seq.push_back(it != vocab.end() ? it->second : UNK_TOKEN);
    }
    while (seq.size() < max_len) {
        seq.push_back(PAD_TOKEN);
    }
    if (seq.size() > max_len) {
        seq.resize(max_len);
    }
    return seq;
}

vector<string> get_files(const string& dir) {
    vector<string> files;
    for (const auto& p : fs::directory_iterator(dir)) {
        if (p.is_regular_file() && p.path().extension() == ".txt") {
            files.push_back(p.path().string());
        }
    }
    sort(files.begin(), files.end());
    return files;
}

void shuffle_data(vector<vector<int>>& X, vector<Tensor1D>& y) {
    vector<size_t> indices(X.size());
    iota(indices.begin(), indices.end(), 0);
    random_device rd;
    mt19937 g(rd());
    shuffle(indices.begin(), indices.end(), g);
    vector<vector<int>> X_shuf(X.size());
    vector<Tensor1D> y_shuf(y.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        X_shuf[i] = X[indices[i]];
        y_shuf[i] = y[indices[i]];
    }
    X = move(X_shuf);
    y = move(y_shuf);
}

float compute_accuracy(const vector<Tensor1D>& preds, const vector<Tensor1D>& targets) {
    int correct = 0;
    for (size_t i = 0; i < preds.size(); ++i) {
        float pred = preds[i][0] > 0.5f ? 1.0f : 0.0f;
        if (pred == targets[i][0]) ++correct;
    }
    return static_cast<float>(correct) / preds.size();
}

int main() {
    string input_dir = "data/inputs/";
    string output_dir = "data/outputs/";

    auto input_files = get_files(input_dir);
    auto output_files = get_files(output_dir);

    if (input_files.size() != output_files.size()) {
        cerr << "Mismatch in number of input and output files!" << endl;
        return 1;
    }

    size_t num_samples = input_files.size();
    cout << "Found " << num_samples << " samples." << endl;

    unordered_map<string, int> freq;
    int max_len = 0;
    for (size_t i = 0; i < num_samples; ++i) {
        ifstream in(input_files[i]);
        string text((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
        auto tokens = tokenize(text);
        max_len = max(max_len, static_cast<int>(tokens.size()));
        unordered_map<string, int> local_freq;
        for (const auto& t : tokens) {
            local_freq[t]++;
        }
        for (const auto& lf : local_freq) {
            freq[lf.first]++;
        }
    }

    vector<pair<int, string>> freq_list;
    for (const auto& f : freq) {
        freq_list.emplace_back(f.second, f.first);
    }
    sort(freq_list.rbegin(), freq_list.rend());

    int max_vocab = 10000;
    unordered_map<string, int> vocab;
    vocab["<pad>"] = PAD_TOKEN;
    vocab["<unk>"] = UNK_TOKEN;
    for (size_t i = 0; i < min(static_cast<size_t>(max_vocab), freq_list.size()); ++i) {
        vocab[freq_list[i].second] = static_cast<int>(i) + 2;
    }
    int v_size = static_cast<int>(vocab.size());
    cout << "Vocabulary size: " << v_size << endl;
    cout << "Max sequence length: " << max_len << endl;

    vector<vector<int>> X_seq(num_samples);
    vector<Tensor1D> y(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        ifstream in(input_files[i]);
        string text((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
        X_seq[i] = sequenceize(text, vocab, max_len);

        ifstream out(output_files[i]);
        string label_str((istreambuf_iterator<char>(out)), istreambuf_iterator<char>());
        float label = stof(label_str);
        y[i] = {label};
    }

    shuffle_data(X_seq, y);
    size_t train_size = static_cast<size_t>(0.8 * num_samples);
    vector<vector<int>> X_train(X_seq.begin(), X_seq.begin() + train_size);
    vector<Tensor1D> y_train(y.begin(), y.begin() + train_size);
    vector<vector<int>> X_val(X_seq.begin() + train_size, X_seq.end());
    vector<Tensor1D> y_val(y.begin() + train_size, y.end());

    int embed_dim = 128;
    int hidden_size = 256;
    float lr = 0.001f;
    int epochs = 5;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;

    Embedding embedding(v_size, embed_dim);
    LSTMLayer lstm(embed_dim, hidden_size, ActivationType::TANH);
    DenseLayer dense(hidden_size, 1, ActivationType::SIGMOID);

    int t = 1;
    for (int e = 0; e < epochs; ++e) {
        float total_loss = 0.0f;
        for (size_t i = 0; i < train_size; ++i) {
            // Forward
            Tensor2D emb_out = embedding.forward_seq(X_train[i]);
            Tensor2D lstm_out = lstm.forward_seq(emb_out);
            Tensor1D last_hidden = lstm_out.back();
            Tensor1D out = dense.forward1D(last_hidden);

            float loss = - (y_train[i][0] * log(out[0] + 1e-10f) + (1 - y_train[i][0]) * log(1 - out[0] + 1e-10f));
            total_loss += loss;

            Tensor1D grad_out(1, (out[0] - y_train[i][0]) / (out[0] * (1 - out[0]) + 1e-10f));

            dense.compute_deltas1D(grad_out, dense.get_z1D());

            Tensor1D grad_last_hidden = dense.get_grad_input1D();

            Tensor2D grad_lstm_out(lstm_out.size(), Tensor1D(hidden_size, 0.0f));
            grad_lstm_out.back() = grad_last_hidden;
            lstm.compute_deltas_seq(grad_lstm_out, lstm.get_z_seq());

            Tensor2D grad_emb = lstm.get_grad_input_seq();

            Tensor2D grad_weights(v_size, Tensor1D(embed_dim, 0.0f));
            for (size_t s = 0; s < X_train[i].size(); ++s) {
                int tok = X_train[i][s];
                if (tok == PAD_TOKEN) continue;
                for (int d = 0; d < embed_dim; ++d) {
                    grad_weights[tok][d] += grad_emb[s][d];
                }
            }

            dense.update_params(last_hidden, lr, t, beta1, beta2, epsilon);
            lstm.update_params_seq(emb_out, lr, t, beta1, beta2, epsilon);

            auto update = [beta1, beta2, epsilon, lr, t](Tensor2D& W, Tensor2D& m, Tensor2D& v, const Tensor2D& g) {
                for (int j = 0; j < W.size(); ++j) {
                    for (int k = 0; k < W[0].size(); ++k) {
                        float gg = g[j][k];
                        if (gg == 0.0f) continue;  // Sparse
                        m[j][k] = beta1 * m[j][k] + (1 - beta1) * gg;
                        v[j][k] = beta2 * v[j][k] + (1 - beta2) * gg * gg;
                        float m_hat = m[j][k] / (1 - pow(beta1, t));
                        float v_hat = v[j][k] / (1 - pow(beta2, t));
                        W[j][k] -= lr * m_hat / (sqrt(v_hat) + epsilon);
                    }
                }
            };
            update(embedding.weights, embedding.m_weights, embedding.v_weights, grad_weights);

            ++t;
        }
        cout << "Epoch " << e + 1 << " loss: " << total_loss / train_size << endl;
    }

    vector<Tensor1D> preds;
    for (const auto& seq : X_val) {
        Tensor2D emb_out = embedding.forward_seq(seq);
        Tensor2D lstm_out = lstm.forward_seq(emb_out);
        Tensor1D last_hidden = lstm_out.back();
        Tensor1D out = dense.forward1D(last_hidden);
        preds.push_back(out);
    }
    float acc = compute_accuracy(preds, y_val);
    cout << "Validation accuracy: " << acc << endl;

    ofstream os("advanced_model.bin", ios::binary);
    embedding.save(os);
    lstm.save(os);
    dense.save(os);
    os.close();
    cout << "Trained advanced LSTM model and saved to advanced_model.bin" << endl;

    return 0;
}