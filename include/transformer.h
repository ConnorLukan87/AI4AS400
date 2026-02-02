#ifndef TRANSFORMER_H
#define TRANSFORMER_H
#include "layer.h"
#include "types.h"
namespace nn {
    class LayerNorm {
    public:
        int dim;
        Tensor1D gamma;
        Tensor1D beta;
        float epsilon;
        Tensor1D normalized;
        Tensor1D input_cache;
        float mean_cache;
        float var_cache;
        Tensor1D m_gamma;
        Tensor1D v_gamma;
        Tensor1D m_beta;
        Tensor1D v_beta;
        LayerNorm(int d, float eps = 1e-5f);
        Tensor1D forward(const Tensor1D& x);
        Tensor1D backward(const Tensor1D& grad_out);
        void save(std::ostream& os) const;
        void load(std::istream& is);
    };
    
    class Embedding {
    public:
        int vocab_size;
        int embed_dim;
        Tensor2D weights;
        Tensor2D m_weights;
        Tensor2D v_weights;
        Embedding(int vocab, int dim);
        Tensor1D forward(int token_id) const;
        Tensor2D forward_seq(
            const std::vector<int>& tokens) const;
        void save(std::ostream& os) const;
        void load(std::istream& is);
    };

    class RoPE {
    public:
        int dim;
        int max_seq_len;
        Tensor2D cos_cache;
        Tensor2D sin_cache;
        RoPE(int d,
            int max_len = 4096,
            float base = 10000.0f);
        Tensor1D apply(const Tensor1D& x, int pos) const;
    };
    
    class Attention {
    public:
        static Tensor2D scaled_dot_product(
            const Tensor2D& Q,
            const Tensor2D& K,
            const Tensor2D& V,
            bool causal_mask = true);
    };

    class MultiHeadAttention {
    public:
        int embed_dim;
        int num_heads;
        int head_dim;
        Tensor2D W_q;
        Tensor2D W_k;
        Tensor2D W_v;
        Tensor2D W_o;
        bool use_bias;
        Tensor1D b_q;
        Tensor1D b_k;
        Tensor1D b_v;
        Tensor1D b_o;
        RoPE* rope;
        Tensor2D k_cache;
        Tensor2D v_cache;
        int cache_len;
        Tensor2D m_Wq;
        Tensor2D v_Wq;
        Tensor2D m_Wk;
        Tensor2D v_Wk;
        Tensor2D m_Wv;
        Tensor2D v_Wv;
        Tensor2D m_Wo;
        Tensor2D v_Wo;
        MultiHeadAttention(int dim,
                            int heads,
                            int max_seq = 4096,
                            bool bias = false);
        ~MultiHeadAttention();
        void clear_cache();
        Tensor2D forward(const Tensor2D& x);
        Tensor1D forward_cached(const Tensor1D& x,
                                int pos);
        void save(std::ostream& os) const;
        void load(std::istream& is);
    private:
        void init_weight(Tensor2D& W,
                        int rows,
                        int cols,
                        float scale);
        void init_adam(Tensor2D& m,
                        Tensor2D& v,
                        int rows,
                        int cols);
        void add_bias(Tensor2D& X,
                        const Tensor1D& b);
        void save_matrix(std::ostream& os,
                        const Tensor2D& M) const;
        void load_matrix(std::istream& is,
                        Tensor2D& M);
        void save_vector(std::ostream& os,
                        const Tensor1D& v) const;
        void load_vector(std::istream& is,
                        Tensor1D& v);
        MultiHeadAttention(const MultiHeadAttention&) = delete;
        MultiHeadAttention& operator=(const MultiHeadAttention&) = delete;
    };

    class FeedForward {
    public:
        int embed_dim;
        int hidden_dim;
        Tensor2D W_gate;
        Tensor2D W_up;
        Tensor2D W_down;
        Tensor2D m_Wgate;
        Tensor2D v_Wgate;
        Tensor2D m_Wup;
        Tensor2D v_Wup;
        Tensor2D m_Wdown;
        Tensor2D v_Wdown;
        FeedForward(int dim, int hidden);
        Tensor1D forward(const Tensor1D& x);
        void save(std::ostream& os) const;
        void load(std::istream& is);
    private:
        void init_weight(Tensor2D& W,
                        int rows,
                        int cols,
                        float scale);
        void init_adam(Tensor2D& m,
                        Tensor2D& v,
                        int rows,
                        int cols);
        void save_matrix(std::ostream& os,
                        const Tensor2D& M) const;
        void load_matrix(std::istream& is,
                        Tensor2D& M);
    };

    class TransformerBlock {
    public:
        int embed_dim;
        int num_heads;
        int hidden_dim;
        LayerNorm ln1;
        LayerNorm ln2;
        MultiHeadAttention attn;
        FeedForward ffn;
        TransformerBlock(int dim,
                        int heads,
                        int hidden,
                        int max_seq = 4096);
        Tensor2D forward(const Tensor2D& x);
        Tensor1D forward_cached(const Tensor1D& x,
                                int pos);
        void clear_cache();
        void save(std::ostream& os) const;
        void load(std::istream& is);
    private:
        TransformerBlock(const TransformerBlock&) = delete;
        TransformerBlock& operator=(const TransformerBlock&) = delete;
    };

    class TransformerLLM {
    public:
        int vocab_size;
        int embed_dim;
        int num_layers;
        int num_heads;
        int hidden_dim;
        int max_seq_len;
        Embedding* token_embed;
        std::vector<TransformerBlock*> layers;
        LayerNorm* final_norm;
        Tensor2D lm_head;
        TransformerLLM(int vocab,
                        int dim,
                        int n_layers,
                        int heads,
                        int hidden,
                        int max_seq = 4096);
        ~TransformerLLM();
        void clear_cache();
        Tensor2D forward(const std::vector<int>& tokens);
        int generate_next(int token,
                            int pos,
                            float temperature = 1.0f);
        std::vector<int> generate(
            const std::vector<int>& prompt,
            int max_new_tokens,
            float temperature = 1.0f);
        void save(const std::string& filename) const;
        void load(const std::string& filename);
        void load_gguf(const std::string& filename);
    private:
        TransformerLLM(const TransformerLLM&) = delete;
        TransformerLLM& operator=(const TransformerLLM&) = delete;
    };
} // namespace nn
#endif // TRANSFORMER_H
