#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "types.h"
namespace nn {
    class NeuralNetwork {
    public:
        std::vector<DenseLayer> layers;
        LossType loss_type;
        LossFunc custom_loss;
        LossDerivFunc custom_loss_deriv;
        NeuralNetwork();
        void set_loss(LossType type);
        void set_custom_loss(LossFunc loss,
                            LossDerivFunc deriv);
        void add_layer(int in_size,
                        int out_size,
                        ActivationType act);
        Tensor1D predict(const Tensor1D& inputs);
        float compute_loss(const Tensor1D& output,
                            const Tensor1D& target);
        Tensor1D compute_loss_deriv(const Tensor1D& output,
                                    const Tensor1D& target);
        void train(const std::vector<Tensor1D>& X,
                    const std::vector<Tensor1D>& y,
                    float lr,
                    int epochs,
                    float beta1 = 0.9f,
                    float beta2 = 0.999f,
                    float epsilon = 1e-8f);
        void save(const std::string& filename) const;
        void load(const std::string& filename);
    };
    class CNN {
    public:
        std::vector<Layer*> layers;
        std::vector<LayerType> layer_types;
        LossType loss_type;
        LossFunc custom_loss;
        LossDerivFunc custom_loss_deriv;
        CNN();
        ~CNN();
        void set_loss(LossType type);
        void set_custom_loss(LossFunc loss,
                            LossDerivFunc deriv);
        void add_conv_layer(int in_ch,
                            int out_ch,
                            int k_size,
                            int str,
                            ActivationType act);
        void add_flatten_layer();
        void add_dense_layer(int in_size,
                            int out_size,
                            ActivationType act);
        void add_rnn_layer(int in_size,
                            int hid_size,
                            ActivationType act);
        Tensor1D predict(const Tensor3D& input);
        float calculate_loss(const Tensor1D& output,
                            const Tensor1D& target);
        Tensor1D calculate_loss_deriv(
            const Tensor1D& output,
            const Tensor1D& target);
        void train(const std::vector<Tensor3D>& X,
                    const std::vector<Tensor1D>& y,
                    float lr,
                    int epochs,
                    float beta1 = 0.9f,
                    float beta2 = 0.999f,
                    float epsilon = 1e-8f);
        void save(const std::string& filename) const;
        void load(const std::string& filename);
    private:
        void clear_layers();
        CNN(const CNN&) = delete;
        CNN& operator=(const CNN&) = delete;
    };
    
    class YOLO : public CNN {
    public:
        int grid_size;
        int num_boxes;
        int num_classes;
        std::vector<Tensor1D> anchors;
        YOLO(int gs,
            int nb,
            int nc,
            const std::vector<Tensor1D>& anch);
        float yolo_loss(const Tensor1D& output,
                        const Tensor1D& target);
        std::vector<std::vector<float>> non_max_suppression(
            const Tensor1D& pred,
            float iou_thres,
            float conf_thres);
        std::vector<std::vector<float>> predict_objects(
            const Tensor3D& input,
            float conf_thres = 0.5f,
            float iou_thres = 0.4f);
    };
    
    struct Node {
        int feature = -1;
        float threshold = 0.0f;
        float value = 0.0f;
        Node* left = nullptr;
        Node* right = nullptr;
        bool is_leaf = false;
        ~Node() {
            delete left;
            delete right;
        }
    };

    class Tree {
    public:
        Tree(int md, float l, float g);
        ~Tree();
        void fit(const std::vector<Tensor1D>& X,
                const Tensor1D& gradients,
                const Tensor1D& hessians);
        float predict(const Tensor1D& x) const;
        void save(std::ostream& os) const;
        void load(std::istream& is);
        private:
        int max_depth;
        float lambda;
        float gamma;
        Node* root;
        Node* build(const std::vector<Tensor1D>& X,
                    const Tensor1D& gradients,
                    const Tensor1D& hessians,
                    const std::vector<int>& indices,
                    int depth);
        float predict_node(Node* node,
                            const Tensor1D& x) const;
        void save_node(std::ostream& os,
                        Node* node) const;
        Node* load_node(std::istream& is);
        Tree(const Tree&) = delete;
        Tree& operator=(const Tree&) = delete;
    };

    class XGBoost {
    public:
        std::vector<Tree*> trees;
        float learning_rate = 0.3f;
        int num_boost_round = 100;
        int max_depth = 6;
        float lambda = 1.0f;
        float gamma = 0.0f;
        float subsample = 1.0f;
        LossType loss_type = LossType(
            LossType::Type::MSE);
        LossFunc custom_loss = nullptr;
        LossDerivFunc custom_loss_deriv = nullptr;
        XGBoost();
        ~XGBoost();
        void set_loss(LossType type);
        void set_custom_loss(LossFunc loss,
                            LossDerivFunc deriv);
        void train(const std::vector<Tensor1D>& X,
                    const std::vector<Tensor1D>& y);
        Tensor1D predict(const Tensor1D& x) const;
        float compute_loss(const Tensor1D& output,
                            const Tensor1D& target) const;
        void save(const std::string& filename) const;
        void load(const std::string& filename);
    private:
        void clear_trees();
        XGBoost(const XGBoost&) = delete;
        XGBoost& operator=(const XGBoost&) = delete;
    };
} // namespace nn
#endif // NETWORK_H
