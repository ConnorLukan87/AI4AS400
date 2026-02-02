#ifndef LAYER_H
#define LAYER_H
#include "types.h"
namespace nn {
    class Layer {
    public:
        ActivationType activation_type;
        ActivationFunc act;
        DerivativeFunc deriv;
        explicit Layer(ActivationType act_type);
        virtual ~Layer() = default;
        virtual Tensor1D forward1D(const Tensor1D& input);
        virtual Tensor2D forward_seq(const Tensor2D& input);
        virtual Tensor3D forward3D(const Tensor3D& input);
        virtual Tensor1D flatten_forward(
            const Tensor3D& input);
        virtual void compute_deltas1D(
            const Tensor1D& grad_output,
            const Tensor1D& z_val);
        virtual void compute_deltas_seq(
            const Tensor2D& grad_output,
            const Tensor2D& z_val);
        virtual void compute_deltas3D(
            const Tensor3D& grad_output,
            const Tensor3D& z_val);
        virtual Tensor1D get_grad_input1D();
        virtual Tensor2D get_grad_input_seq();
        virtual Tensor3D get_grad_input3D();
        virtual Tensor1D get_outputs1D() const;
        virtual Tensor2D get_outputs_seq() const;
        virtual Tensor3D get_outputs3D() const;
        virtual Tensor1D get_z1D() const;
        virtual Tensor2D get_z_seq() const;
        virtual Tensor3D get_z3D() const;
        virtual Tensor1D get_deltas1D() const;
        virtual Tensor2D get_deltas_seq() const;
        virtual Tensor3D get_deltas3D() const;
        virtual void update_params(
            const Tensor1D& prev_outputs,
            float lr,
            int t,
            float beta1,
            float beta2,
            float epsilon);
        virtual void update_params_seq(
            const Tensor2D& prev_outputs,
            float lr,
            int t,
            float beta1,
            float beta2,
            float epsilon);
        virtual void update_params3D(
            const Tensor3D& prev_outputs,
            float lr,
            int t,
            float beta1,
            float beta2,
            float epsilon);
        virtual void save(std::ostream& os) const = 0;
        virtual void load(std::istream& is) = 0;
        virtual LayerType get_layer_type() const = 0;
    };

    class DenseLayer : public Layer {
    public:
        int input_size;
        int output_size;
        Tensor2D weights;
        Tensor1D biases;
        Tensor1D outputs;
        Tensor1D deltas;
        Tensor1D z;
        Tensor2D m_weights;
        Tensor2D v_weights;
        Tensor1D m_biases;
        Tensor1D v_biases;
        DenseLayer(int in,
                    int out,
                    ActivationType act_type);
        Tensor1D forward1D(const Tensor1D& input) override;
        void compute_deltas1D(const Tensor1D& grad_output,
                                const Tensor1D& z_val) override;
        Tensor1D get_grad_input1D() override;
        void update_params(const Tensor1D& prev_outputs,
                            float lr,
                            int t,
                            float beta1,
                            float beta2,
                            float epsilon) override;
        Tensor1D get_outputs1D() const override;
        Tensor1D get_z1D() const override;
        Tensor1D get_deltas1D() const override;
        void save(std::ostream& os) const override;
        void load(std::istream& is) override;
        LayerType get_layer_type() const override;
    };

    class ConvLayer : public Layer {
    public:
        int in_channels;
        int out_channels;
        int kernel_size;
        int stride;
        Tensor4D weights;
        Tensor1D biases;
        Tensor3D outputs;
        Tensor3D deltas;
        Tensor3D z;
        Tensor4D m_weights;
        Tensor4D v_weights;
        Tensor1D m_biases;
        Tensor1D v_biases;
        ConvLayer(int in_ch,
                    int out_ch,
                    int k_size,
                    int str,
                    ActivationType act_type);
        Tensor3D forward3D(const Tensor3D& input) override;
        void compute_deltas3D(const Tensor3D& grad_output,
                                const Tensor3D& z_val) override;
        Tensor3D get_grad_input3D() override;
        void update_params3D(const Tensor3D& prev_input,
                            float lr,
                            int t,
                            float beta1,
                            float beta2,
                            float epsilon) override;
        Tensor3D get_outputs3D() const override;
        Tensor3D get_z3D() const override;
        Tensor3D get_deltas3D() const override;
        void save(std::ostream& os) const override;
        void load(std::istream& is) override;
        LayerType get_layer_type() const override;
    };

    class FlattenLayer : public Layer {
    public:
        Tensor1D outputs;
        Tensor1D deltas;
        int input_channels;
        int input_h;
        int input_w;
        FlattenLayer();
        Tensor1D flatten_forward(const Tensor3D& input) override;
        void compute_deltas1D(const Tensor1D& grad_output,
                                const Tensor1D& z_val) override;
        Tensor3D get_grad_input3D() override;
        Tensor1D get_outputs1D() const override;
        Tensor1D get_deltas1D() const override;
        void save(std::ostream& os) const override;
        void load(std::istream& is) override;
        LayerType get_layer_type() const override;
    };

    class RNNLayer : public Layer {
    public:
        int input_size;
        int hidden_size;
        Tensor2D Wxh;
        Tensor2D Whh;
        Tensor1D bh;
        Tensor2D outputs;
        Tensor2D deltas;
        Tensor2D z;
        Tensor2D m_Wxh;
        Tensor2D v_Wxh;
        Tensor2D m_Whh;
        Tensor2D v_Whh;
        Tensor1D m_bh;
        Tensor1D v_bh;
        RNNLayer(int in, int hid, ActivationType act_type);
        Tensor2D forward_seq(const Tensor2D& inputs) override;
        void compute_deltas_seq(
            const Tensor2D& grad_output,
            const Tensor2D& z_val) override;
        Tensor2D get_grad_input_seq() override;
        void update_params_seq(
            const Tensor2D& inputs,
            float lr,
            int t,
            float beta1,
            float beta2,
            float epsilon) override;
        Tensor2D get_outputs_seq() const override;
        Tensor2D get_z_seq() const override;
        Tensor2D get_deltas_seq() const override;
        void save(std::ostream& os) const override;
        void load(std::istream& is) override;
        LayerType get_layer_type() const override;
    };

    class LSTMLayer : public Layer {
    public:
        int input_size;
        int hidden_size;
        Tensor2D Wf, Wi, Wc, Wo;
        Tensor2D Uf, Ui, Uc, Uo;
        Tensor1D bf, bi, bc, bo;
        Tensor2D outputs;
        Tensor2D cells;
        Tensor2D deltas_h;
        Tensor2D deltas_c;
        Tensor2D z_f, z_i, z_c, z_o;
        Tensor2D m_Wf, v_Wf, m_Uf, v_Uf;
        Tensor2D m_Wi, v_Wi, m_Ui, v_Ui;
        Tensor2D m_Wc, v_Wc, m_Uc, v_Uc;
        Tensor2D m_Wo, v_Wo, m_Uo, v_Uo;
        Tensor1D m_bf, v_bf, m_bi, v_bi;
        Tensor1D m_bc, v_bc, m_bo, v_bo;
        LSTMLayer(int in, int hid, ActivationType act_type);
        Tensor2D forward_seq(const Tensor2D& inputs) override;
        void compute_deltas_seq(
            const Tensor2D& grad_output,
            const Tensor2D& /*z_val*/) override;
        Tensor2D get_grad_input_seq() override;
        void update_params_seq(
            const Tensor2D& inputs,
            float lr,
            int t,
            float beta1,
            float beta2,
            float epsilon) override;
        Tensor2D get_outputs_seq() const override;
        Tensor2D get_z_seq() const override;
        Tensor2D get_deltas_seq() const override;
        void save(std::ostream& os) const override;
        void load(std::istream& is) override;
        LayerType get_layer_type() const override;
    };
    
    Tensor2D matmul(const Tensor2D& A, const Tensor2D& B);
    Tensor1D matvec(const Tensor2D& A, const Tensor1D& x);
} // namespace nn
#endif // LAYER_H
