#ifndef DATASCIENCE_H
#define DATASCIENCE_H

#include "dataframe.h"
#include "network.h"
#include "types.h"
#include <functional>
#include <numeric>
#include <set>
#include <unordered_map>
#include <chrono>

namespace nn {

namespace stats {
    // Descriptive
    float mean(const Tensor1D& x);
    float variance(const Tensor1D& x, bool sample = true);
    float stddev(const Tensor1D& x, bool sample = true);
    float covariance(const Tensor1D& x, const Tensor1D& y);
    float pearson_r(const Tensor1D& x, const Tensor1D& y);
    float spearman_r(const Tensor1D& x, const Tensor1D& y);
    float median(const Tensor1D& x);
    float percentile(const Tensor1D& x, float p);
    float iqr(const Tensor1D& x);
    float skewness(const Tensor1D& x);
    float kurtosis(const Tensor1D& x);
    float entropy(const Tensor1D& probs);
    float mutual_information(const Tensor1D& x, const Tensor1D& y,
                             int bins = 10);
    Tensor1D ranks(const Tensor1D& x);
    Tensor2D correlation_matrix(const Tensor2D& data);
    Tensor2D covariance_matrix(const Tensor2D& data);

    // Hypothesis tests (return {statistic, p_value})
    struct TestResult {
        float statistic;
        float p_value;
        std::string test_name;
        bool reject_null(float alpha = 0.05f) const;
        std::string summary() const;
    };

    TestResult t_test_one_sample(const Tensor1D& x, float mu0);
    TestResult t_test_two_sample(const Tensor1D& x, const Tensor1D& y,
                                  bool equal_var = true);
    TestResult t_test_paired(const Tensor1D& x, const Tensor1D& y);
    TestResult chi_squared_test(const Tensor1D& observed,
                                 const Tensor1D& expected);
    TestResult ks_test(const Tensor1D& x);  // normality
    TestResult shapiro_wilk(const Tensor1D& x);
    TestResult anova_one_way(const std::vector<Tensor1D>& groups);
    TestResult mann_whitney_u(const Tensor1D& x, const Tensor1D& y);

    // Distributions
    float normal_pdf(float x, float mu, float sigma);
    float normal_cdf(float x, float mu = 0.0f, float sigma = 1.0f);
    float normal_quantile(float p, float mu = 0.0f, float sigma = 1.0f);
    float t_cdf(float t, int df);
    float chi2_cdf(float x, int df);
    float f_cdf(float x, int df1, int df2);
    float beta_func(float a, float b);
    float gamma_func(float x);
    float regularized_beta(float x, float a, float b);
    
    // Confidence intervals
    struct ConfInterval {
        float lower;
        float upper;
        float point_estimate;
        float confidence_level;
    };
    ConfInterval confidence_interval_mean(const Tensor1D& x,
                                           float confidence = 0.95f);
    ConfInterval confidence_interval_proportion(int successes, int n,
                                                 float confidence = 0.95f);

    // Bootstrap
    Tensor1D bootstrap_means(const Tensor1D& x, int n_bootstrap = 1000);
    ConfInterval bootstrap_ci(const Tensor1D& x,
                               std::function<float(const Tensor1D&)> statistic,
                               int n_bootstrap = 1000,
                               float confidence = 0.95f);
} // namespace stats

namespace metrics {
    // Regression
    float mse(const Tensor1D& y_true, const Tensor1D& y_pred);
    float rmse(const Tensor1D& y_true, const Tensor1D& y_pred);
    float mae(const Tensor1D& y_true, const Tensor1D& y_pred);
    float mape(const Tensor1D& y_true, const Tensor1D& y_pred);
    float r_squared(const Tensor1D& y_true, const Tensor1D& y_pred);
    float adjusted_r_squared(const Tensor1D& y_true, const Tensor1D& y_pred,
                              int n_features);
    float explained_variance(const Tensor1D& y_true, const Tensor1D& y_pred);
    float max_error(const Tensor1D& y_true, const Tensor1D& y_pred);
    float median_absolute_error(const Tensor1D& y_true, const Tensor1D& y_pred);

    // Classification
    struct ConfusionMatrix {
        int tp, fp, tn, fn;
        int total() const;
        float accuracy() const;
        float precision() const;
        float recall() const;
        float f1() const;
        float specificity() const;
        float npv() const;
        float mcc() const;  // Matthews Correlation Coefficient
        std::string to_string() const;
    };

    ConfusionMatrix confusion_matrix(const Tensor1D& y_true,
                                     const Tensor1D& y_pred,
                                     float threshold = 0.5f);

    float accuracy(const Tensor1D& y_true, const Tensor1D& y_pred,
                   float threshold = 0.5f);
    float precision(const Tensor1D& y_true, const Tensor1D& y_pred,
                    float threshold = 0.5f);
    float recall(const Tensor1D& y_true, const Tensor1D& y_pred,
                 float threshold = 0.5f);
    float f1_score(const Tensor1D& y_true, const Tensor1D& y_pred,
                   float threshold = 0.5f);
    float fbeta_score(const Tensor1D& y_true, const Tensor1D& y_pred,
                      float beta, float threshold = 0.5f);
    float log_loss(const Tensor1D& y_true, const Tensor1D& y_pred);
    float cohen_kappa(const Tensor1D& y_true, const Tensor1D& y_pred,
                      float threshold = 0.5f);

    // ROC / AUC
    struct ROCPoint { float fpr; float tpr; float threshold; };
    std::vector<ROCPoint> roc_curve(const Tensor1D& y_true,
                                     const Tensor1D& y_scores);
    float auc_roc(const Tensor1D& y_true, const Tensor1D& y_scores);

    // Precision-Recall
    struct PRPoint { float precision; float recall; float threshold; };
    std::vector<PRPoint> pr_curve(const Tensor1D& y_true,
                                   const Tensor1D& y_scores);
    float auc_pr(const Tensor1D& y_true, const Tensor1D& y_scores);
    float average_precision(const Tensor1D& y_true,
                            const Tensor1D& y_scores);

    // Multi-class
    struct ClassReport {
        std::vector<int> classes;
        std::vector<float> precisions;
        std::vector<float> recalls;
        std::vector<float> f1s;
        std::vector<int> supports;
        float macro_precision;
        float macro_recall;
        float macro_f1;
        float weighted_f1;
        std::string to_string() const;
    };

    ClassReport classification_report(const std::vector<int>& y_true,
                                       const std::vector<int>& y_pred);
    Tensor2D confusion_matrix_multi(const std::vector<int>& y_true,
                                     const std::vector<int>& y_pred,
                                     int num_classes);

    // Ranking
    float ndcg(const Tensor1D& relevances, int k = -1);
    float mean_reciprocal_rank(const std::vector<Tensor1D>& rankings);
    float precision_at_k(const Tensor1D& y_true,
                          const Tensor1D& y_scores, int k);

    // Clustering
    float silhouette_score(const Tensor2D& X,
                            const std::vector<int>& labels);
    float davies_bouldin_index(const Tensor2D& X,
                                const std::vector<int>& labels);
    float calinski_harabasz_index(const Tensor2D& X,
                                   const std::vector<int>& labels);
} // namespace metrics

namespace preprocess {
    // Scalers (fit + transform pattern)
    struct ScalerParams {
        Tensor1D means;
        Tensor1D stds;
        Tensor1D mins;
        Tensor1D maxs;
        Tensor1D medians;
        Tensor1D q1s;
        Tensor1D q3s;
    };

    ScalerParams fit_standard_scaler(const std::vector<Tensor1D>& X);
    std::vector<Tensor1D> transform_standard(
        const std::vector<Tensor1D>& X, const ScalerParams& params);
    std::vector<Tensor1D> inverse_standard(
        const std::vector<Tensor1D>& X, const ScalerParams& params);

    ScalerParams fit_minmax_scaler(const std::vector<Tensor1D>& X);
    std::vector<Tensor1D> transform_minmax(
        const std::vector<Tensor1D>& X, const ScalerParams& params);
    std::vector<Tensor1D> inverse_minmax(
        const std::vector<Tensor1D>& X, const ScalerParams& params);

    ScalerParams fit_robust_scaler(const std::vector<Tensor1D>& X);
    std::vector<Tensor1D> transform_robust(
        const std::vector<Tensor1D>& X, const ScalerParams& params);

    // Encoding
    struct LabelEncoderState {
        std::map<std::string, int> mapping;
        std::map<int, std::string> inverse_mapping;
    };

    LabelEncoderState fit_label_encoder(
        const std::vector<std::string>& labels);
    std::vector<int> transform_labels(
        const std::vector<std::string>& labels,
        const LabelEncoderState& state);
    std::vector<std::string> inverse_transform_labels(
        const std::vector<int>& encoded,
        const LabelEncoderState& state);

    Tensor2D one_hot_encode(const std::vector<int>& labels,
                             int num_classes = -1);
    std::vector<int> decode_one_hot(const Tensor2D& encoded);

    // Target encoding
    Tensor1D target_encode(const std::vector<std::string>& categories,
                            const Tensor1D& target,
                            float smoothing = 1.0f);

    // Feature engineering
    Tensor2D polynomial_features(const Tensor2D& X, int degree = 2,
                                  bool interaction_only = false);
    Tensor2D add_interaction_terms(const Tensor2D& X);

    // Binning
    Tensor1D equal_width_bins(const Tensor1D& x, int n_bins);
    Tensor1D equal_freq_bins(const Tensor1D& x, int n_bins);
    Tensor1D custom_bins(const Tensor1D& x,
                          const Tensor1D& bin_edges);

    // Imputation
    std::vector<Tensor1D> impute_mean(const std::vector<Tensor1D>& X);
    std::vector<Tensor1D> impute_median(const std::vector<Tensor1D>& X);
    std::vector<Tensor1D> impute_knn(const std::vector<Tensor1D>& X,
                                      int k = 5);

    // Outlier detection
    Tensor1D z_score_outliers(const Tensor1D& x, float threshold = 3.0f);
    Tensor1D iqr_outliers(const Tensor1D& x, float multiplier = 1.5f);
    std::vector<bool> isolation_forest_outliers(
        const std::vector<Tensor1D>& X,
        int n_trees = 100,
        float contamination = 0.1f);

    // Text features (basic)
    Tensor2D tfidf(const std::vector<std::string>& documents,
                    int max_features = 1000);
    Tensor2D bag_of_words(const std::vector<std::string>& documents,
                           int max_features = 1000);

    // Train/test split
    struct SplitResult {
        std::vector<Tensor1D> X_train;
        std::vector<Tensor1D> X_test;
        std::vector<Tensor1D> y_train;
        std::vector<Tensor1D> y_test;
        std::vector<size_t> train_indices;
        std::vector<size_t> test_indices;
    };

    SplitResult train_test_split(const std::vector<Tensor1D>& X,
                                  const std::vector<Tensor1D>& y,
                                  float test_ratio = 0.2f,
                                  bool shuffle = true,
                                  int seed = -1);
    SplitResult stratified_split(const std::vector<Tensor1D>& X,
                                  const std::vector<Tensor1D>& y,
                                  float test_ratio = 0.2f,
                                  int seed = -1);

    // Oversampling / undersampling
    struct ResampleResult {
        std::vector<Tensor1D> X;
        std::vector<Tensor1D> y;
    };

    ResampleResult random_oversample(const std::vector<Tensor1D>& X,
                                      const std::vector<Tensor1D>& y);
    ResampleResult random_undersample(const std::vector<Tensor1D>& X,
                                       const std::vector<Tensor1D>& y);
    ResampleResult smote(const std::vector<Tensor1D>& X,
                          const std::vector<Tensor1D>& y,
                          int k = 5);
} // namespace preprocess

namespace feature_selection {
    // Variance threshold
    std::vector<int> variance_threshold(const std::vector<Tensor1D>& X,
                                         float threshold = 0.0f);

    // Univariate selection
    struct FeatureScore {
        int index;
        float score;
        float p_value;
    };

    std::vector<FeatureScore> f_classif(
        const std::vector<Tensor1D>& X,
        const std::vector<int>& y);
    std::vector<FeatureScore> f_regression(
        const std::vector<Tensor1D>& X,
        const Tensor1D& y);
    std::vector<FeatureScore> mutual_info_classif(
        const std::vector<Tensor1D>& X,
        const std::vector<int>& y,
        int bins = 10);

    // Select K best
    std::vector<int> select_k_best(
        const std::vector<FeatureScore>& scores, int k);

    // Recursive feature elimination (with any model)
    struct RFEResult {
        std::vector<int> selected_features;
        std::vector<int> ranking;
        std::vector<float> scores;
    };

    // Correlation-based
    std::vector<int> remove_correlated(const std::vector<Tensor1D>& X,
                                        float threshold = 0.95f);
} // namespace feature_selection

namespace cv {
    struct Fold {
        std::vector<size_t> train_indices;
        std::vector<size_t> val_indices;
    };

    std::vector<Fold> k_fold(size_t n_samples, int k = 5,
                              bool shuffle = true, int seed = -1);
    std::vector<Fold> stratified_k_fold(const std::vector<int>& labels,
                                         int k = 5, int seed = -1);
    std::vector<Fold> time_series_split(size_t n_samples, int n_splits = 5,
                                         int gap = 0);
    std::vector<Fold> leave_one_out(size_t n_samples);
    std::vector<Fold> group_k_fold(const std::vector<int>& groups,
                                    int k = 5);

    struct CVResult {
        std::vector<float> train_scores;
        std::vector<float> val_scores;
        float mean_train;
        float std_train;
        float mean_val;
        float std_val;
        std::string summary() const;
    };

    // Cross-validate a NeuralNetwork
    CVResult cross_validate_nn(
        const std::vector<Tensor1D>& X,
        const std::vector<Tensor1D>& y,
        int k,
        std::function<NeuralNetwork()> model_factory,
        float lr, int epochs,
        std::function<float(const Tensor1D&, const Tensor1D&)> metric);

    // Cross-validate XGBoost
    CVResult cross_validate_xgb(
        const std::vector<Tensor1D>& X,
        const std::vector<Tensor1D>& y,
        int k,
        std::function<XGBoost*()> model_factory,
        std::function<float(const Tensor1D&, const Tensor1D&)> metric);
} // namespace cv

namespace unsupervised {
    // KMeans
    struct KMeansResult {
        Tensor2D centroids;
        std::vector<int> labels;
        Tensor1D inertia_history;
        int n_iterations;
    };

    KMeansResult kmeans(const std::vector<Tensor1D>& X,
                         int k,
                         int max_iter = 300,
                         int seed = -1,
                         int n_init = 10);

    // KMeans++ initialization
    Tensor2D kmeans_plus_plus_init(const std::vector<Tensor1D>& X,
                                    int k, int seed = -1);

    // Elbow method: inertia for k=1..max_k
    Tensor1D elbow_inertias(const std::vector<Tensor1D>& X,
                             int max_k = 10);

    // DBSCAN
    struct DBSCANResult {
        std::vector<int> labels;  // -1 = noise
        int n_clusters;
        int n_noise;
    };

    DBSCANResult dbscan(const std::vector<Tensor1D>& X,
                         float eps = 0.5f,
                         int min_samples = 5);

    // Hierarchical (agglomerative, single/complete/average linkage)
    struct DendrogramNode {
        int left;
        int right;
        float distance;
        int size;
    };

    std::vector<DendrogramNode> agglomerative(
        const std::vector<Tensor1D>& X,
        const std::string& linkage = "average");
    std::vector<int> cut_dendrogram(
        const std::vector<DendrogramNode>& dendrogram,
        int n_clusters);

    // PCA
    struct PCAResult {
        Tensor2D components;         // principal components (rows)
        Tensor1D explained_variance;
        Tensor1D explained_ratio;
        Tensor1D singular_values;
        Tensor1D mean;
        int n_components;
    };

    PCAResult fit_pca(const std::vector<Tensor1D>& X,
                       int n_components = -1);
    std::vector<Tensor1D> transform_pca(
        const std::vector<Tensor1D>& X,
        const PCAResult& pca);
    std::vector<Tensor1D> inverse_pca(
        const std::vector<Tensor1D>& X_transformed,
        const PCAResult& pca);

    // t-SNE (simplified Barnes-Hut approximation)
    Tensor2D tsne(const std::vector<Tensor1D>& X,
                   int n_components = 2,
                   float perplexity = 30.0f,
                   int max_iter = 1000,
                   float learning_rate = 200.0f);
} // namespace unsupervised

namespace timeseries {
    // Decomposition
    struct DecompResult {
        Tensor1D trend;
        Tensor1D seasonal;
        Tensor1D residual;
        int period;
    };

    DecompResult seasonal_decompose(const Tensor1D& x,
                                     int period,
                                     const std::string& model = "additive");

    // Stationarity
    stats::TestResult adf_test(const Tensor1D& x, int max_lag = -1);

    // Autocorrelation
    Tensor1D autocorrelation(const Tensor1D& x, int max_lag = -1);
    Tensor1D partial_autocorrelation(const Tensor1D& x, int max_lag = -1);

    // Differencing
    Tensor1D difference(const Tensor1D& x, int d = 1);
    Tensor1D seasonal_difference(const Tensor1D& x, int period);

    // Moving averages
    Tensor1D simple_moving_average(const Tensor1D& x, int window);
    Tensor1D exponential_moving_average(const Tensor1D& x, float alpha);
    Tensor1D weighted_moving_average(const Tensor1D& x,
                                      const Tensor1D& weights);

    // Simple forecasting models
    struct ARIMAParams {
        int p, d, q;  // AR, differencing, MA orders
        Tensor1D ar_coeffs;
        Tensor1D ma_coeffs;
        float intercept;
    };

    ARIMAParams fit_arima(const Tensor1D& x, int p, int d, int q);
    Tensor1D predict_arima(const ARIMAParams& params,
                            const Tensor1D& x,
                            int n_ahead);

    // Exponential smoothing
    struct HoltWintersParams {
        float alpha;    // level
        float beta;     // trend
        float gamma;    // seasonal
        int period;
        bool multiplicative;
    };

    HoltWintersParams fit_holt_winters(const Tensor1D& x,
                                        int period,
                                        bool multiplicative = false);
    Tensor1D predict_holt_winters(const HoltWintersParams& params,
                                   const Tensor1D& x,
                                   int n_ahead);

    // Change point detection
    struct ChangePoint {
        int index;
        float score;
    };

    std::vector<ChangePoint> detect_change_points(
        const Tensor1D& x,
        int min_size = 10,
        float penalty = 1.0f);

    // Time series features extraction
    struct TSFeatures {
        float mean, std, min, max, median;
        float skewness, kurtosis;
        float autocorr_lag1, autocorr_lag5;
        float trend_strength, seasonal_strength;
        float entropy;
        int n_crossings;  // zero crossings
        float longest_streak_above_mean;
    };

    TSFeatures extract_features(const Tensor1D& x, int period = -1);
} // namespace timeseries

namespace linear_models {
    struct LinearResult {
        Tensor1D coefficients;
        float intercept;
        float r_squared;
        float adj_r_squared;
        Tensor1D std_errors;
        Tensor1D t_values;
        Tensor1D p_values;
        float f_statistic;
        float f_p_value;
        float aic;
        float bic;
        float durbin_watson;
        int n_obs;
        int n_features;
        std::string summary() const;
    };

    LinearResult ols_regression(const std::vector<Tensor1D>& X,
                                 const Tensor1D& y);

    struct RidgeResult {
        Tensor1D coefficients;
        float intercept;
        float alpha;
    };

    RidgeResult ridge_regression(const std::vector<Tensor1D>& X,
                                  const Tensor1D& y,
                                  float alpha = 1.0f);

    struct LassoResult {
        Tensor1D coefficients;
        float intercept;
        float alpha;
        int n_iterations;
    };

    LassoResult lasso_regression(const std::vector<Tensor1D>& X,
                                  const Tensor1D& y,
                                  float alpha = 1.0f,
                                  int max_iter = 1000,
                                  float tol = 1e-4f);

    struct ElasticNetResult {
        Tensor1D coefficients;
        float intercept;
        float alpha;
        float l1_ratio;
    };

    ElasticNetResult elastic_net(const std::vector<Tensor1D>& X,
                                  const Tensor1D& y,
                                  float alpha = 1.0f,
                                  float l1_ratio = 0.5f,
                                  int max_iter = 1000);

    // Logistic regression
    struct LogisticResult {
        Tensor1D coefficients;
        float intercept;
        int n_iterations;
        Tensor1D predict_proba(const Tensor1D& x) const;
        int predict(const Tensor1D& x) const;
    };

    LogisticResult logistic_regression(const std::vector<Tensor1D>& X,
                                        const std::vector<int>& y,
                                        float lr = 0.01f,
                                        int max_iter = 1000,
                                        float reg = 0.0f);
} // namespace linear_models

namespace knn {
    struct KNNModel {
        std::vector<Tensor1D> X_train;
        std::vector<int> y_class;
        Tensor1D y_reg;
        int k;
        bool is_classification;
    };

    KNNModel fit_classifier(const std::vector<Tensor1D>& X,
                             const std::vector<int>& y, int k = 5);
    int predict_class(const KNNModel& model, const Tensor1D& x);
    std::vector<int> predict_class_batch(const KNNModel& model,
                                          const std::vector<Tensor1D>& X);

    KNNModel fit_regressor(const std::vector<Tensor1D>& X,
                            const Tensor1D& y, int k = 5);
    float predict_reg(const KNNModel& model, const Tensor1D& x);
    Tensor1D predict_reg_batch(const KNNModel& model,
                                const std::vector<Tensor1D>& X);
} // namespace knn

namespace naive_bayes {
    struct GaussianNBModel {
        int n_classes;
        Tensor1D class_priors;          // log priors
        Tensor2D class_means;           // [class][feature]
        Tensor2D class_vars;            // [class][feature]
    };

    GaussianNBModel fit(const std::vector<Tensor1D>& X,
                         const std::vector<int>& y);
    int predict(const GaussianNBModel& model, const Tensor1D& x);
    Tensor1D predict_proba(const GaussianNBModel& model, const Tensor1D& x);
    std::vector<int> predict_batch(const GaussianNBModel& model,
                                    const std::vector<Tensor1D>& X);
} // namespace naive_bayes

class Pipeline {
public:
    enum class StepType {
        STANDARD_SCALE,
        MINMAX_SCALE,
        ROBUST_SCALE,
        PCA,
        POLY_FEATURES,
        IMPUTE_MEAN,
        IMPUTE_MEDIAN,
        CUSTOM
    };

    struct Step {
        StepType type;
        std::string name;
        preprocess::ScalerParams scaler_params;
        unsupervised::PCAResult pca_params;
        int poly_degree;
        std::function<std::vector<Tensor1D>(const std::vector<Tensor1D>&)> custom_fn;
        bool is_fitted;
    };

    Pipeline();

    Pipeline& add_standard_scaler(const std::string& name = "standard_scaler");
    Pipeline& add_minmax_scaler(const std::string& name = "minmax_scaler");
    Pipeline& add_robust_scaler(const std::string& name = "robust_scaler");
    Pipeline& add_pca(int n_components, const std::string& name = "pca");
    Pipeline& add_poly_features(int degree, const std::string& name = "poly");
    Pipeline& add_impute_mean(const std::string& name = "impute_mean");
    Pipeline& add_impute_median(const std::string& name = "impute_median");
    Pipeline& add_custom(
        const std::string& name,
        std::function<std::vector<Tensor1D>(const std::vector<Tensor1D>&)> fn);

    void fit(const std::vector<Tensor1D>& X);
    std::vector<Tensor1D> transform(const std::vector<Tensor1D>& X) const;
    std::vector<Tensor1D> fit_transform(const std::vector<Tensor1D>& X);

    void save(const std::string& filename) const;
    void load(const std::string& filename);

private:
    std::vector<Step> steps;
};

namespace hyperopt {
    struct ParamRange {
        std::string name;
        float min_val;
        float max_val;
        bool is_int;
        std::vector<float> choices;
    };

    struct TrialResult {
        std::map<std::string, float> params;
        float score;
        float train_score;
        float duration_seconds;
    };

    struct SearchResult {
        std::vector<TrialResult> trials;
        TrialResult best;
        std::string summary() const;
    };

    SearchResult grid_search(
        const std::vector<ParamRange>& param_ranges,
        std::function<float(const std::map<std::string, float>&)> objective,
        bool maximize = false);

    SearchResult random_search(
        const std::vector<ParamRange>& param_ranges,
        std::function<float(const std::map<std::string, float>&)> objective,
        int n_trials = 50,
        bool maximize = false,
        int seed = -1);

    SearchResult bayesian_search(
        const std::vector<ParamRange>& param_ranges,
        std::function<float(const std::map<std::string, float>&)> objective,
        int n_trials = 50,
        bool maximize = false,
        int seed = -1);
} // namespace hyperopt

namespace reporting {
    struct TrainingLog {
        std::vector<float> train_losses;
        std::vector<float> val_losses;
        std::vector<float> train_metrics;
        std::vector<float> val_metrics;
        std::vector<float> learning_rates;
        std::string metric_name;
        float best_val_loss;
        int best_epoch;
        float total_time_seconds;

        void add_epoch(float train_loss, float val_loss,
                       float train_metric = 0.0f,
                       float val_metric = 0.0f,
                       float lr = 0.0f);
        std::string summary() const;
        std::string to_csv() const;
        void save(const std::string& filename) const;
    };

    struct ModelReport {
        std::string model_name;
        std::string model_type;
        int n_parameters;
        std::map<std::string, float> hyperparams;
        std::map<std::string, float> train_metrics;
        std::map<std::string, float> val_metrics;
        std::map<std::string, float> test_metrics;
        TrainingLog log;
        std::string timestamp;
        float training_time_seconds;

        std::string summary() const;
        std::string to_csv_row() const;
        void save(const std::string& filename) const;
    };

    // Comparison report
    struct ComparisonReport {
        std::vector<ModelReport> models;
        std::string best_model_name;
        std::string comparison_metric;

        void add_model(const ModelReport& report);
        std::string summary() const;
        std::string to_csv() const;
        void save(const std::string& filename) const;
    };

    // Data quality report
    struct DataQualityReport {
        int n_rows;
        int n_cols;
        std::vector<std::string> column_names;
        std::vector<DType> column_types;
        std::vector<int> null_counts;
        std::vector<float> null_percentages;
        std::vector<int> unique_counts;
        std::vector<float> means;
        std::vector<float> stds;
        std::vector<float> mins;
        std::vector<float> maxs;
        std::vector<float> skewnesses;
        std::vector<float> kurtoses;
        int duplicate_rows;
        float memory_usage_bytes;

        std::string summary() const;
        void save(const std::string& filename) const;
    };

    DataQualityReport generate_data_quality_report(const DataFrame& df);

    // Feature importance
    struct FeatureImportance {
        std::vector<std::string> feature_names;
        Tensor1D importances;

        std::string to_string() const;
        std::string to_csv() const;
    };

    FeatureImportance permutation_importance(
        const std::vector<Tensor1D>& X,
        const Tensor1D& y,
        std::function<Tensor1D(const Tensor1D&)> predict_fn,
        std::function<float(const Tensor1D&, const Tensor1D&)> metric,
        const std::vector<std::string>& feature_names,
        int n_repeats = 10);

    // Learning curve
    struct LearningCurvePoint {
        int n_samples;
        float train_score;
        float val_score;
    };

    std::vector<LearningCurvePoint> learning_curve(
        const std::vector<Tensor1D>& X,
        const std::vector<Tensor1D>& y,
        std::function<void(const std::vector<Tensor1D>&,
                           const std::vector<Tensor1D>&)> fit_fn,
        std::function<Tensor1D(const Tensor1D&)> predict_fn,
        std::function<float(const Tensor1D&, const Tensor1D&)> metric,
        int n_points = 10,
        int cv_folds = 3);

    // Residual analysis
    struct ResidualAnalysis {
        Tensor1D residuals;
        float mean_residual;
        float std_residual;
        float durbin_watson;
        stats::TestResult normality_test;
        bool heteroscedastic;
        std::string summary() const;
    };

    ResidualAnalysis analyze_residuals(const Tensor1D& y_true,
                                       const Tensor1D& y_pred);
} // namespace reporting

namespace distance {
    float euclidean(const Tensor1D& a, const Tensor1D& b);
    float manhattan(const Tensor1D& a, const Tensor1D& b);
    float cosine_similarity(const Tensor1D& a, const Tensor1D& b);
    float cosine_distance(const Tensor1D& a, const Tensor1D& b);
    float minkowski(const Tensor1D& a, const Tensor1D& b, float p);
    float chebyshev(const Tensor1D& a, const Tensor1D& b);
    float hamming(const Tensor1D& a, const Tensor1D& b);
    float jaccard(const Tensor1D& a, const Tensor1D& b);
    float mahalanobis(const Tensor1D& a, const Tensor1D& b,
                       const Tensor2D& cov_inv);
    Tensor2D pairwise_distances(const std::vector<Tensor1D>& X,
                                 const std::string& metric = "euclidean");
    Tensor2D cosine_similarity_matrix(const Tensor2D& X);
} // namespace distance

namespace math_util {
    Tensor1D softmax(const Tensor1D& x);
    Tensor1D log_softmax(const Tensor1D& x);
    Tensor1D linspace(float start, float end, int n);
    Tensor1D arange(float start, float end, float step = 1.0f);
    Tensor1D zeros(int n);
    Tensor1D ones(int n);
    Tensor1D random_normal(int n, float mean = 0.0f, float std = 1.0f);
    Tensor1D random_uniform(int n, float lo = 0.0f, float hi = 1.0f);
    Tensor2D eye(int n);
    Tensor2D transpose(const Tensor2D& A);
    Tensor1D dot(const Tensor2D& A, const Tensor1D& x);
    float dot(const Tensor1D& a, const Tensor1D& b);
    float norm(const Tensor1D& x, float p = 2.0f);
    Tensor1D elementwise_add(const Tensor1D& a, const Tensor1D& b);
    Tensor1D elementwise_mul(const Tensor1D& a, const Tensor1D& b);
    Tensor1D scalar_mul(const Tensor1D& a, float s);
    Tensor2D matrix_add(const Tensor2D& A, const Tensor2D& B);

    // Linear algebra
    struct LUDecomp {
        Tensor2D L;
        Tensor2D U;
        std::vector<int> pivot;
    };

    LUDecomp lu_decomposition(const Tensor2D& A);
    Tensor1D solve(const Tensor2D& A, const Tensor1D& b);
    Tensor2D inverse(const Tensor2D& A);
    float determinant(const Tensor2D& A);

    struct EigenResult {
        Tensor1D eigenvalues;
        Tensor2D eigenvectors;
    };

    EigenResult eigen(const Tensor2D& A, int max_iter = 100);

    struct SVDResult {
        Tensor2D U;
        Tensor1D S;
        Tensor2D Vt;
    };

    SVDResult svd(const Tensor2D& A, int max_iter = 100);
} // namespace math_util

} // namespace nn
#endif // DATASCIENCE_H
