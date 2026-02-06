#ifndef DATAFRAME_H
#define DATAFRAME_H

#include "types.h"
#include <functional>
#include <numeric>
#include <set>
#include <unordered_map>
#include <iomanip>
#include <regex>

namespace nn {

enum class DType { FLOAT, INT, STRING };

class Column {
public:
    std::string name;
    DType dtype;
    std::vector<float> fdata;
    std::vector<int> idata;
    std::vector<std::string> sdata;
    std::vector<bool> null_mask;  // true = missing

    Column();
    explicit Column(const std::string& name, DType dt = DType::FLOAT);
    Column(const std::string& name, const Tensor1D& data);
    Column(const std::string& name, const std::vector<int>& data);
    Column(const std::string& name, const std::vector<std::string>& data);

    size_t size() const;
    bool is_null(size_t i) const;
    void set_null(size_t i);
    float get_float(size_t i) const;
    int get_int(size_t i) const;
    std::string get_string(size_t i) const;
    void push_float(float v, bool null = false);
    void push_int(int v, bool null = false);
    void push_string(const std::string& v, bool null = false);

    // Numeric ops (FLOAT columns)
    float sum() const;
    float mean() const;
    float variance() const;
    float stddev() const;
    float min_val() const;
    float max_val() const;
    float median() const;
    float percentile(float p) const;
    float skewness() const;
    float kurtosis() const;
    int count_nulls() const;
    int count_valid() const;

    // Fill missing values
    Column fill_null(float value) const;
    Column fill_null_mean() const;
    Column fill_null_median() const;
    Column fill_forward() const;
    Column fill_backward() const;

    // Transformations
    Column apply(std::function<float(float)> fn) const;
    Column log_transform() const;
    Column sqrt_transform() const;
    Column normalize() const;        // min-max [0,1]
    Column standardize() const;      // z-score
    Column robust_scale() const;     // IQR-based
    Column clip(float lo, float hi) const;
    Column abs_col() const;
    Column diff(int periods = 1) const;
    Column pct_change(int periods = 1) const;
    Column cumsum() const;
    Column cumprod() const;
    Column rolling_mean(int window) const;
    Column rolling_std(int window) const;
    Column rolling_min(int window) const;
    Column rolling_max(int window) const;
    Column ewm_mean(float alpha) const;
    Column shift(int periods) const;
    Column rank() const;

    // String ops (STRING columns)
    Column str_lower() const;
    Column str_upper() const;
    Column str_len() const;  // returns FLOAT column
    Column str_contains(const std::string& substr) const; // returns FLOAT (0/1)
    Column str_replace(const std::string& from, const std::string& to) const;

    // Encoding
    Column label_encode() const;
    std::vector<Column> one_hot_encode() const;

    // Sort helpers
    std::vector<size_t> argsort(bool ascending = true) const;

    // Conversion
    Tensor1D to_tensor() const;
    Column as_float() const;
};

class DataFrame {
public:
    std::vector<Column> columns;
    std::unordered_map<std::string, size_t> col_index;

    DataFrame();
    DataFrame(const std::vector<Column>& cols);

    // Shape
    size_t nrows() const;
    size_t ncols() const;
    std::vector<std::string> column_names() const;
    std::vector<DType> column_types() const;

    // Column access
    Column& operator[](const std::string& name);
    const Column& operator[](const std::string& name) const;
    Column& col(size_t i);
    const Column& col(size_t i) const;
    bool has_column(const std::string& name) const;

    // Add / remove columns
    void add_column(const Column& c);
    void add_column(const std::string& name, const Tensor1D& data);
    void add_column(const std::string& name, const std::vector<int>& data);
    void add_column(const std::string& name, const std::vector<std::string>& data);
    void drop_column(const std::string& name);
    void rename_column(const std::string& old_name, const std::string& new_name);

    // Row ops
    void add_row(const std::vector<float>& row);
    DataFrame head(size_t n = 5) const;
    DataFrame tail(size_t n = 5) const;
    DataFrame sample(size_t n) const;
    DataFrame slice(size_t start, size_t end) const;

    // Selection / filtering
    DataFrame select(const std::vector<std::string>& cols) const;
    DataFrame filter(const std::string& col_name,
                     std::function<bool(float)> pred) const;
    DataFrame filter_str(const std::string& col_name,
                         std::function<bool(const std::string&)> pred) const;
    DataFrame drop_nulls(const std::string& col_name = "") const;
    DataFrame drop_duplicates(const std::vector<std::string>& subset = {}) const;

    // Sort
    DataFrame sort_by(const std::string& col_name, bool ascending = true) const;
    DataFrame sort_by(const std::vector<std::string>& cols,
                      const std::vector<bool>& ascending) const;

    // GroupBy aggregation
    struct GroupResult {
        std::vector<std::string> group_keys;
        std::vector<DataFrame> groups;
    };
    GroupResult group_by(const std::string& col_name) const;
    DataFrame group_agg(const std::string& group_col,
                        const std::string& value_col,
                        const std::string& agg_func) const;

    // Pivot / melt
    DataFrame pivot(const std::string& index_col,
                    const std::string& columns_col,
                    const std::string& values_col) const;

    // Join
    DataFrame merge(const DataFrame& other,
                    const std::string& on,
                    const std::string& how = "inner") const;

    // Descriptive stats
    DataFrame describe() const;
    DataFrame corr() const;
    DataFrame cov() const;
    DataFrame value_counts(const std::string& col_name) const;

    // Missing data report
    DataFrame null_report() const;

    // Apply
    DataFrame apply_col(const std::string& col_name,
                        std::function<float(float)> fn) const;
    DataFrame add_derived(const std::string& new_name,
                          std::function<float(const DataFrame&, size_t)> fn) const;

    // Conversion
    std::vector<Tensor1D> to_tensors(
        const std::vector<std::string>& feature_cols) const;
    Tensor2D to_matrix(
        const std::vector<std::string>& feature_cols) const;
    static DataFrame from_tensors(
        const std::vector<Tensor1D>& data,
        const std::vector<std::string>& col_names);

    // I/O
    static DataFrame read_csv(const std::string& filename,
                              char delimiter = ',',
                              bool header = true);
    void to_csv(const std::string& filename,
                char delimiter = ',') const;
    std::string to_string(size_t max_rows = 20,
                          int col_width = 12) const;
    void print(size_t max_rows = 20) const;

    // Serialization
    void save(const std::string& filename) const;
    static DataFrame load(const std::string& filename);
};

} // namespace nn
#endif // DATAFRAME_H
