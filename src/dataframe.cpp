// dataframe.cpp
#include "dataframe.h"

namespace nn {

Column::Column() : dtype(DType::FLOAT) {}

Column::Column(const std::string& n, DType dt)
    : name(n), dtype(dt) {}

Column::Column(const std::string& n, const Tensor1D& data)
    : name(n), dtype(DType::FLOAT), fdata(data),
      null_mask(data.size(), false) {}

Column::Column(const std::string& n, const std::vector<int>& data)
    : name(n), dtype(DType::INT), idata(data),
      null_mask(data.size(), false) {}

Column::Column(const std::string& n, const std::vector<std::string>& data)
    : name(n), dtype(DType::STRING), sdata(data),
      null_mask(data.size(), false) {}

size_t Column::size() const {
    switch (dtype) {
        case DType::FLOAT: return fdata.size();
        case DType::INT: return idata.size();
        case DType::STRING: return sdata.size();
    }
    return 0;
}

bool Column::is_null(size_t i) const {
    return i < null_mask.size() && null_mask[i];
}

void Column::set_null(size_t i) {
    if (i < null_mask.size()) null_mask[i] = true;
}

float Column::get_float(size_t i) const {
    if (dtype == DType::FLOAT) return fdata[i];
    if (dtype == DType::INT) return static_cast<float>(idata[i]);
    return 0.0f;
}

int Column::get_int(size_t i) const {
    if (dtype == DType::INT) return idata[i];
    if (dtype == DType::FLOAT) return static_cast<int>(fdata[i]);
    return 0;
}

std::string Column::get_string(size_t i) const {
    if (dtype == DType::STRING) return sdata[i];
    if (dtype == DType::FLOAT) return std::to_string(fdata[i]);
    if (dtype == DType::INT) return std::to_string(idata[i]);
    return "";
}

void Column::push_float(float v, bool null) {
    fdata.push_back(v);
    null_mask.push_back(null);
}

void Column::push_int(int v, bool null) {
    idata.push_back(v);
    null_mask.push_back(null);
}

void Column::push_string(const std::string& v, bool null) {
    sdata.push_back(v);
    null_mask.push_back(null);
}

float Column::sum() const {
    float s = 0.0f;
    for (size_t i = 0; i < size(); ++i) {
        if (!is_null(i)) s += get_float(i);
    }
    return s;
}

float Column::mean() const {
    int n = count_valid();
    if (n == 0) return 0.0f;
    return sum() / n;
}

float Column::variance() const {
    int n = count_valid();
    if (n <= 1) return 0.0f;
    float m = mean();
    float s = 0.0f;
    for (size_t i = 0; i < size(); ++i) {
        if (!is_null(i)) {
            float d = get_float(i) - m;
            s += d * d;
        }
    }
    return s / (n - 1);
}

float Column::stddev() const {
    return std::sqrt(variance());
}

float Column::min_val() const {
    float mn = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < size(); ++i) {
        if (!is_null(i)) mn = std::min(mn, get_float(i));
    }
    return mn;
}

float Column::max_val() const {
    float mx = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < size(); ++i) {
        if (!is_null(i)) mx = std::max(mx, get_float(i));
    }
    return mx;
}

float Column::median() const {
    return percentile(50.0f);
}

float Column::percentile(float p) const {
    std::vector<float> vals;
    for (size_t i = 0; i < size(); ++i) {
        if (!is_null(i)) vals.push_back(get_float(i));
    }
    if (vals.empty()) return 0.0f;
    std::sort(vals.begin(), vals.end());
    float idx = (p / 100.0f) * (vals.size() - 1);
    int lo = static_cast<int>(idx);
    int hi = lo + 1;
    if (hi >= (int)vals.size()) return vals.back();
    float frac = idx - lo;
    return vals[lo] * (1.0f - frac) + vals[hi] * frac;
}

float Column::skewness() const {
    int n = count_valid();
    if (n < 3) return 0.0f;
    float m = mean();
    float sd = stddev();
    if (sd == 0.0f) return 0.0f;
    float s = 0.0f;
    for (size_t i = 0; i < size(); ++i) {
        if (!is_null(i)) {
            float d = (get_float(i) - m) / sd;
            s += d * d * d;
        }
    }
    return s * n / ((n - 1.0f) * (n - 2.0f));
}

float Column::kurtosis() const {
    int n = count_valid();
    if (n < 4) return 0.0f;
    float m = mean();
    float sd = stddev();
    if (sd == 0.0f) return 0.0f;
    float s = 0.0f;
    for (size_t i = 0; i < size(); ++i) {
        if (!is_null(i)) {
            float d = (get_float(i) - m) / sd;
            s += d * d * d * d;
        }
    }
    float k = (s * n * (n + 1.0f)) / ((n - 1.0f) * (n - 2.0f) * (n - 3.0f));
    k -= 3.0f * (n - 1.0f) * (n - 1.0f) / ((n - 2.0f) * (n - 3.0f));
    return k;
}

int Column::count_nulls() const {
    int c = 0;
    for (bool b : null_mask) if (b) ++c;
    return c;
}

int Column::count_valid() const {
    return static_cast<int>(size()) - count_nulls();
}

Column Column::fill_null(float value) const {
    Column out = *this;
    for (size_t i = 0; i < out.size(); ++i) {
        if (out.is_null(i)) {
            if (out.dtype == DType::FLOAT) out.fdata[i] = value;
            else if (out.dtype == DType::INT) out.idata[i] = static_cast<int>(value);
            out.null_mask[i] = false;
        }
    }
    return out;
}

Column Column::fill_null_mean() const {
    return fill_null(mean());
}

Column Column::fill_null_median() const {
    return fill_null(median());
}

Column Column::fill_forward() const {
    Column out = *this;
    for (size_t i = 1; i < out.size(); ++i) {
        if (out.is_null(i) && !out.is_null(i - 1)) {
            if (out.dtype == DType::FLOAT) out.fdata[i] = out.fdata[i - 1];
            else if (out.dtype == DType::INT) out.idata[i] = out.idata[i - 1];
            else if (out.dtype == DType::STRING) out.sdata[i] = out.sdata[i - 1];
            out.null_mask[i] = false;
        }
    }
    return out;
}

Column Column::fill_backward() const {
    Column out = *this;
    for (int i = static_cast<int>(out.size()) - 2; i >= 0; --i) {
        if (out.is_null(i) && !out.is_null(i + 1)) {
            if (out.dtype == DType::FLOAT) out.fdata[i] = out.fdata[i + 1];
            else if (out.dtype == DType::INT) out.idata[i] = out.idata[i + 1];
            else if (out.dtype == DType::STRING) out.sdata[i] = out.sdata[i + 1];
            out.null_mask[i] = false;
        }
    }
    return out;
}

Column Column::apply(std::function<float(float)> fn) const {
    Column out(name, DType::FLOAT);
    for (size_t i = 0; i < size(); ++i) {
        if (is_null(i)) {
            out.push_float(0.0f, true);
        } else {
            out.push_float(fn(get_float(i)));
        }
    }
    return out;
}

Column Column::log_transform() const {
    return apply([](float x) { return std::log(x + 1e-10f); });
}

Column Column::sqrt_transform() const {
    return apply([](float x) { return std::sqrt(std::max(0.0f, x)); });
}

Column Column::normalize() const {
    float mn = min_val(), mx = max_val();
    float range = mx - mn;
    if (range == 0.0f) range = 1.0f;
    return apply([mn, range](float x) { return (x - mn) / range; });
}

Column Column::standardize() const {
    float m = mean(), s = stddev();
    if (s == 0.0f) s = 1.0f;
    return apply([m, s](float x) { return (x - m) / s; });
}

Column Column::robust_scale() const {
    float med = median();
    float q1 = percentile(25.0f);
    float q3 = percentile(75.0f);
    float iqr_val = q3 - q1;
    if (iqr_val == 0.0f) iqr_val = 1.0f;
    return apply([med, iqr_val](float x) { return (x - med) / iqr_val; });
}

Column Column::clip(float lo, float hi) const {
    return apply([lo, hi](float x) { return std::max(lo, std::min(hi, x)); });
}

Column Column::abs_col() const {
    return apply([](float x) { return std::abs(x); });
}

Column Column::diff(int periods) const {
    Column out(name + "_diff", DType::FLOAT);
    for (size_t i = 0; i < size(); ++i) {
        if (static_cast<int>(i) < periods || is_null(i) || is_null(i - periods)) {
            out.push_float(0.0f, true);
        } else {
            out.push_float(get_float(i) - get_float(i - periods));
        }
    }
    return out;
}

Column Column::pct_change(int periods) const {
    Column out(name + "_pct", DType::FLOAT);
    for (size_t i = 0; i < size(); ++i) {
        if (static_cast<int>(i) < periods || is_null(i) || is_null(i - periods)) {
            out.push_float(0.0f, true);
        } else {
            float prev = get_float(i - periods);
            if (prev == 0.0f) out.push_float(0.0f, true);
            else out.push_float((get_float(i) - prev) / prev);
        }
    }
    return out;
}

Column Column::cumsum() const {
    Column out(name + "_cumsum", DType::FLOAT);
    float acc = 0.0f;
    for (size_t i = 0; i < size(); ++i) {
        if (is_null(i)) {
            out.push_float(acc);
        } else {
            acc += get_float(i);
            out.push_float(acc);
        }
    }
    return out;
}

Column Column::cumprod() const {
    Column out(name + "_cumprod", DType::FLOAT);
    float acc = 1.0f;
    for (size_t i = 0; i < size(); ++i) {
        if (is_null(i)) {
            out.push_float(acc);
        } else {
            acc *= get_float(i);
            out.push_float(acc);
        }
    }
    return out;
}

Column Column::rolling_mean(int window) const {
    Column out(name + "_rmean", DType::FLOAT);
    for (size_t i = 0; i < size(); ++i) {
        if (static_cast<int>(i) < window - 1) {
            out.push_float(0.0f, true);
        } else {
            float s = 0.0f;
            int cnt = 0;
            for (int j = 0; j < window; ++j) {
                if (!is_null(i - j)) {
                    s += get_float(i - j);
                    ++cnt;
                }
            }
            out.push_float(cnt > 0 ? s / cnt : 0.0f, cnt == 0);
        }
    }
    return out;
}

Column Column::rolling_std(int window) const {
    Column out(name + "_rstd", DType::FLOAT);
    for (size_t i = 0; i < size(); ++i) {
        if (static_cast<int>(i) < window - 1) {
            out.push_float(0.0f, true);
        } else {
            float s = 0.0f, s2 = 0.0f;
            int cnt = 0;
            for (int j = 0; j < window; ++j) {
                if (!is_null(i - j)) {
                    float v = get_float(i - j);
                    s += v;
                    s2 += v * v;
                    ++cnt;
                }
            }
            if (cnt <= 1) {
                out.push_float(0.0f, true);
            } else {
                float m = s / cnt;
                float var = (s2 / cnt) - m * m;
                out.push_float(std::sqrt(std::max(0.0f, var)));
            }
        }
    }
    return out;
}

Column Column::rolling_min(int window) const {
    Column out(name + "_rmin", DType::FLOAT);
    for (size_t i = 0; i < size(); ++i) {
        if (static_cast<int>(i) < window - 1) {
            out.push_float(0.0f, true);
        } else {
            float mn = std::numeric_limits<float>::infinity();
            for (int j = 0; j < window; ++j) {
                if (!is_null(i - j)) mn = std::min(mn, get_float(i - j));
            }
            out.push_float(mn);
        }
    }
    return out;
}

Column Column::rolling_max(int window) const {
    Column out(name + "_rmax", DType::FLOAT);
    for (size_t i = 0; i < size(); ++i) {
        if (static_cast<int>(i) < window - 1) {
            out.push_float(0.0f, true);
        } else {
            float mx = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < window; ++j) {
                if (!is_null(i - j)) mx = std::max(mx, get_float(i - j));
            }
            out.push_float(mx);
        }
    }
    return out;
}

Column Column::ewm_mean(float alpha) const {
    Column out(name + "_ewm", DType::FLOAT);
    float ewm = 0.0f;
    bool started = false;
    for (size_t i = 0; i < size(); ++i) {
        if (is_null(i)) {
            out.push_float(ewm, !started);
        } else {
            if (!started) {
                ewm = get_float(i);
                started = true;
            } else {
                ewm = alpha * get_float(i) + (1.0f - alpha) * ewm;
            }
            out.push_float(ewm);
        }
    }
    return out;
}

Column Column::shift(int periods) const {
    Column out(name + "_shift", dtype);
    for (size_t i = 0; i < size(); ++i) {
        int src = static_cast<int>(i) - periods;
        if (src < 0 || src >= static_cast<int>(size())) {
            if (dtype == DType::FLOAT) out.push_float(0.0f, true);
            else if (dtype == DType::INT) out.push_int(0, true);
            else out.push_string("", true);
        } else {
            if (dtype == DType::FLOAT) out.push_float(fdata[src], is_null(src));
            else if (dtype == DType::INT) out.push_int(idata[src], is_null(src));
            else out.push_string(sdata[src], is_null(src));
        }
    }
    return out;
}

Column Column::rank() const {
    Column out(name + "_rank", DType::FLOAT);
    auto sorted = argsort(true);
    std::vector<float> ranks_out(size(), 0.0f);
    for (size_t i = 0; i < sorted.size(); ++i) {
        ranks_out[sorted[i]] = static_cast<float>(i + 1);
    }
    for (size_t i = 0; i < size(); ++i) {
        out.push_float(ranks_out[i], is_null(i));
    }
    return out;
}

Column Column::str_lower() const {
    Column out(name, DType::STRING);
    for (size_t i = 0; i < size(); ++i) {
        std::string s = get_string(i);
        std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        out.push_string(s, is_null(i));
    }
    return out;
}

Column Column::str_upper() const {
    Column out(name, DType::STRING);
    for (size_t i = 0; i < size(); ++i) {
        std::string s = get_string(i);
        std::transform(s.begin(), s.end(), s.begin(), ::toupper);
        out.push_string(s, is_null(i));
    }
    return out;
}

Column Column::str_len() const {
    Column out(name + "_len", DType::FLOAT);
    for (size_t i = 0; i < size(); ++i) {
        out.push_float(static_cast<float>(get_string(i).size()), is_null(i));
    }
    return out;
}

Column Column::str_contains(const std::string& substr) const {
    Column out(name + "_contains", DType::FLOAT);
    for (size_t i = 0; i < size(); ++i) {
        out.push_float(get_string(i).find(substr) != std::string::npos ? 1.0f : 0.0f, is_null(i));
    }
    return out;
}

Column Column::str_replace(const std::string& from, const std::string& to) const {
    Column out(name, DType::STRING);
    for (size_t i = 0; i < size(); ++i) {
        std::string s = get_string(i);
        size_t pos = 0;
        while ((pos = s.find(from, pos)) != std::string::npos) {
            s.replace(pos, from.length(), to);
            pos += to.length();
        }
        out.push_string(s, is_null(i));
    }
    return out;
}

Column Column::label_encode() const {
    Column out(name + "_encoded", DType::INT);
    std::map<std::string, int> mapping;
    int next_id = 0;
    for (size_t i = 0; i < size(); ++i) {
        if (is_null(i)) {
            out.push_int(-1, true);
        } else {
            std::string key = get_string(i);
            if (mapping.find(key) == mapping.end()) {
                mapping[key] = next_id++;
            }
            out.push_int(mapping[key]);
        }
    }
    return out;
}

std::vector<Column> Column::one_hot_encode() const {
    std::set<std::string> unique_vals;
    for (size_t i = 0; i < size(); ++i) {
        if (!is_null(i)) unique_vals.insert(get_string(i));
    }
    std::vector<Column> out;
    for (const auto& val : unique_vals) {
        Column c(name + "_" + val, DType::FLOAT);
        for (size_t i = 0; i < size(); ++i) {
            c.push_float(get_string(i) == val ? 1.0f : 0.0f, is_null(i));
        }
        out.push_back(c);
    }
    return out;
}

std::vector<size_t> Column::argsort(bool ascending) const {
    std::vector<size_t> idx(size());
    std::iota(idx.begin(), idx.end(), 0);
    if (dtype == DType::FLOAT || dtype == DType::INT) {
        std::sort(idx.begin(), idx.end(), [this, ascending](size_t a, size_t b) {
            if (is_null(a)) return false;
            if (is_null(b)) return true;
            float va = get_float(a), vb = get_float(b);
            return ascending ? va < vb : va > vb;
        });
    } else {
        std::sort(idx.begin(), idx.end(), [this, ascending](size_t a, size_t b) {
            if (is_null(a)) return false;
            if (is_null(b)) return true;
            return ascending ? sdata[a] < sdata[b] : sdata[a] > sdata[b];
        });
    }
    return idx;
}

Tensor1D Column::to_tensor() const {
    Tensor1D out(size());
    for (size_t i = 0; i < size(); ++i) {
        out[i] = is_null(i) ? 0.0f : get_float(i);
    }
    return out;
}

Column Column::as_float() const {
    Column out(name, DType::FLOAT);
    for (size_t i = 0; i < size(); ++i) {
        out.push_float(get_float(i), is_null(i));
    }
    return out;
}

DataFrame::DataFrame() {}

DataFrame::DataFrame(const std::vector<Column>& cols) {
    for (const auto& c : cols) add_column(c);
}

size_t DataFrame::nrows() const {
    return columns.empty() ? 0 : columns[0].size();
}

size_t DataFrame::ncols() const {
    return columns.size();
}

std::vector<std::string> DataFrame::column_names() const {
    std::vector<std::string> names;
    for (const auto& c : columns) names.push_back(c.name);
    return names;
}

std::vector<DType> DataFrame::column_types() const {
    std::vector<DType> types;
    for (const auto& c : columns) types.push_back(c.dtype);
    return types;
}

Column& DataFrame::operator[](const std::string& name) {
    auto it = col_index.find(name);
    if (it == col_index.end()) throw std::invalid_argument("Column not found: " + name);
    return columns[it->second];
}

const Column& DataFrame::operator[](const std::string& name) const {
    auto it = col_index.find(name);
    if (it == col_index.end()) throw std::invalid_argument("Column not found: " + name);
    return columns[it->second];
}

Column& DataFrame::col(size_t i) { return columns[i]; }
const Column& DataFrame::col(size_t i) const { return columns[i]; }

bool DataFrame::has_column(const std::string& name) const {
    return col_index.find(name) != col_index.end();
}

void DataFrame::add_column(const Column& c) {
    col_index[c.name] = columns.size();
    columns.push_back(c);
}

void DataFrame::add_column(const std::string& name, const Tensor1D& data) {
    add_column(Column(name, data));
}

void DataFrame::add_column(const std::string& name, const std::vector<int>& data) {
    add_column(Column(name, data));
}

void DataFrame::add_column(const std::string& name, const std::vector<std::string>& data) {
    add_column(Column(name, data));
}

void DataFrame::drop_column(const std::string& name) {
    auto it = col_index.find(name);
    if (it == col_index.end()) return;
    size_t idx = it->second;
    columns.erase(columns.begin() + idx);
    col_index.clear();
    for (size_t i = 0; i < columns.size(); ++i) {
        col_index[columns[i].name] = i;
    }
}

void DataFrame::rename_column(const std::string& old_name, const std::string& new_name) {
    auto it = col_index.find(old_name);
    if (it == col_index.end()) return;
    size_t idx = it->second;
    columns[idx].name = new_name;
    col_index.erase(old_name);
    col_index[new_name] = idx;
}

void DataFrame::add_row(const std::vector<float>& row) {
    for (size_t i = 0; i < std::min(row.size(), columns.size()); ++i) {
        if (columns[i].dtype == DType::FLOAT) columns[i].push_float(row[i]);
        else if (columns[i].dtype == DType::INT) columns[i].push_int(static_cast<int>(row[i]));
    }
}

DataFrame DataFrame::head(size_t n) const {
    return slice(0, std::min(n, nrows()));
}

DataFrame DataFrame::tail(size_t n) const {
    size_t nr = nrows();
    size_t start = nr > n ? nr - n : 0;
    return slice(start, nr);
}

DataFrame DataFrame::sample(size_t n) const {
    size_t nr = nrows();
    n = std::min(n, nr);
    std::vector<size_t> indices(nr);
    std::iota(indices.begin(), indices.end(), 0);
    for (size_t i = nr - 1; i > 0; --i) {
        size_t j = std::rand() % (i + 1);
        std::swap(indices[i], indices[j]);
    }
    indices.resize(n);
    std::sort(indices.begin(), indices.end());

    DataFrame out;
    for (const auto& c : columns) {
        Column nc(c.name, c.dtype);
        for (size_t idx : indices) {
            if (c.dtype == DType::FLOAT) nc.push_float(c.fdata[idx], c.is_null(idx));
            else if (c.dtype == DType::INT) nc.push_int(c.idata[idx], c.is_null(idx));
            else nc.push_string(c.sdata[idx], c.is_null(idx));
        }
        out.add_column(nc);
    }
    return out;
}

DataFrame DataFrame::slice(size_t start, size_t end) const {
    DataFrame out;
    for (const auto& c : columns) {
        Column nc(c.name, c.dtype);
        for (size_t i = start; i < std::min(end, c.size()); ++i) {
            if (c.dtype == DType::FLOAT) nc.push_float(c.fdata[i], c.is_null(i));
            else if (c.dtype == DType::INT) nc.push_int(c.idata[i], c.is_null(i));
            else nc.push_string(c.sdata[i], c.is_null(i));
        }
        out.add_column(nc);
    }
    return out;
}

DataFrame DataFrame::select(const std::vector<std::string>& cols) const {
    DataFrame out;
    for (const auto& name : cols) {
        if (has_column(name)) out.add_column((*this)[name]);
    }
    return out;
}

DataFrame DataFrame::filter(const std::string& col_name,
                             std::function<bool(float)> pred) const {
    const Column& fc = (*this)[col_name];
    std::vector<size_t> indices;
    for (size_t i = 0; i < nrows(); ++i) {
        if (!fc.is_null(i) && pred(fc.get_float(i))) {
            indices.push_back(i);
        }
    }

    DataFrame out;
    for (const auto& c : columns) {
        Column nc(c.name, c.dtype);
        for (size_t idx : indices) {
            if (c.dtype == DType::FLOAT) nc.push_float(c.fdata[idx], c.is_null(idx));
            else if (c.dtype == DType::INT) nc.push_int(c.idata[idx], c.is_null(idx));
            else nc.push_string(c.sdata[idx], c.is_null(idx));
        }
        out.add_column(nc);
    }
    return out;
}

DataFrame DataFrame::filter_str(const std::string& col_name,
                                 std::function<bool(const std::string&)> pred) const {
    const Column& fc = (*this)[col_name];
    std::vector<size_t> indices;
    for (size_t i = 0; i < nrows(); ++i) {
        if (!fc.is_null(i) && pred(fc.get_string(i))) {
            indices.push_back(i);
        }
    }

    DataFrame out;
    for (const auto& c : columns) {
        Column nc(c.name, c.dtype);
        for (size_t idx : indices) {
            if (c.dtype == DType::FLOAT) nc.push_float(c.fdata[idx], c.is_null(idx));
            else if (c.dtype == DType::INT) nc.push_int(c.idata[idx], c.is_null(idx));
            else nc.push_string(c.sdata[idx], c.is_null(idx));
        }
        out.add_column(nc);
    }
    return out;
}

DataFrame DataFrame::drop_nulls(const std::string& col_name) const {
    std::vector<size_t> indices;
    for (size_t i = 0; i < nrows(); ++i) {
        bool has_null = false;
        if (col_name.empty()) {
            for (const auto& c : columns) {
                if (c.is_null(i)) { has_null = true; break; }
            }
        } else {
            has_null = (*this)[col_name].is_null(i);
        }
        if (!has_null) indices.push_back(i);
    }

    DataFrame out;
    for (const auto& c : columns) {
        Column nc(c.name, c.dtype);
        for (size_t idx : indices) {
            if (c.dtype == DType::FLOAT) nc.push_float(c.fdata[idx], c.is_null(idx));
            else if (c.dtype == DType::INT) nc.push_int(c.idata[idx], c.is_null(idx));
            else nc.push_string(c.sdata[idx], c.is_null(idx));
        }
        out.add_column(nc);
    }
    return out;
}

DataFrame DataFrame::drop_duplicates(const std::vector<std::string>& subset) const {
    std::set<std::string> seen;
    std::vector<size_t> indices;
    auto cols_to_check = subset.empty() ? column_names() : subset;

    for (size_t i = 0; i < nrows(); ++i) {
        std::string key;
        for (const auto& cn : cols_to_check) {
            key += (*this)[cn].get_string(i) + "|";
        }
        if (seen.find(key) == seen.end()) {
            seen.insert(key);
            indices.push_back(i);
        }
    }

    DataFrame out;
    for (const auto& c : columns) {
        Column nc(c.name, c.dtype);
        for (size_t idx : indices) {
            if (c.dtype == DType::FLOAT) nc.push_float(c.fdata[idx], c.is_null(idx));
            else if (c.dtype == DType::INT) nc.push_int(c.idata[idx], c.is_null(idx));
            else nc.push_string(c.sdata[idx], c.is_null(idx));
        }
        out.add_column(nc);
    }
    return out;
}

DataFrame DataFrame::sort_by(const std::string& col_name, bool ascending) const {
    auto idx = (*this)[col_name].argsort(ascending);
    DataFrame out;
    for (const auto& c : columns) {
        Column nc(c.name, c.dtype);
        for (size_t i : idx) {
            if (c.dtype == DType::FLOAT) nc.push_float(c.fdata[i], c.is_null(i));
            else if (c.dtype == DType::INT) nc.push_int(c.idata[i], c.is_null(i));
            else nc.push_string(c.sdata[i], c.is_null(i));
        }
        out.add_column(nc);
    }
    return out;
}

DataFrame DataFrame::sort_by(const std::vector<std::string>& cols,
                              const std::vector<bool>& ascending) const {
    std::vector<size_t> idx(nrows());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [this, &cols, &ascending](size_t a, size_t b) {
        for (size_t c = 0; c < cols.size(); ++c) {
            const Column& col = (*this)[cols[c]];
            float va = col.get_float(a), vb = col.get_float(b);
            if (va != vb) return ascending[c] ? va < vb : va > vb;
        }
        return false;
    });
    DataFrame out;
    for (const auto& c : columns) {
        Column nc(c.name, c.dtype);
        for (size_t i : idx) {
            if (c.dtype == DType::FLOAT) nc.push_float(c.fdata[i], c.is_null(i));
            else if (c.dtype == DType::INT) nc.push_int(c.idata[i], c.is_null(i));
            else nc.push_string(c.sdata[i], c.is_null(i));
        }
        out.add_column(nc);
    }
    return out;
}

DataFrame::GroupResult DataFrame::group_by(const std::string& col_name) const {
    GroupResult result;
    const Column& gc = (*this)[col_name];
    std::map<std::string, std::vector<size_t>> groups_map;
    for (size_t i = 0; i < nrows(); ++i) {
        groups_map[gc.get_string(i)].push_back(i);
    }
    for (const auto& [key, indices] : groups_map) {
        result.group_keys.push_back(key);
        DataFrame g;
        for (const auto& c : columns) {
            Column nc(c.name, c.dtype);
            for (size_t idx : indices) {
                if (c.dtype == DType::FLOAT) nc.push_float(c.fdata[idx], c.is_null(idx));
                else if (c.dtype == DType::INT) nc.push_int(c.idata[idx], c.is_null(idx));
                else nc.push_string(c.sdata[idx], c.is_null(idx));
            }
            g.add_column(nc);
        }
        result.groups.push_back(g);
    }
    return result;
}

DataFrame DataFrame::group_agg(const std::string& group_col,
                                const std::string& value_col,
                                const std::string& agg_func) const {
    auto gr = group_by(group_col);
    DataFrame out;
    Column gc(group_col, DType::STRING);
    Column vc(value_col + "_" + agg_func, DType::FLOAT);

    for (size_t i = 0; i < gr.group_keys.size(); ++i) {
        gc.push_string(gr.group_keys[i]);
        const Column& val_col = gr.groups[i][value_col];
        float agg_val = 0.0f;
        if (agg_func == "sum") agg_val = val_col.sum();
        else if (agg_func == "mean") agg_val = val_col.mean();
        else if (agg_func == "min") agg_val = val_col.min_val();
        else if (agg_func == "max") agg_val = val_col.max_val();
        else if (agg_func == "std") agg_val = val_col.stddev();
        else if (agg_func == "median") agg_val = val_col.median();
        else if (agg_func == "count") agg_val = static_cast<float>(val_col.count_valid());
        vc.push_float(agg_val);
    }
    out.add_column(gc);
    out.add_column(vc);
    return out;
}

DataFrame DataFrame::pivot(const std::string& index_col,
                            const std::string& columns_col,
                            const std::string& values_col) const {
    std::set<std::string> unique_indices, unique_cols;
    for (size_t i = 0; i < nrows(); ++i) {
        unique_indices.insert((*this)[index_col].get_string(i));
        unique_cols.insert((*this)[columns_col].get_string(i));
    }

    DataFrame out;
    Column idx(index_col, DType::STRING);
    for (const auto& v : unique_indices) idx.push_string(v);
    out.add_column(idx);

    for (const auto& col_val : unique_cols) {
        Column c(col_val, DType::FLOAT);
        for (const auto& idx_val : unique_indices) {
            float found_val = 0.0f;
            bool found = false;
            for (size_t i = 0; i < nrows(); ++i) {
                if ((*this)[index_col].get_string(i) == idx_val &&
                    (*this)[columns_col].get_string(i) == col_val) {
                    found_val = (*this)[values_col].get_float(i);
                    found = true;
                    break;
                }
            }
            c.push_float(found_val, !found);
        }
        out.add_column(c);
    }
    return out;
}

DataFrame DataFrame::merge(const DataFrame& other,
                            const std::string& on,
                            const std::string& how) const {
    // Build index for right table
    std::multimap<std::string, size_t> right_idx;
    for (size_t i = 0; i < other.nrows(); ++i) {
        right_idx.insert({other[on].get_string(i), i});
    }

    DataFrame out;
    // Initialize all columns
    for (const auto& c : columns) {
        out.add_column(Column(c.name, c.dtype));
    }
    for (const auto& c : other.columns) {
        if (c.name != on) {
            std::string n = c.name;
            if (has_column(n)) n += "_right";
            out.add_column(Column(n, c.dtype));
        }
    }

    auto push_left = [&](size_t i) {
        for (size_t ci = 0; ci < columns.size(); ++ci) {
            const auto& c = columns[ci];
            if (c.dtype == DType::FLOAT) out.columns[ci].push_float(c.fdata[i], c.is_null(i));
            else if (c.dtype == DType::INT) out.columns[ci].push_int(c.idata[i], c.is_null(i));
            else out.columns[ci].push_string(c.sdata[i], c.is_null(i));
        }
    };

    auto push_right = [&](size_t i) {
        size_t offset = columns.size();
        for (size_t ci = 0; ci < other.columns.size(); ++ci) {
            if (other.columns[ci].name == on) continue;
            const auto& c = other.columns[ci];
            if (c.dtype == DType::FLOAT) out.columns[offset].push_float(c.fdata[i], c.is_null(i));
            else if (c.dtype == DType::INT) out.columns[offset].push_int(c.idata[i], c.is_null(i));
            else out.columns[offset].push_string(c.sdata[i], c.is_null(i));
            ++offset;
        }
    };

    auto push_right_null = [&]() {
        size_t offset = columns.size();
        for (size_t ci = 0; ci < other.columns.size(); ++ci) {
            if (other.columns[ci].name == on) continue;
            const auto& c = other.columns[ci];
            if (c.dtype == DType::FLOAT) out.columns[offset].push_float(0.0f, true);
            else if (c.dtype == DType::INT) out.columns[offset].push_int(0, true);
            else out.columns[offset].push_string("", true);
            ++offset;
        }
    };

    for (size_t i = 0; i < nrows(); ++i) {
        std::string key = (*this)[on].get_string(i);
        auto range = right_idx.equal_range(key);
        bool matched = range.first != range.second;
        if (matched) {
            for (auto it = range.first; it != range.second; ++it) {
                push_left(i);
                push_right(it->second);
            }
        } else if (how == "left" || how == "outer") {
            push_left(i);
            push_right_null();
        }
    }
    return out;
}

DataFrame DataFrame::describe() const {
    DataFrame out;
    Column stat_col("statistic", DType::STRING);
    stat_col.push_string("count");
    stat_col.push_string("mean");
    stat_col.push_string("std");
    stat_col.push_string("min");
    stat_col.push_string("25%");
    stat_col.push_string("50%");
    stat_col.push_string("75%");
    stat_col.push_string("max");
    out.add_column(stat_col);

    for (const auto& c : columns) {
        if (c.dtype == DType::STRING) continue;
        Column vc(c.name, DType::FLOAT);
        vc.push_float(static_cast<float>(c.count_valid()));
        vc.push_float(c.mean());
        vc.push_float(c.stddev());
        vc.push_float(c.min_val());
        vc.push_float(c.percentile(25.0f));
        vc.push_float(c.median());
        vc.push_float(c.percentile(75.0f));
        vc.push_float(c.max_val());
        out.add_column(vc);
    }
    return out;
}

DataFrame DataFrame::corr() const {
    std::vector<std::string> numeric_cols;
    for (const auto& c : columns) {
        if (c.dtype != DType::STRING) numeric_cols.push_back(c.name);
    }

    DataFrame out;
    Column name_col("", DType::STRING);
    for (const auto& n : numeric_cols) name_col.push_string(n);
    out.add_column(name_col);

    for (const auto& ci : numeric_cols) {
        Column vc(ci, DType::FLOAT);
        Tensor1D xi = (*this)[ci].to_tensor();
        for (const auto& cj : numeric_cols) {
            Tensor1D xj = (*this)[cj].to_tensor();
            // Pearson correlation
            float mx = 0, my = 0;
            int n = xi.size();
            for (int k = 0; k < n; ++k) { mx += xi[k]; my += xj[k]; }
            mx /= n; my /= n;
            float num = 0, dx = 0, dy = 0;
            for (int k = 0; k < n; ++k) {
                num += (xi[k] - mx) * (xj[k] - my);
                dx += (xi[k] - mx) * (xi[k] - mx);
                dy += (xj[k] - my) * (xj[k] - my);
            }
            float denom = std::sqrt(dx * dy);
            vc.push_float(denom > 0 ? num / denom : 0.0f);
        }
        out.add_column(vc);
    }
    return out;
}

DataFrame DataFrame::cov() const {
    std::vector<std::string> numeric_cols;
    for (const auto& c : columns) {
        if (c.dtype != DType::STRING) numeric_cols.push_back(c.name);
    }

    DataFrame out;
    Column name_col("", DType::STRING);
    for (const auto& n : numeric_cols) name_col.push_string(n);
    out.add_column(name_col);

    for (const auto& ci : numeric_cols) {
        Column vc(ci, DType::FLOAT);
        Tensor1D xi = (*this)[ci].to_tensor();
        for (const auto& cj : numeric_cols) {
            Tensor1D xj = (*this)[cj].to_tensor();
            float mx = 0, my = 0;
            int n = xi.size();
            for (int k = 0; k < n; ++k) { mx += xi[k]; my += xj[k]; }
            mx /= n; my /= n;
            float cov_val = 0;
            for (int k = 0; k < n; ++k) {
                cov_val += (xi[k] - mx) * (xj[k] - my);
            }
            vc.push_float(n > 1 ? cov_val / (n - 1) : 0.0f);
        }
        out.add_column(vc);
    }
    return out;
}

DataFrame DataFrame::value_counts(const std::string& col_name) const {
    std::map<std::string, int> counts;
    const Column& c = (*this)[col_name];
    for (size_t i = 0; i < c.size(); ++i) {
        if (!c.is_null(i)) counts[c.get_string(i)]++;
    }
    DataFrame out;
    Column vc(col_name, DType::STRING);
    Column cc("count", DType::INT);
    for (const auto& [k, v] : counts) {
        vc.push_string(k);
        cc.push_int(v);
    }
    out.add_column(vc);
    out.add_column(cc);
    return out;
}

DataFrame DataFrame::null_report() const {
    DataFrame out;
    Column name_col("column", DType::STRING);
    Column null_col("null_count", DType::INT);
    Column pct_col("null_pct", DType::FLOAT);
    Column valid_col("valid_count", DType::INT);

    for (const auto& c : columns) {
        name_col.push_string(c.name);
        null_col.push_int(c.count_nulls());
        pct_col.push_float(c.size() > 0 ? 100.0f * c.count_nulls() / c.size() : 0.0f);
        valid_col.push_int(c.count_valid());
    }
    out.add_column(name_col);
    out.add_column(null_col);
    out.add_column(pct_col);
    out.add_column(valid_col);
    return out;
}

DataFrame DataFrame::apply_col(const std::string& col_name,
                                std::function<float(float)> fn) const {
    DataFrame out = *this;
    out[col_name] = out[col_name].apply(fn);
    return out;
}

DataFrame DataFrame::add_derived(const std::string& new_name,
                                  std::function<float(const DataFrame&, size_t)> fn) const {
    DataFrame out = *this;
    Column c(new_name, DType::FLOAT);
    for (size_t i = 0; i < nrows(); ++i) {
        c.push_float(fn(*this, i));
    }
    out.add_column(c);
    return out;
}

std::vector<Tensor1D> DataFrame::to_tensors(
    const std::vector<std::string>& feature_cols) const {
    size_t nr = nrows();
    std::vector<Tensor1D> out(nr, Tensor1D(feature_cols.size()));
    for (size_t j = 0; j < feature_cols.size(); ++j) {
        const Column& c = (*this)[feature_cols[j]];
        for (size_t i = 0; i < nr; ++i) {
            out[i][j] = c.is_null(i) ? 0.0f : c.get_float(i);
        }
    }
    return out;
}

Tensor2D DataFrame::to_matrix(const std::vector<std::string>& feature_cols) const {
    size_t nr = nrows();
    Tensor2D out(nr, Tensor1D(feature_cols.size()));
    for (size_t j = 0; j < feature_cols.size(); ++j) {
        const Column& c = (*this)[feature_cols[j]];
        for (size_t i = 0; i < nr; ++i) {
            out[i][j] = c.is_null(i) ? 0.0f : c.get_float(i);
        }
    }
    return out;
}

DataFrame DataFrame::from_tensors(const std::vector<Tensor1D>& data,
                                   const std::vector<std::string>& col_names) {
    DataFrame out;
    if (data.empty()) return out;
    size_t n_cols = data[0].size();
    for (size_t j = 0; j < n_cols; ++j) {
        Column c(j < col_names.size() ? col_names[j] : "col_" + std::to_string(j),
                 DType::FLOAT);
        for (const auto& row : data) {
            c.push_float(j < row.size() ? row[j] : 0.0f);
        }
        out.add_column(c);
    }
    return out;
}

DataFrame DataFrame::read_csv(const std::string& filename,
                               char delimiter, bool header) {
    DataFrame out;
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::string line;
    std::vector<std::string> col_names;
    std::vector<std::vector<std::string>> raw_data;

    // Read header
    if (header && std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;
        while (std::getline(ss, token, delimiter)) {
            // Trim whitespace
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);
            // Remove surrounding quotes
            if (token.size() >= 2 && token.front() == '"' && token.back() == '"') {
                token = token.substr(1, token.size() - 2);
            }
            col_names.push_back(token);
        }
    }

    // Read data
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string token;
        std::vector<std::string> row;
        while (std::getline(ss, token, delimiter)) {
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);
            if (token.size() >= 2 && token.front() == '"' && token.back() == '"') {
                token = token.substr(1, token.size() - 2);
            }
            row.push_back(token);
        }
        raw_data.push_back(row);
    }

    if (raw_data.empty()) return out;
    size_t n_cols = col_names.empty() ? raw_data[0].size() : col_names.size();

    if (col_names.empty()) {
        for (size_t i = 0; i < n_cols; ++i) {
            col_names.push_back("col_" + std::to_string(i));
        }
    }

    // Detect types: try float, fallback to string
    for (size_t j = 0; j < n_cols; ++j) {
        bool all_numeric = true;
        for (const auto& row : raw_data) {
            if (j >= row.size() || row[j].empty() || row[j] == "NA" || row[j] == "null" || row[j] == "NaN") continue;
            try {
                std::stof(row[j]);
            } catch (...) {
                all_numeric = false;
                break;
            }
        }

        if (all_numeric) {
            Column c(col_names[j], DType::FLOAT);
            for (const auto& row : raw_data) {
                if (j >= row.size() || row[j].empty() || row[j] == "NA" || row[j] == "null" || row[j] == "NaN") {
                    c.push_float(0.0f, true);
                } else {
                    c.push_float(std::stof(row[j]));
                }
            }
            out.add_column(c);
        } else {
            Column c(col_names[j], DType::STRING);
            for (const auto& row : raw_data) {
                if (j >= row.size()) {
                    c.push_string("", true);
                } else {
                    bool is_null = row[j].empty() || row[j] == "NA" || row[j] == "null" || row[j] == "NaN";
                    c.push_string(row[j], is_null);
                }
            }
            out.add_column(c);
        }
    }
    return out;
}

void DataFrame::to_csv(const std::string& filename, char delimiter) const {
    std::ofstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    // Header
    for (size_t i = 0; i < columns.size(); ++i) {
        if (i > 0) file << delimiter;
        file << columns[i].name;
    }
    file << "\n";

    // Data
    for (size_t r = 0; r < nrows(); ++r) {
        for (size_t c = 0; c < columns.size(); ++c) {
            if (c > 0) file << delimiter;
            if (columns[c].is_null(r)) {
                file << "NA";
            } else {
                file << columns[c].get_string(r);
            }
        }
        file << "\n";
    }
}

std::string DataFrame::to_string(size_t max_rows, int col_width) const {
    std::ostringstream oss;
    size_t nr = nrows();

    // Header
    for (size_t i = 0; i < columns.size(); ++i) {
        oss << std::setw(col_width) << std::left
            << columns[i].name.substr(0, col_width - 1);
    }
    oss << "\n";
    for (size_t i = 0; i < columns.size(); ++i) {
        for (int j = 0; j < col_width; ++j) oss << "-";
    }
    oss << "\n";

    // Rows
    size_t show = std::min(max_rows, nr);
    bool truncated = nr > max_rows;
    size_t head_n = truncated ? max_rows / 2 : show;
    size_t tail_n = truncated ? max_rows - head_n : 0;

    for (size_t r = 0; r < head_n; ++r) {
        for (size_t c = 0; c < columns.size(); ++c) {
            std::string val = columns[c].is_null(r) ? "NA" : columns[c].get_string(r);
            oss << std::setw(col_width) << std::left << val.substr(0, col_width - 1);
        }
        oss << "\n";
    }

    if (truncated) {
        oss << "... (" << nr << " rows total)\n";
        for (size_t r = nr - tail_n; r < nr; ++r) {
            for (size_t c = 0; c < columns.size(); ++c) {
                std::string val = columns[c].is_null(r) ? "NA" : columns[c].get_string(r);
                oss << std::setw(col_width) << std::left << val.substr(0, col_width - 1);
            }
            oss << "\n";
        }
    }

    oss << "\n[" << nr << " rows x " << ncols() << " columns]\n";
    return oss.str();
}

void DataFrame::print(size_t max_rows) const {
    std::cout << to_string(max_rows) << std::endl;
}

void DataFrame::save(const std::string& filename) const {
    std::ofstream os(filename, std::ios::binary);
    size_t nc = ncols(), nr = nrows();
    os.write(reinterpret_cast<const char*>(&nc), sizeof(nc));
    os.write(reinterpret_cast<const char*>(&nr), sizeof(nr));
    for (const auto& c : columns) {
        size_t name_len = c.name.size();
        os.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        os.write(c.name.c_str(), name_len);
        int dt = static_cast<int>(c.dtype);
        os.write(reinterpret_cast<const char*>(&dt), sizeof(dt));
        os.write(reinterpret_cast<const char*>(c.null_mask.data()), nr); // approximate
        if (c.dtype == DType::FLOAT) {
            os.write(reinterpret_cast<const char*>(c.fdata.data()), nr * sizeof(float));
        } else if (c.dtype == DType::INT) {
            os.write(reinterpret_cast<const char*>(c.idata.data()), nr * sizeof(int));
        } else {
            for (const auto& s : c.sdata) {
                size_t len = s.size();
                os.write(reinterpret_cast<const char*>(&len), sizeof(len));
                os.write(s.c_str(), len);
            }
        }
    }
}

DataFrame DataFrame::load(const std::string& filename) {
    DataFrame out;
    std::ifstream is(filename, std::ios::binary);
    size_t nc, nr;
    is.read(reinterpret_cast<char*>(&nc), sizeof(nc));
    is.read(reinterpret_cast<char*>(&nr), sizeof(nr));
    for (size_t ci = 0; ci < nc; ++ci) {
        size_t name_len;
        is.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        is.read(&name[0], name_len);
        int dt;
        is.read(reinterpret_cast<char*>(&dt), sizeof(dt));
        Column c(name, static_cast<DType>(dt));
        c.null_mask.resize(nr);
        is.read(reinterpret_cast<char*>(c.null_mask.data()), nr);
        if (c.dtype == DType::FLOAT) {
            c.fdata.resize(nr);
            is.read(reinterpret_cast<char*>(c.fdata.data()), nr * sizeof(float));
        } else if (c.dtype == DType::INT) {
            c.idata.resize(nr);
            is.read(reinterpret_cast<char*>(c.idata.data()), nr * sizeof(int));
        } else {
            c.sdata.resize(nr);
            for (size_t i = 0; i < nr; ++i) {
                size_t len;
                is.read(reinterpret_cast<char*>(&len), sizeof(len));
                c.sdata[i].resize(len);
                is.read(&c.sdata[i][0], len);
            }
        }
        out.add_column(c);
    }
    return out;
}

} // namespace nn
