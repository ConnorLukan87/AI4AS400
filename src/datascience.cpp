// datascience.cpp - Part 1: stats, metrics, preprocess
#include "datascience.h"

namespace nn {

// ============================================================
// stats namespace
// ============================================================
namespace stats {

float mean(const Tensor1D& x) {
    if (x.empty()) return 0.0f;
    float s = 0.0f;
    for (float v : x) s += v;
    return s / x.size();
}

float variance(const Tensor1D& x, bool sample) {
    if (x.size() <= 1) return 0.0f;
    float m = mean(x);
    float s = 0.0f;
    for (float v : x) s += (v - m) * (v - m);
    return s / (sample ? (x.size() - 1) : x.size());
}

float stddev(const Tensor1D& x, bool sample) { return std::sqrt(variance(x, sample)); }

float covariance(const Tensor1D& x, const Tensor1D& y) {
    int n = std::min(x.size(), y.size());
    if (n <= 1) return 0.0f;
    float mx = mean(x), my = mean(y), s = 0.0f;
    for (int i = 0; i < n; ++i) s += (x[i] - mx) * (y[i] - my);
    return s / (n - 1);
}

float pearson_r(const Tensor1D& x, const Tensor1D& y) {
    float cov = covariance(x, y);
    float sx = stddev(x), sy = stddev(y);
    return (sx * sy == 0.0f) ? 0.0f : cov / (sx * sy);
}

Tensor1D ranks(const Tensor1D& x) {
    int n = x.size();
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&x](size_t a, size_t b) { return x[a] < x[b]; });
    Tensor1D r(n);
    for (int i = 0; i < n; ++i) r[idx[i]] = static_cast<float>(i + 1);
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n - 1 && x[idx[j]] == x[idx[j + 1]]) ++j;
        if (j > i) {
            float avg = 0.0f;
            for (int k = i; k <= j; ++k) avg += r[idx[k]];
            avg /= (j - i + 1);
            for (int k = i; k <= j; ++k) r[idx[k]] = avg;
        }
        i = j + 1;
    }
    return r;
}

float spearman_r(const Tensor1D& x, const Tensor1D& y) { return pearson_r(ranks(x), ranks(y)); }

float median(const Tensor1D& x) {
    if (x.empty()) return 0.0f;
    Tensor1D sorted = x;
    std::sort(sorted.begin(), sorted.end());
    int n = sorted.size();
    return (n % 2 == 0) ? (sorted[n/2-1] + sorted[n/2]) / 2.0f : sorted[n/2];
}

float percentile(const Tensor1D& x, float p) {
    if (x.empty()) return 0.0f;
    Tensor1D sorted = x;
    std::sort(sorted.begin(), sorted.end());
    float idx = (p / 100.0f) * (sorted.size() - 1);
    int lo = (int)idx, hi = lo + 1;
    if (hi >= (int)sorted.size()) return sorted.back();
    return sorted[lo] * (1.0f - (idx - lo)) + sorted[hi] * (idx - lo);
}

float iqr(const Tensor1D& x) { return percentile(x, 75.0f) - percentile(x, 25.0f); }

float skewness(const Tensor1D& x) {
    int n = x.size(); if (n < 3) return 0.0f;
    float m = mean(x), sd = stddev(x);
    if (sd == 0.0f) return 0.0f;
    float s = 0.0f;
    for (float v : x) { float d = (v - m) / sd; s += d*d*d; }
    return s * n / ((n-1.0f) * (n-2.0f));
}

float kurtosis(const Tensor1D& x) {
    int n = x.size(); if (n < 4) return 0.0f;
    float m = mean(x), sd = stddev(x);
    if (sd == 0.0f) return 0.0f;
    float s = 0.0f;
    for (float v : x) { float d = (v - m) / sd; s += d*d*d*d; }
    float k = (s * n * (n+1.0f)) / ((n-1.0f)*(n-2.0f)*(n-3.0f));
    return k - 3.0f*(n-1.0f)*(n-1.0f) / ((n-2.0f)*(n-3.0f));
}

float entropy(const Tensor1D& probs) {
    float h = 0.0f;
    for (float p : probs) if (p > 0.0f) h -= p * std::log2(p);
    return h;
}

float mutual_information(const Tensor1D& x, const Tensor1D& y, int bins) {
    int n = std::min(x.size(), y.size());
    if (n == 0) return 0.0f;
    float xmin = *std::min_element(x.begin(), x.end()), xmax = *std::max_element(x.begin(), x.end());
    float ymin = *std::min_element(y.begin(), y.end()), ymax = *std::max_element(y.begin(), y.end());
    float xstep = (xmax - xmin + 1e-10f) / bins, ystep = (ymax - ymin + 1e-10f) / bins;
    Tensor2D joint(bins, Tensor1D(bins, 0.0f));
    for (int i = 0; i < n; ++i) {
        int bx = std::min((int)((x[i]-xmin)/xstep), bins-1);
        int by = std::min((int)((y[i]-ymin)/ystep), bins-1);
        joint[bx][by] += 1.0f / n;
    }
    Tensor1D px(bins, 0.0f), py(bins, 0.0f);
    for (int i = 0; i < bins; ++i) for (int j = 0; j < bins; ++j) { px[i] += joint[i][j]; py[j] += joint[i][j]; }
    float mi = 0.0f;
    for (int i = 0; i < bins; ++i) for (int j = 0; j < bins; ++j)
        if (joint[i][j] > 0 && px[i] > 0 && py[j] > 0)
            mi += joint[i][j] * std::log2(joint[i][j] / (px[i] * py[j]));
    return mi;
}

Tensor2D correlation_matrix(const Tensor2D& data) {
    if (data.empty()) return {};
    int n = data.size(), p = data[0].size();
    std::vector<Tensor1D> cols(p, Tensor1D(n));
    for (int i = 0; i < n; ++i) for (int j = 0; j < p; ++j) cols[j][i] = data[i][j];
    Tensor2D corr(p, Tensor1D(p, 0.0f));
    for (int i = 0; i < p; ++i) for (int j = i; j < p; ++j) corr[i][j] = corr[j][i] = pearson_r(cols[i], cols[j]);
    return corr;
}

Tensor2D covariance_matrix(const Tensor2D& data) {
    if (data.empty()) return {};
    int n = data.size(), p = data[0].size();
    std::vector<Tensor1D> cols(p, Tensor1D(n));
    for (int i = 0; i < n; ++i) for (int j = 0; j < p; ++j) cols[j][i] = data[i][j];
    Tensor2D cov(p, Tensor1D(p, 0.0f));
    for (int i = 0; i < p; ++i) for (int j = i; j < p; ++j) cov[i][j] = cov[j][i] = covariance(cols[i], cols[j]);
    return cov;
}

bool TestResult::reject_null(float alpha) const { return p_value < alpha; }
std::string TestResult::summary() const {
    std::ostringstream oss;
    oss << test_name << ": stat=" << statistic << ", p=" << p_value
        << " (" << (reject_null() ? "REJECT" : "FAIL TO REJECT") << " H0)";
    return oss.str();
}

float normal_cdf(float x, float mu, float sigma) {
    float z = (x - mu) / sigma;
    float t = 1.0f / (1.0f + 0.2316419f * std::abs(z));
    float d = 0.3989422804f * std::exp(-z*z / 2.0f);
    float p = d * t * (0.3193815f + t * (-0.3565638f + t * (1.781478f + t * (-1.821256f + t * 1.330274f))));
    return z > 0 ? 1.0f - p : p;
}

float normal_pdf(float x, float mu, float sigma) {
    float z = (x - mu) / sigma;
    return std::exp(-0.5f * z * z) / (sigma * std::sqrt(2.0f * 3.14159265f));
}

float normal_quantile(float p, float mu, float sigma) {
    if (p <= 0.0f) return -std::numeric_limits<float>::infinity();
    if (p >= 1.0f) return std::numeric_limits<float>::infinity();
    float t = std::sqrt(-2.0f * std::log(p < 0.5f ? p : 1.0f - p));
    float z = t - (2.515517f + 0.802853f*t + 0.010328f*t*t) / (1.0f + 1.432788f*t + 0.189269f*t*t + 0.001308f*t*t*t);
    return mu + sigma * (p < 0.5f ? -z : z);
}

float t_cdf(float t_val, int df) {
    if (df > 100) return normal_cdf(t_val, 0.0f, 1.0f);
    float a = -10.0f, b = t_val; int steps = 1000;
    float h = (b - a) / steps, nu = df, sum = 0, full = 0;
    for (int i = 0; i <= steps; ++i) {
        float x = a + i * h;
        float fx = std::pow(1.0f + x*x/nu, -(nu+1.0f)/2.0f);
        sum += (i == 0 || i == steps) ? fx : 2.0f * fx;
    }
    sum *= h / 2.0f;
    float h2 = 20.0f / steps;
    for (int i = 0; i <= steps; ++i) {
        float x = -10.0f + i * h2;
        float fx = std::pow(1.0f + x*x/nu, -(nu+1.0f)/2.0f);
        full += (i == 0 || i == steps) ? fx : 2.0f * fx;
    }
    full *= h2 / 2.0f;
    return full > 0 ? sum / full : 0.5f;
}

TestResult t_test_one_sample(const Tensor1D& x, float mu0) {
    int n = x.size(); float m = mean(x), se = stddev(x) / std::sqrt((float)n);
    float t = (m - mu0) / (se + 1e-10f);
    float p = 2.0f * (1.0f - t_cdf(std::abs(t), n-1));
    return {t, std::max(0.0f, std::min(1.0f, p)), "One-sample t-test"};
}

TestResult t_test_two_sample(const Tensor1D& x, const Tensor1D& y, bool equal_var) {
    int n1 = x.size(), n2 = y.size();
    float m1 = mean(x), m2 = mean(y), v1 = variance(x), v2 = variance(y);
    float se; int df;
    if (equal_var) {
        float sp2 = ((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2);
        se = std::sqrt(sp2 * (1.0f/n1 + 1.0f/n2)); df = n1+n2-2;
    } else {
        se = std::sqrt(v1/n1 + v2/n2);
        float num = (v1/n1+v2/n2)*(v1/n1+v2/n2);
        float den = (v1/n1)*(v1/n1)/(n1-1) + (v2/n2)*(v2/n2)/(n2-1);
        df = (int)(num / (den + 1e-10f));
    }
    float t = (m1 - m2) / (se + 1e-10f);
    float p = 2.0f * (1.0f - t_cdf(std::abs(t), df));
    return {t, std::max(0.0f, std::min(1.0f, p)), "Two-sample t-test"};
}

TestResult t_test_paired(const Tensor1D& x, const Tensor1D& y) {
    int n = std::min(x.size(), y.size()); Tensor1D diff(n);
    for (int i = 0; i < n; ++i) diff[i] = x[i] - y[i];
    return t_test_one_sample(diff, 0.0f);
}

TestResult chi_squared_test(const Tensor1D& observed, const Tensor1D& expected) {
    float chi2 = 0.0f;
    for (size_t i = 0; i < observed.size(); ++i) {
        float e = expected[i] + 1e-10f;
        chi2 += (observed[i]-e)*(observed[i]-e) / e;
    }
    int df = observed.size() - 1;
    float z = std::pow(chi2/df, 1.0f/3.0f) - (1.0f - 2.0f/(9.0f*df));
    z /= std::sqrt(2.0f / (9.0f*df));
    return {chi2, std::max(0.0f, std::min(1.0f, 1.0f - normal_cdf(z, 0, 1))), "Chi-squared test"};
}

TestResult anova_one_way(const std::vector<Tensor1D>& groups) {
    int k = groups.size(), N = 0; float grand = 0;
    for (const auto& g : groups) { N += g.size(); for (float v : g) grand += v; }
    grand /= N;
    float ssb = 0, ssw = 0;
    for (const auto& g : groups) {
        float gm = mean(g);
        ssb += g.size() * (gm - grand) * (gm - grand);
        for (float v : g) ssw += (v - gm) * (v - gm);
    }
    float f = (ssb / (k-1)) / (ssw / (N-k) + 1e-10f);
    float p = 1.0f - normal_cdf(std::sqrt(f), 0, 1);
    return {f, std::max(0.0f, std::min(1.0f, p)), "One-way ANOVA"};
}

TestResult mann_whitney_u(const Tensor1D& x, const Tensor1D& y) {
    int n1 = x.size(), n2 = y.size();
    Tensor1D combined; combined.insert(combined.end(), x.begin(), x.end());
    combined.insert(combined.end(), y.begin(), y.end());
    Tensor1D r = ranks(combined);
    float R1 = 0; for (int i = 0; i < n1; ++i) R1 += r[i];
    float U1 = R1 - n1*(n1+1.0f)/2.0f, U = std::min(U1, (float)(n1*n2) - U1);
    float mu_u = n1*n2/2.0f, sig = std::sqrt(n1*n2*(n1+n2+1.0f)/12.0f);
    float z = (U - mu_u) / (sig + 1e-10f);
    return {U, std::max(0.0f, std::min(1.0f, 2.0f * normal_cdf(-std::abs(z), 0, 1))), "Mann-Whitney U"};
}

ConfInterval confidence_interval_mean(const Tensor1D& x, float confidence) {
    float m = mean(x), se = stddev(x) / std::sqrt((float)x.size());
    float z = normal_quantile(0.5f + confidence / 2.0f);
    return {m - z*se, m + z*se, m, confidence};
}

ConfInterval confidence_interval_proportion(int successes, int n, float confidence) {
    float p_hat = (float)successes / n, se = std::sqrt(p_hat*(1-p_hat)/n);
    float z = normal_quantile(0.5f + confidence / 2.0f);
    return {p_hat - z*se, p_hat + z*se, p_hat, confidence};
}

Tensor1D bootstrap_means(const Tensor1D& x, int n_bootstrap) {
    Tensor1D means(n_bootstrap); int n = x.size();
    for (int b = 0; b < n_bootstrap; ++b) {
        float s = 0; for (int i = 0; i < n; ++i) s += x[std::rand() % n];
        means[b] = s / n;
    }
    return means;
}

ConfInterval bootstrap_ci(const Tensor1D& x, std::function<float(const Tensor1D&)> statistic,
                            int n_bootstrap, float confidence) {
    int n = x.size(); Tensor1D stats(n_bootstrap);
    for (int b = 0; b < n_bootstrap; ++b) {
        Tensor1D sample(n); for (int i = 0; i < n; ++i) sample[i] = x[std::rand() % n];
        stats[b] = statistic(sample);
    }
    std::sort(stats.begin(), stats.end());
    float alpha = (1.0f - confidence) / 2.0f;
    return {stats[(int)(alpha*n_bootstrap)], stats[std::min((int)((1-alpha)*n_bootstrap), n_bootstrap-1)], statistic(x), confidence};
}

} // namespace stats

namespace metrics {

float mse(const Tensor1D& yt, const Tensor1D& yp) {
    float s = 0; for (size_t i = 0; i < yt.size(); ++i) { float d = yt[i]-yp[i]; s += d*d; } return s / yt.size();
}
float rmse(const Tensor1D& yt, const Tensor1D& yp) { return std::sqrt(mse(yt, yp)); }
float mae(const Tensor1D& yt, const Tensor1D& yp) {
    float s = 0; for (size_t i = 0; i < yt.size(); ++i) s += std::abs(yt[i]-yp[i]); return s / yt.size();
}
float mape(const Tensor1D& yt, const Tensor1D& yp) {
    float s = 0; int c = 0;
    for (size_t i = 0; i < yt.size(); ++i) if (yt[i] != 0) { s += std::abs((yt[i]-yp[i])/yt[i]); ++c; }
    return c > 0 ? s / c * 100 : 0;
}
float r_squared(const Tensor1D& yt, const Tensor1D& yp) {
    float m = stats::mean(yt), ssr = 0, sst = 0;
    for (size_t i = 0; i < yt.size(); ++i) { ssr += (yt[i]-yp[i])*(yt[i]-yp[i]); sst += (yt[i]-m)*(yt[i]-m); }
    return sst > 0 ? 1.0f - ssr/sst : 0;
}
float adjusted_r_squared(const Tensor1D& yt, const Tensor1D& yp, int nf) {
    float r2 = r_squared(yt, yp); int n = yt.size();
    return 1.0f - (1.0f-r2)*(n-1.0f)/(n-nf-1.0f);
}
float explained_variance(const Tensor1D& yt, const Tensor1D& yp) {
    Tensor1D res(yt.size()); for (size_t i = 0; i < yt.size(); ++i) res[i] = yt[i]-yp[i];
    return 1.0f - stats::variance(res, false) / (stats::variance(yt, false) + 1e-10f);
}
float max_error(const Tensor1D& yt, const Tensor1D& yp) {
    float mx = 0; for (size_t i = 0; i < yt.size(); ++i) mx = std::max(mx, std::abs(yt[i]-yp[i])); return mx;
}
float median_absolute_error(const Tensor1D& yt, const Tensor1D& yp) {
    Tensor1D e(yt.size()); for (size_t i = 0; i < yt.size(); ++i) e[i] = std::abs(yt[i]-yp[i]);
    return stats::median(e);
}

int ConfusionMatrix::total() const { return tp+fp+tn+fn; }
float ConfusionMatrix::accuracy() const { return (float)(tp+tn) / (total()+1e-10f); }
float ConfusionMatrix::precision() const { return (float)tp / (tp+fp+1e-10f); }
float ConfusionMatrix::recall() const { return (float)tp / (tp+fn+1e-10f); }
float ConfusionMatrix::f1() const { float p = precision(), r = recall(); return 2*p*r / (p+r+1e-10f); }
float ConfusionMatrix::specificity() const { return (float)tn / (tn+fp+1e-10f); }
float ConfusionMatrix::npv() const { return (float)tn / (tn+fn+1e-10f); }
float ConfusionMatrix::mcc() const {
    float num = (float)(tp*tn - fp*fn);
    float den = std::sqrt((float)((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)));
    return den > 0 ? num/den : 0;
}
std::string ConfusionMatrix::to_string() const {
    std::ostringstream o;
    o << "CM: TP=" << tp << " FP=" << fp << " FN=" << fn << " TN=" << tn
      << " | Acc=" << accuracy() << " P=" << precision() << " R=" << recall()
      << " F1=" << f1() << " MCC=" << mcc();
    return o.str();
}

ConfusionMatrix confusion_matrix(const Tensor1D& yt, const Tensor1D& yp, float thr) {
    ConfusionMatrix cm{0,0,0,0};
    for (size_t i = 0; i < yt.size(); ++i) {
        bool a = yt[i] >= 0.5f, p = yp[i] >= thr;
        if (a && p) cm.tp++; else if (!a && p) cm.fp++; else if (a && !p) cm.fn++; else cm.tn++;
    }
    return cm;
}
float accuracy(const Tensor1D& yt, const Tensor1D& yp, float t) { return confusion_matrix(yt,yp,t).accuracy(); }
float precision(const Tensor1D& yt, const Tensor1D& yp, float t) { return confusion_matrix(yt,yp,t).precision(); }
float recall(const Tensor1D& yt, const Tensor1D& yp, float t) { return confusion_matrix(yt,yp,t).recall(); }
float f1_score(const Tensor1D& yt, const Tensor1D& yp, float t) { return confusion_matrix(yt,yp,t).f1(); }
float fbeta_score(const Tensor1D& yt, const Tensor1D& yp, float beta, float t) {
    auto cm = confusion_matrix(yt,yp,t); float p = cm.precision(), r = cm.recall(), b2 = beta*beta;
    return (1+b2)*p*r / (b2*p + r + 1e-10f);
}
float log_loss(const Tensor1D& yt, const Tensor1D& yp) {
    float s = 0;
    for (size_t i = 0; i < yt.size(); ++i) {
        float p = std::max(1e-7f, std::min(1-1e-7f, yp[i]));
        s += yt[i]*std::log(p) + (1-yt[i])*std::log(1-p);
    }
    return -s / yt.size();
}
float cohen_kappa(const Tensor1D& yt, const Tensor1D& yp, float t) {
    auto cm = confusion_matrix(yt,yp,t); float n = cm.total(), po = cm.accuracy();
    float pe = ((cm.tp+cm.fp)*(cm.tp+cm.fn) + (cm.tn+cm.fp)*(cm.tn+cm.fn)) / (n*n);
    return (po - pe) / (1 - pe + 1e-10f);
}

std::vector<ROCPoint> roc_curve(const Tensor1D& yt, const Tensor1D& ys) {
    std::vector<size_t> idx(yt.size()); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&ys](size_t a, size_t b) { return ys[a] > ys[b]; });
    int P = 0, N = 0; for (float v : yt) { if (v >= 0.5f) P++; else N++; }
    std::vector<ROCPoint> curve; curve.push_back({0,0,1.1f});
    int tp = 0, fp = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (yt[idx[i]] >= 0.5f) tp++; else fp++;
        curve.push_back({(float)fp/(N+1e-10f), (float)tp/(P+1e-10f), ys[idx[i]]});
    }
    return curve;
}
float auc_roc(const Tensor1D& yt, const Tensor1D& ys) {
    auto c = roc_curve(yt, ys); float auc = 0;
    for (size_t i = 1; i < c.size(); ++i) auc += (c[i].fpr - c[i-1].fpr) * (c[i].tpr + c[i-1].tpr) / 2;
    return auc;
}

std::vector<PRPoint> pr_curve(const Tensor1D& yt, const Tensor1D& ys) {
    std::vector<size_t> idx(yt.size()); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&ys](size_t a, size_t b) { return ys[a] > ys[b]; });
    int tp_count = 0; for (float v : yt) if (v >= 0.5f) tp_count++;
    std::vector<PRPoint> curve; int tp = 0, fp = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (yt[idx[i]] >= 0.5f) tp++; else fp++;
        curve.push_back({(float)tp/(tp+fp), (float)tp/(tp_count+1e-10f), ys[idx[i]]});
    }
    return curve;
}
float auc_pr(const Tensor1D& yt, const Tensor1D& ys) {
    auto c = pr_curve(yt, ys); float auc = 0;
    for (size_t i = 1; i < c.size(); ++i) auc += (c[i].recall - c[i-1].recall) * (c[i].precision + c[i-1].precision) / 2;
    return std::abs(auc);
}
float average_precision(const Tensor1D& yt, const Tensor1D& ys) { return auc_pr(yt, ys); }

ClassReport classification_report(const std::vector<int>& yt, const std::vector<int>& yp) {
    std::set<int> cs(yt.begin(), yt.end());
    ClassReport r; r.classes.assign(cs.begin(), cs.end());
    int nc = r.classes.size();
    r.precisions.resize(nc); r.recalls.resize(nc); r.f1s.resize(nc); r.supports.resize(nc);
    for (int c = 0; c < nc; ++c) {
        int cls = r.classes[c], tp = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < yt.size(); ++i) {
            if (yp[i]==cls && yt[i]==cls) tp++; else if (yp[i]==cls) fp++; else if (yt[i]==cls) fn++;
        }
        r.supports[c] = tp+fn;
        r.precisions[c] = (float)tp / (tp+fp+1e-10f);
        r.recalls[c] = (float)tp / (tp+fn+1e-10f);
        r.f1s[c] = 2*r.precisions[c]*r.recalls[c] / (r.precisions[c]+r.recalls[c]+1e-10f);
    }
    r.macro_precision = r.macro_recall = r.macro_f1 = 0; r.weighted_f1 = 0; float ts = 0;
    for (int c = 0; c < nc; ++c) {
        r.macro_precision += r.precisions[c]; r.macro_recall += r.recalls[c]; r.macro_f1 += r.f1s[c];
        r.weighted_f1 += r.f1s[c] * r.supports[c]; ts += r.supports[c];
    }
    r.macro_precision /= nc; r.macro_recall /= nc; r.macro_f1 /= nc; r.weighted_f1 /= (ts+1e-10f);
    return r;
}
std::string ClassReport::to_string() const {
    std::ostringstream o;
    o << std::left << std::setw(10) << "Class" << std::setw(12) << "Prec" << std::setw(12) << "Rec" << std::setw(12) << "F1" << "Sup\n";
    for (size_t i = 0; i < classes.size(); ++i)
        o << std::setw(10) << classes[i] << std::setw(12) << precisions[i] << std::setw(12) << recalls[i] << std::setw(12) << f1s[i] << supports[i] << "\n";
    o << "Macro F1: " << macro_f1 << "  Weighted F1: " << weighted_f1 << "\n";
    return o.str();
}

Tensor2D confusion_matrix_multi(const std::vector<int>& yt, const std::vector<int>& yp, int nc) {
    Tensor2D cm(nc, Tensor1D(nc, 0)); for (size_t i = 0; i < yt.size(); ++i) if (yt[i]>=0 && yt[i]<nc && yp[i]>=0 && yp[i]<nc) cm[yt[i]][yp[i]]+=1;
    return cm;
}

float ndcg(const Tensor1D& rel, int k) {
    if (k < 0) k = rel.size(); k = std::min(k, (int)rel.size());
    float dcg = 0; for (int i = 0; i < k; ++i) dcg += (std::pow(2,rel[i])-1) / std::log2(i+2.0f);
    Tensor1D s = rel; std::sort(s.begin(), s.end(), std::greater<float>());
    float idcg = 0; for (int i = 0; i < k; ++i) idcg += (std::pow(2,s[i])-1) / std::log2(i+2.0f);
    return idcg > 0 ? dcg/idcg : 0;
}

float silhouette_score(const Tensor2D& X, const std::vector<int>& labels) {
    int n = X.size(); if (n <= 1) return 0;
    float total = 0;
    for (int i = 0; i < n; ++i) {
        float a = 0; int ac = 0;
        float b_min = std::numeric_limits<float>::infinity();
        std::map<int, std::pair<float,int>> cd;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            float d = 0; for (size_t k = 0; k < X[i].size(); ++k) { float df = X[i][k]-X[j][k]; d += df*df; }
            d = std::sqrt(d);
            if (labels[j] == labels[i]) { a += d; ac++; } else { cd[labels[j]].first += d; cd[labels[j]].second++; }
        }
        a = ac > 0 ? a/ac : 0;
        for (const auto& [cls, p] : cd) b_min = std::min(b_min, p.first/p.second);
        total += (b_min - a) / (std::max(a, b_min) + 1e-10f);
    }
    return total / n;
}

} // namespace metrics

namespace preprocess {

ScalerParams fit_standard_scaler(const std::vector<Tensor1D>& X) {
    if (X.empty()) return {};
    int p = X[0].size(), n = X.size();
    ScalerParams sp; sp.means.resize(p, 0); sp.stds.resize(p, 0);
    for (int j = 0; j < p; ++j) { for (int i = 0; i < n; ++i) sp.means[j] += X[i][j]; sp.means[j] /= n; }
    for (int j = 0; j < p; ++j) { float s = 0; for (int i = 0; i < n; ++i) { float d = X[i][j]-sp.means[j]; s += d*d; } sp.stds[j] = std::sqrt(s/(n-1)); if (sp.stds[j]==0) sp.stds[j]=1; }
    return sp;
}
std::vector<Tensor1D> transform_standard(const std::vector<Tensor1D>& X, const ScalerParams& p) {
    auto o = X; for (auto& r : o) for (size_t j = 0; j < r.size(); ++j) r[j] = (r[j]-p.means[j])/p.stds[j]; return o;
}
std::vector<Tensor1D> inverse_standard(const std::vector<Tensor1D>& X, const ScalerParams& p) {
    auto o = X; for (auto& r : o) for (size_t j = 0; j < r.size(); ++j) r[j] = r[j]*p.stds[j]+p.means[j]; return o;
}
ScalerParams fit_minmax_scaler(const std::vector<Tensor1D>& X) {
    if (X.empty()) return {};
    int p = X[0].size(); ScalerParams sp;
    sp.mins.assign(p, std::numeric_limits<float>::infinity()); sp.maxs.assign(p, -std::numeric_limits<float>::infinity());
    for (const auto& r : X) for (int j = 0; j < p; ++j) { sp.mins[j] = std::min(sp.mins[j], r[j]); sp.maxs[j] = std::max(sp.maxs[j], r[j]); }
    return sp;
}
std::vector<Tensor1D> transform_minmax(const std::vector<Tensor1D>& X, const ScalerParams& p) {
    auto o = X; for (auto& r : o) for (size_t j = 0; j < r.size(); ++j) { float rng = p.maxs[j]-p.mins[j]; if (rng==0) rng=1; r[j] = (r[j]-p.mins[j])/rng; } return o;
}
std::vector<Tensor1D> inverse_minmax(const std::vector<Tensor1D>& X, const ScalerParams& p) {
    auto o = X; for (auto& r : o) for (size_t j = 0; j < r.size(); ++j) r[j] = r[j]*(p.maxs[j]-p.mins[j])+p.mins[j]; return o;
}
LabelEncoderState fit_label_encoder(const std::vector<std::string>& labels) {
    LabelEncoderState s; int id = 0;
    for (const auto& l : labels) if (s.mapping.find(l)==s.mapping.end()) { s.mapping[l]=id; s.inverse_mapping[id]=l; ++id; }
    return s;
}
std::vector<int> transform_labels(const std::vector<std::string>& labels, const LabelEncoderState& s) {
    std::vector<int> o; for (const auto& l : labels) { auto it = s.mapping.find(l); o.push_back(it!=s.mapping.end() ? it->second : -1); } return o;
}
Tensor2D one_hot_encode(const std::vector<int>& labels, int nc) {
    if (nc < 0) nc = *std::max_element(labels.begin(), labels.end()) + 1;
    Tensor2D o(labels.size(), Tensor1D(nc, 0)); for (size_t i = 0; i < labels.size(); ++i) if (labels[i]>=0 && labels[i]<nc) o[i][labels[i]]=1; return o;
}
Tensor2D polynomial_features(const Tensor2D& X, int degree, bool interaction_only) {
    if (X.empty()) return {};
    int n = X.size(), p = X[0].size(); Tensor2D out;
    for (int i = 0; i < n; ++i) {
        Tensor1D row; for (int j = 0; j < p; ++j) row.push_back(X[i][j]);
        if (degree >= 2) for (int j = 0; j < p; ++j) for (int k = j; k < p; ++k) { if (interaction_only && j==k) continue; row.push_back(X[i][j]*X[i][k]); }
        if (degree >= 3 && !interaction_only) for (int j = 0; j < p; ++j) row.push_back(X[i][j]*X[i][j]*X[i][j]);
        out.push_back(row);
    }
    return out;
}
Tensor1D equal_width_bins(const Tensor1D& x, int nb) {
    float mn = *std::min_element(x.begin(),x.end()), mx = *std::max_element(x.begin(),x.end()), step = (mx-mn+1e-10f)/nb;
    Tensor1D o(x.size()); for (size_t i = 0; i < x.size(); ++i) o[i] = std::min((float)(int)((x[i]-mn)/step), (float)(nb-1)); return o;
}
SplitResult train_test_split(const std::vector<Tensor1D>& X, const std::vector<Tensor1D>& y, float test_ratio, bool shuffle, int seed) {
    int n = X.size(); std::vector<size_t> idx(n); std::iota(idx.begin(), idx.end(), 0);
    if (shuffle) { if (seed >= 0) std::srand(seed); for (int i = n-1; i > 0; --i) { int j = std::rand()%(i+1); std::swap(idx[i], idx[j]); } }
    int tn = (int)(n * test_ratio); SplitResult r;
    for (int i = 0; i < n-tn; ++i) { r.X_train.push_back(X[idx[i]]); r.y_train.push_back(y[idx[i]]); r.train_indices.push_back(idx[i]); }
    for (int i = n-tn; i < n; ++i) { r.X_test.push_back(X[idx[i]]); r.y_test.push_back(y[idx[i]]); r.test_indices.push_back(idx[i]); }
    return r;
}
ResampleResult smote(const std::vector<Tensor1D>& X, const std::vector<Tensor1D>& y, int k) {
    std::map<float, std::vector<size_t>> ci;
    for (size_t i = 0; i < y.size(); ++i) ci[y[i][0]].push_back(i);
    float min_cls = 0; size_t min_c = y.size(), max_c = 0;
    for (const auto& [c, idx] : ci) { if (idx.size() < min_c) { min_c = idx.size(); min_cls = c; } max_c = std::max(max_c, idx.size()); }
    ResampleResult r; r.X = X; r.y = y;
    auto& minority = ci[min_cls]; int ns = max_c - min_c;
    for (int s = 0; s < ns; ++s) {
        size_t idx = minority[std::rand() % minority.size()];
        std::vector<std::pair<float,size_t>> dists;
        for (size_t j : minority) { if (j==idx) continue; float d=0; for (size_t f=0;f<X[idx].size();++f){float df=X[idx][f]-X[j][f];d+=df*df;} dists.push_back({d,j}); }
        std::sort(dists.begin(), dists.end());
        int nn = dists[std::rand() % std::min(k,(int)dists.size())].second;
        float lam = (float)std::rand()/RAND_MAX; Tensor1D syn(X[idx].size());
        for (size_t f = 0; f < X[idx].size(); ++f) syn[f] = X[idx][f] + lam*(X[nn][f]-X[idx][f]);
        r.X.push_back(syn); r.y.push_back({min_cls});
    }
    return r;
}
} // namespace preprocess

namespace feature_selection {
std::vector<int> variance_threshold(const std::vector<Tensor1D>& X, float threshold) {
    if (X.empty()) return {};
    int p = X[0].size(); std::vector<int> sel;
    for (int j = 0; j < p; ++j) { Tensor1D c(X.size()); for (size_t i=0;i<X.size();++i) c[i]=X[i][j]; if (stats::variance(c) > threshold) sel.push_back(j); }
    return sel;
}
std::vector<int> remove_correlated(const std::vector<Tensor1D>& X, float threshold) {
    if (X.empty()) return {};
    int p = X[0].size(), n = X.size();
    std::vector<Tensor1D> cols(p, Tensor1D(n)); for (int i=0;i<n;++i) for (int j=0;j<p;++j) cols[j][i]=X[i][j];
    std::vector<bool> dropped(p, false);
    for (int i=0;i<p;++i) { if (dropped[i]) continue; for (int j=i+1;j<p;++j) { if (dropped[j]) continue; if (std::abs(stats::pearson_r(cols[i],cols[j])) > threshold) dropped[j]=true; } }
    std::vector<int> sel; for (int i=0;i<p;++i) if (!dropped[i]) sel.push_back(i); return sel;
}
} // namespace feature_selection

namespace cv {
std::vector<Fold> k_fold(size_t ns, int k, bool shuffle, int seed) {
    std::vector<size_t> idx(ns); std::iota(idx.begin(),idx.end(),0);
    if (shuffle) { if (seed>=0) std::srand(seed); for (int i=ns-1;i>0;--i) { int j=std::rand()%(i+1); std::swap(idx[i],idx[j]); } }
    std::vector<Fold> folds(k); size_t fs = ns / k;
    for (int f=0;f<k;++f) { size_t s=f*fs, e=(f==k-1)?ns:(f+1)*fs; for (size_t i=0;i<ns;++i) { if(i>=s&&i<e) folds[f].val_indices.push_back(idx[i]); else folds[f].train_indices.push_back(idx[i]); } }
    return folds;
}
std::vector<Fold> time_series_split(size_t ns, int n_splits, int gap) {
    std::vector<Fold> folds; size_t ts = ns / (n_splits+1);
    for (int i=0;i<n_splits;++i) { Fold f; size_t te=(i+1)*ts, tss=te+gap, tse=std::min(tss+ts,ns);
        for (size_t j=0;j<te;++j) f.train_indices.push_back(j); for (size_t j=tss;j<tse;++j) f.val_indices.push_back(j);
        if (!f.val_indices.empty()) folds.push_back(f); }
    return folds;
}
std::string CVResult::summary() const {
    std::ostringstream o; o << "CV: Train=" << mean_train << "(+/-" << std_train << ") Val=" << mean_val << "(+/-" << std_val << ")"; return o.str();
}
} // namespace cv

namespace unsupervised {
Tensor2D kmeans_plus_plus_init(const std::vector<Tensor1D>& X, int k, int seed) {
    if (seed >= 0) std::srand(seed);
    int n = X.size(); Tensor2D centroids; centroids.push_back(X[std::rand()%n]);
    for (int c = 1; c < k; ++c) {
        Tensor1D dists(n, std::numeric_limits<float>::infinity());
        for (int i=0;i<n;++i) for (const auto& ct : centroids) { float d=0; for (size_t j=0;j<X[i].size();++j){float df=X[i][j]-ct[j];d+=df*df;} dists[i]=std::min(dists[i],d); }
        float total=0; for (float d : dists) total += d;
        float r = (float)std::rand()/RAND_MAX * total, cum = 0;
        for (int i=0;i<n;++i) { cum += dists[i]; if (cum >= r) { centroids.push_back(X[i]); break; } }
    }
    return centroids;
}

KMeansResult kmeans(const std::vector<Tensor1D>& X, int k, int max_iter, int seed, int n_init) {
    KMeansResult best; float best_in = std::numeric_limits<float>::infinity();
    for (int init=0;init<n_init;++init) {
        Tensor2D cent = kmeans_plus_plus_init(X, k, seed>=0?seed+init:-1);
        int n = X.size(), p = X[0].size(); std::vector<int> lab(n,0); Tensor1D ih;
        for (int iter=0;iter<max_iter;++iter) {
            float inertia = 0;
            for (int i=0;i<n;++i) { float md=std::numeric_limits<float>::infinity(); for (int c=0;c<k;++c) { float d=0; for (int j=0;j<p;++j){float df=X[i][j]-cent[c][j];d+=df*df;} if (d<md){md=d;lab[i]=c;} } inertia+=md; }
            ih.push_back(inertia);
            Tensor2D nc(k,Tensor1D(p,0)); std::vector<int> cnt(k,0);
            for (int i=0;i<n;++i) { for (int j=0;j<p;++j) nc[lab[i]][j]+=X[i][j]; cnt[lab[i]]++; }
            bool conv = true;
            for (int c=0;c<k;++c) if (cnt[c]>0) for (int j=0;j<p;++j) { float nv=nc[c][j]/cnt[c]; if (std::abs(nv-cent[c][j])>1e-6f) conv=false; cent[c][j]=nv; }
            if (conv) break;
        }
        if (ih.back() < best_in) { best_in=ih.back(); best={cent,lab,ih,(int)ih.size()}; }
    }
    return best;
}

Tensor1D elbow_inertias(const std::vector<Tensor1D>& X, int max_k) {
    Tensor1D in; for (int k=1;k<=max_k;++k) { auto r=kmeans(X,k,100,42,3); in.push_back(r.inertia_history.back()); } return in;
}

DBSCANResult dbscan(const std::vector<Tensor1D>& X, float eps, int min_samples) {
    int n = X.size(); std::vector<int> lab(n,-1); std::vector<bool> vis(n,false); int cid = 0;
    auto rq = [&](int idx) { std::vector<int> nb; for (int j=0;j<n;++j) { float d=0; for (size_t k=0;k<X[idx].size();++k){float df=X[idx][k]-X[j][k];d+=df*df;} if (std::sqrt(d)<=eps) nb.push_back(j); } return nb; };
    for (int i=0;i<n;++i) {
        if (vis[i]) continue; vis[i]=true;
        auto nb = rq(i); if ((int)nb.size() < min_samples) continue;
        lab[i]=cid; std::vector<int> ss = nb;
        for (size_t j=0;j<ss.size();++j) { int q=ss[j]; if (!vis[q]) { vis[q]=true; auto qn=rq(q); if ((int)qn.size()>=min_samples) for (int nb2 : qn) ss.push_back(nb2); } if (lab[q]==-1) lab[q]=cid; }
        cid++;
    }
    int nn=0; for (int l : lab) if (l==-1) nn++;
    return {lab,cid,nn};
}

PCAResult fit_pca(const std::vector<Tensor1D>& X, int nc) {
    int n = X.size(), p = X[0].size();
    if (nc < 0) nc = std::min(n,p); nc = std::min(nc, std::min(n,p));
    Tensor1D mv(p,0); for (int i=0;i<n;++i) for (int j=0;j<p;++j) mv[j]+=X[i][j]; for (int j=0;j<p;++j) mv[j]/=n;
    Tensor2D cen(n,Tensor1D(p)); for (int i=0;i<n;++i) for (int j=0;j<p;++j) cen[i][j]=X[i][j]-mv[j];
    Tensor2D cov = stats::covariance_matrix(cen);
    PCAResult r; r.mean=mv; r.n_components=nc; r.components.resize(nc,Tensor1D(p)); r.explained_variance.resize(nc); r.singular_values.resize(nc);
    Tensor2D A = cov;
    for (int comp=0;comp<nc;++comp) {
        Tensor1D v(p); for (int j=0;j<p;++j) v[j]=(float)std::rand()/RAND_MAX;
        for (int iter=0;iter<200;++iter) {
            Tensor1D Av(p,0); for (int i=0;i<p;++i) for (int j=0;j<p;++j) Av[i]+=A[i][j]*v[j];
            float nm=0; for (float val : Av) nm+=val*val; nm=std::sqrt(nm); if (nm>0) for (auto& val : Av) val/=nm; v=Av;
        }
        Tensor1D Av(p,0); for (int i=0;i<p;++i) for (int j=0;j<p;++j) Av[i]+=A[i][j]*v[j];
        float ev=0; for (int i=0;i<p;++i) ev+=v[i]*Av[i];
        r.components[comp]=v; r.explained_variance[comp]=ev; r.singular_values[comp]=std::sqrt(std::max(0.0f,ev*(n-1)));
        for (int i=0;i<p;++i) for (int j=0;j<p;++j) A[i][j]-=ev*v[i]*v[j];
    }
    float tv=0; for (int i=0;i<p;++i) tv+=cov[i][i];
    r.explained_ratio.resize(nc); for (int i=0;i<nc;++i) r.explained_ratio[i]=r.explained_variance[i]/(tv+1e-10f);
    return r;
}
std::vector<Tensor1D> transform_pca(const std::vector<Tensor1D>& X, const PCAResult& pca) {
    int n=X.size(),p=X[0].size(); std::vector<Tensor1D> o(n,Tensor1D(pca.n_components));
    for (int i=0;i<n;++i) for (int c=0;c<pca.n_components;++c) { float s=0; for (int j=0;j<p;++j) s+=(X[i][j]-pca.mean[j])*pca.components[c][j]; o[i][c]=s; }
    return o;
}
std::vector<Tensor1D> inverse_pca(const std::vector<Tensor1D>& Xt, const PCAResult& pca) {
    int n=Xt.size(),p=pca.mean.size(); std::vector<Tensor1D> o(n,Tensor1D(p,0));
    for (int i=0;i<n;++i) for (int j=0;j<p;++j) { o[i][j]=pca.mean[j]; for (int c=0;c<pca.n_components;++c) o[i][j]+=Xt[i][c]*pca.components[c][j]; }
    return o;
}
} // namespace unsupervised

namespace timeseries {
Tensor1D simple_moving_average(const Tensor1D& x, int w) {
    Tensor1D o(x.size(),0); for (size_t i=0;i<x.size();++i) { if ((int)i<w-1){o[i]=x[i];continue;} float s=0; for (int j=0;j<w;++j) s+=x[i-j]; o[i]=s/w; } return o;
}
Tensor1D exponential_moving_average(const Tensor1D& x, float a) {
    Tensor1D o(x.size()); o[0]=x[0]; for (size_t i=1;i<x.size();++i) o[i]=a*x[i]+(1-a)*o[i-1]; return o;
}
Tensor1D difference(const Tensor1D& x, int d) {
    Tensor1D o=x; for (int dd=0;dd<d;++dd) { Tensor1D n(o.size()-1); for (size_t i=0;i<n.size();++i) n[i]=o[i+1]-o[i]; o=n; } return o;
}
Tensor1D autocorrelation(const Tensor1D& x, int ml) {
    int n=x.size(); if (ml<0) ml=n/2; float m=stats::mean(x),var=0; for (float v : x) var+=(v-m)*(v-m);
    Tensor1D acf(ml+1); for (int lag=0;lag<=ml;++lag) { float s=0; for (int i=0;i<n-lag;++i) s+=(x[i]-m)*(x[i+lag]-m); acf[lag]=var>0?s/var:0; } return acf;
}
DecompResult seasonal_decompose(const Tensor1D& x, int period, const std::string& model) {
    DecompResult r; r.period=period; int n=x.size();
    r.trend = simple_moving_average(x, period);
    Tensor1D det(n); for (int i=0;i<n;++i) det[i] = (model=="multiplicative") ? (r.trend[i]!=0?x[i]/r.trend[i]:1) : x[i]-r.trend[i];
    r.seasonal.resize(n,0);
    for (int p=0;p<period;++p) { float s=0;int c=0; for (int i=p;i<n;i+=period){s+=det[i];c++;} float avg=s/c; for (int i=p;i<n;i+=period) r.seasonal[i]=avg; }
    r.residual.resize(n); for (int i=0;i<n;++i) r.residual[i] = (model=="multiplicative") ? ((r.trend[i]*r.seasonal[i])!=0?x[i]/(r.trend[i]*r.seasonal[i]):1) : x[i]-r.trend[i]-r.seasonal[i];
    return r;
}
std::vector<ChangePoint> detect_change_points(const Tensor1D& x, int ms, float pen) {
    std::vector<ChangePoint> pts; int n=x.size();
    for (int i=ms;i<n-ms;++i) {
        Tensor1D l(x.begin(),x.begin()+i), r(x.begin()+i,x.end());
        float score = stats::variance(x,false)*n - stats::variance(l,false)*l.size() - stats::variance(r,false)*r.size() - pen;
        if (score > 0) pts.push_back({i, score});
    }
    std::sort(pts.begin(),pts.end(),[](const ChangePoint& a, const ChangePoint& b){return a.score>b.score;});
    return pts;
}
TSFeatures extract_features(const Tensor1D& x, int period) {
    TSFeatures f; f.mean=stats::mean(x); f.std=stats::stddev(x);
    f.min=*std::min_element(x.begin(),x.end()); f.max=*std::max_element(x.begin(),x.end());
    f.median=stats::median(x); f.skewness=stats::skewness(x); f.kurtosis=stats::kurtosis(x);
    auto acf=autocorrelation(x,std::min(5,(int)x.size()/2));
    f.autocorr_lag1=acf.size()>1?acf[1]:0; f.autocorr_lag5=acf.size()>5?acf[5]:0;
    f.n_crossings=0; for (size_t i=1;i<x.size();++i) if ((x[i]>=f.mean)!=(x[i-1]>=f.mean)) f.n_crossings++;
    f.longest_streak_above_mean=0; int cur=0; for (float v : x) { if (v>f.mean) cur++; else { f.longest_streak_above_mean=std::max(f.longest_streak_above_mean,(float)cur); cur=0; } }
    f.longest_streak_above_mean=std::max(f.longest_streak_above_mean,(float)cur);
    f.trend_strength=0; f.seasonal_strength=0; f.entropy=0;
    Tensor1D hist(10,0); float range=f.max-f.min+1e-10f;
    for (float v : x) { int bin=std::min(9,(int)((v-f.min)/range*10)); hist[bin]+=1.0f/x.size(); }
    f.entropy=stats::entropy(hist);
    return f;
}
} // namespace timeseries

namespace linear_models {
LinearResult ols_regression(const std::vector<Tensor1D>& X, const Tensor1D& y) {
    int n=X.size(), p=X[0].size();
    Tensor2D Xa(n,Tensor1D(p+1)); for (int i=0;i<n;++i) { for (int j=0;j<p;++j) Xa[i][j]=X[i][j]; Xa[i][p]=1; }
    Tensor2D XtX(p+1,Tensor1D(p+1,0)); Tensor1D Xty(p+1,0);
    for (int i=0;i<p+1;++i) for (int j=0;j<p+1;++j) for (int k=0;k<n;++k) XtX[i][j]+=Xa[k][i]*Xa[k][j];
    for (int i=0;i<p+1;++i) for (int k=0;k<n;++k) Xty[i]+=Xa[k][i]*y[k];
    Tensor2D aug(p+1,Tensor1D(p+2));
    for (int i=0;i<p+1;++i) { for (int j=0;j<p+1;++j) aug[i][j]=XtX[i][j]; aug[i][p+1]=Xty[i]; }
    for (int i=0;i<p+1;++i) {
        int mr=i; for (int k=i+1;k<p+1;++k) if (std::abs(aug[k][i])>std::abs(aug[mr][i])) mr=k; std::swap(aug[i],aug[mr]);
        float piv=aug[i][i]; if (std::abs(piv)<1e-12f) continue;
        for (int j=0;j<p+2;++j) aug[i][j]/=piv;
        for (int k=0;k<p+1;++k) { if (k==i) continue; float f=aug[k][i]; for (int j=0;j<p+2;++j) aug[k][j]-=f*aug[i][j]; }
    }
    LinearResult r; r.n_obs=n; r.n_features=p;
    r.coefficients.resize(p); for (int j=0;j<p;++j) r.coefficients[j]=aug[j][p+1]; r.intercept=aug[p][p+1];
    Tensor1D yp(n); float ssr=0,sst=0,ym=stats::mean(y);
    for (int i=0;i<n;++i) { yp[i]=r.intercept; for (int j=0;j<p;++j) yp[i]+=r.coefficients[j]*X[i][j]; ssr+=(y[i]-yp[i])*(y[i]-yp[i]); sst+=(y[i]-ym)*(y[i]-ym); }
    r.r_squared=1-ssr/(sst+1e-10f); r.adj_r_squared=1-(1-r.r_squared)*(n-1.0f)/(n-p-1.0f);
    float mse_val=ssr/(n-p-1); r.std_errors.resize(p); r.t_values.resize(p); r.p_values.resize(p);
    for (int j=0;j<p;++j) { float se=std::sqrt(std::max(0.0f,mse_val*(XtX[j][j]>0?1.0f/XtX[j][j]:0))); r.std_errors[j]=se; r.t_values[j]=se>0?r.coefficients[j]/se:0; r.p_values[j]=std::max(0.0f,std::min(1.0f,2*(1-stats::t_cdf(std::abs(r.t_values[j]),n-p-1)))); }
    float ll=-n/2.0f*std::log(ssr/n+1e-10f); r.aic=-2*ll+2*(p+1); r.bic=-2*ll+std::log((float)n)*(p+1);
    Tensor1D res(n); for (int i=0;i<n;++i) res[i]=y[i]-yp[i]; float dw_n=0,dw_d=0;
    for (int i=1;i<n;++i) dw_n+=(res[i]-res[i-1])*(res[i]-res[i-1]); for (int i=0;i<n;++i) dw_d+=res[i]*res[i]; r.durbin_watson=dw_d>0?dw_n/dw_d:2; r.f_statistic=0; r.f_p_value=0;
    return r;
}
std::string LinearResult::summary() const {
    std::ostringstream o; o << "=== OLS ===\nR2=" << r_squared << " AdjR2=" << adj_r_squared << " AIC=" << aic << " BIC=" << bic << " DW=" << durbin_watson << "\n";
    for (int j=0;j<n_features;++j) o << "x" << j << ": coef=" << coefficients[j] << " se=" << std_errors[j] << " t=" << t_values[j] << " p=" << p_values[j] << "\n";
    o << "intercept: " << intercept << "\n"; return o.str();
}
LogisticResult logistic_regression(const std::vector<Tensor1D>& X, const std::vector<int>& y, float lr, int max_iter, float reg) {
    int n=X.size(),p=X[0].size(); LogisticResult r; r.coefficients.resize(p,0); r.intercept=0;
    for (int iter=0;iter<max_iter;++iter) {
        Tensor1D gw(p,0); float gb=0;
        for (int i=0;i<n;++i) { float z=r.intercept; for (int j=0;j<p;++j) z+=r.coefficients[j]*X[i][j]; float pred=1/(1+std::exp(-z)),err=pred-y[i]; for (int j=0;j<p;++j) gw[j]+=err*X[i][j]/n+reg*r.coefficients[j]/n; gb+=err/n; }
        for (int j=0;j<p;++j) r.coefficients[j]-=lr*gw[j]; r.intercept-=lr*gb;
    }
    r.n_iterations=max_iter; return r;
}
Tensor1D LogisticResult::predict_proba(const Tensor1D& x) const {
    float z=intercept; for (size_t j=0;j<x.size();++j) z+=coefficients[j]*x[j]; float p=1/(1+std::exp(-z)); return {1-p,p};
}
int LogisticResult::predict(const Tensor1D& x) const { return predict_proba(x)[1]>=0.5f?1:0; }
} // namespace linear_models

namespace knn {
KNNModel fit_classifier(const std::vector<Tensor1D>& X, const std::vector<int>& y, int k) { return {X,y,{},k,true}; }
int predict_class(const KNNModel& m, const Tensor1D& x) {
    std::vector<std::pair<float,int>> d; for (size_t i=0;i<m.X_train.size();++i) { float dd=0; for (size_t j=0;j<x.size();++j){float df=x[j]-m.X_train[i][j];dd+=df*df;} d.push_back({dd,m.y_class[i]}); }
    std::partial_sort(d.begin(),d.begin()+m.k,d.end()); std::map<int,int> v; for (int i=0;i<m.k;++i) v[d[i].second]++;
    int best=-1,bc=0; for (const auto& [c,cnt]:v) if (cnt>bc){best=c;bc=cnt;} return best;
}
std::vector<int> predict_class_batch(const KNNModel& m, const std::vector<Tensor1D>& X) { std::vector<int> o; for (const auto& x:X) o.push_back(predict_class(m,x)); return o; }
KNNModel fit_regressor(const std::vector<Tensor1D>& X, const Tensor1D& y, int k) { return {X,{},y,k,false}; }
float predict_reg(const KNNModel& m, const Tensor1D& x) {
    std::vector<std::pair<float,float>> d; for (size_t i=0;i<m.X_train.size();++i) { float dd=0; for (size_t j=0;j<x.size();++j){float df=x[j]-m.X_train[i][j];dd+=df*df;} d.push_back({dd,m.y_reg[i]}); }
    std::partial_sort(d.begin(),d.begin()+m.k,d.end()); float s=0; for (int i=0;i<m.k;++i) s+=d[i].second; return s/m.k;
}
} // namespace knn

namespace naive_bayes {
GaussianNBModel fit(const std::vector<Tensor1D>& X, const std::vector<int>& y) {
    GaussianNBModel m; std::map<int,std::vector<size_t>> ci; for (size_t i=0;i<y.size();++i) ci[y[i]].push_back(i);
    m.n_classes=ci.size(); int p=X[0].size(); m.class_priors.resize(m.n_classes); m.class_means.resize(m.n_classes,Tensor1D(p)); m.class_vars.resize(m.n_classes,Tensor1D(p));
    int idx=0; for (const auto& [cls,ids]:ci) {
        m.class_priors[idx]=std::log((float)ids.size()/y.size());
        for (int j=0;j<p;++j) { float s=0; for (size_t i:ids) s+=X[i][j]; m.class_means[idx][j]=s/ids.size(); }
        for (int j=0;j<p;++j) { float s=0,mm=m.class_means[idx][j]; for (size_t i:ids) s+=(X[i][j]-mm)*(X[i][j]-mm); m.class_vars[idx][j]=s/ids.size()+1e-9f; }
        idx++;
    }
    return m;
}
Tensor1D predict_proba(const GaussianNBModel& m, const Tensor1D& x) {
    Tensor1D lp(m.n_classes);
    for (int c=0;c<m.n_classes;++c) { lp[c]=m.class_priors[c]; for (size_t j=0;j<x.size();++j) { float d=x[j]-m.class_means[c][j],v=m.class_vars[c][j]; lp[c]-=0.5f*std::log(6.2832f*v)+0.5f*d*d/v; } }
    float mx=*std::max_element(lp.begin(),lp.end()),sum=0; for (auto& l:lp){l=std::exp(l-mx);sum+=l;} for (auto& l:lp) l/=sum; return lp;
}
int predict(const GaussianNBModel& m, const Tensor1D& x) { auto p=predict_proba(m,x); return std::max_element(p.begin(),p.end())-p.begin(); }
std::vector<int> predict_batch(const GaussianNBModel& m, const std::vector<Tensor1D>& X) { std::vector<int> o; for (const auto& x:X) o.push_back(predict(m,x)); return o; }
} // namespace naive_bayes

Pipeline::Pipeline() {}
Pipeline& Pipeline::add_standard_scaler(const std::string& n) { steps.push_back({StepType::STANDARD_SCALE,n,{},{},0,nullptr,false}); return *this; }
Pipeline& Pipeline::add_minmax_scaler(const std::string& n) { steps.push_back({StepType::MINMAX_SCALE,n,{},{},0,nullptr,false}); return *this; }
Pipeline& Pipeline::add_pca(int nc, const std::string& n) { steps.push_back({StepType::PCA,n,{},{},nc,nullptr,false}); return *this; }
Pipeline& Pipeline::add_poly_features(int d, const std::string& n) { steps.push_back({StepType::POLY_FEATURES,n,{},{},d,nullptr,false}); return *this; }
Pipeline& Pipeline::add_custom(const std::string& n, std::function<std::vector<Tensor1D>(const std::vector<Tensor1D>&)> fn) { steps.push_back({StepType::CUSTOM,n,{},{},0,fn,true}); return *this; }

void Pipeline::fit(const std::vector<Tensor1D>& X) {
    auto data=X; for (auto& s:steps) {
        switch (s.type) {
            case StepType::STANDARD_SCALE: s.scaler_params=preprocess::fit_standard_scaler(data); data=preprocess::transform_standard(data,s.scaler_params); break;
            case StepType::MINMAX_SCALE: s.scaler_params=preprocess::fit_minmax_scaler(data); data=preprocess::transform_minmax(data,s.scaler_params); break;
            case StepType::PCA: s.pca_params=unsupervised::fit_pca(data,s.poly_degree); data=unsupervised::transform_pca(data,s.pca_params); break;
            case StepType::POLY_FEATURES: { Tensor2D m(data.size(),data[0]); m=preprocess::polynomial_features(m,s.poly_degree); data.resize(m.size()); for (size_t i=0;i<m.size();++i) data[i]=m[i]; break; }
            case StepType::CUSTOM: if (s.custom_fn) data=s.custom_fn(data); break;
            default: break;
        }
        s.is_fitted=true;
    }
}
std::vector<Tensor1D> Pipeline::transform(const std::vector<Tensor1D>& X) const {
    auto data=X; for (const auto& s:steps) {
        switch (s.type) {
            case StepType::STANDARD_SCALE: data=preprocess::transform_standard(data,s.scaler_params); break;
            case StepType::MINMAX_SCALE: data=preprocess::transform_minmax(data,s.scaler_params); break;
            case StepType::PCA: data=unsupervised::transform_pca(data,s.pca_params); break;
            case StepType::POLY_FEATURES: { Tensor2D m(data.size(),data[0]); m=preprocess::polynomial_features(m,s.poly_degree); data.resize(m.size()); for (size_t i=0;i<m.size();++i) data[i]=m[i]; break; }
            case StepType::CUSTOM: if (s.custom_fn) data=s.custom_fn(data); break;
            default: break;
        }
    }
    return data;
}
std::vector<Tensor1D> Pipeline::fit_transform(const std::vector<Tensor1D>& X) { fit(X); return transform(X); }

namespace hyperopt {
SearchResult random_search(const std::vector<ParamRange>& pr, std::function<float(const std::map<std::string,float>&)> obj, int nt, bool maximize, int seed) {
    if (seed>=0) std::srand(seed); SearchResult r; float bs=maximize?-1e30f:1e30f;
    for (int t=0;t<nt;++t) {
        std::map<std::string,float> params;
        for (const auto& p:pr) { float v; if (!p.choices.empty()) v=p.choices[std::rand()%p.choices.size()]; else { v=p.min_val+(float)std::rand()/RAND_MAX*(p.max_val-p.min_val); if (p.is_int) v=std::round(v); } params[p.name]=v; }
        auto start=std::chrono::steady_clock::now(); float score=obj(params); float dur=std::chrono::duration<float>(std::chrono::steady_clock::now()-start).count();
        TrialResult tr{params,score,0,dur}; r.trials.push_back(tr);
        if (maximize?score>bs:score<bs) { bs=score; r.best=tr; }
    }
    return r;
}
std::string SearchResult::summary() const {
    std::ostringstream o; o << "HyperOpt: " << trials.size() << " trials, best=" << best.score << "\n";
    for (const auto& [k,v]:best.params) o << "  " << k << "=" << v << "\n"; return o.str();
}
} // namespace hyperopt

namespace reporting {
void TrainingLog::add_epoch(float tl, float vl, float tm, float vm, float lr) {
    train_losses.push_back(tl); val_losses.push_back(vl); train_metrics.push_back(tm); val_metrics.push_back(vm); learning_rates.push_back(lr);
    if (vl < best_val_loss || train_losses.size()==1) { best_val_loss=vl; best_epoch=train_losses.size(); }
}
std::string TrainingLog::summary() const { std::ostringstream o; o << "Training: " << train_losses.size() << " epochs, best_val=" << best_val_loss << " @" << best_epoch; return o.str(); }
std::string TrainingLog::to_csv() const {
    std::ostringstream o; o << "epoch,train_loss,val_loss\n";
    for (size_t i=0;i<train_losses.size();++i) o << i+1 << "," << train_losses[i] << "," << val_losses[i] << "\n"; return o.str();
}
void TrainingLog::save(const std::string& f) const { std::ofstream o(f); o << to_csv(); }
std::string ModelReport::summary() const {
    std::ostringstream o; o << "=== " << model_name << " ===\nType: " << model_type << " Params: " << n_parameters << " Time: " << training_time_seconds << "s\n";
    for (const auto& [k,v]:test_metrics) o << "  " << k << ": " << v << "\n"; return o.str();
}
void ComparisonReport::add_model(const ModelReport& r) { models.push_back(r); }
std::string ComparisonReport::summary() const {
    std::ostringstream o; o << "=== Comparison (" << comparison_metric << ") ===\n";
    for (const auto& m:models) { o << std::left << std::setw(20) << m.model_name; auto it=m.test_metrics.find(comparison_metric); o << (it!=m.test_metrics.end()?it->second:0) << "\n"; }
    return o.str();
}
DataQualityReport generate_data_quality_report(const DataFrame& df) {
    DataQualityReport r; r.n_rows=df.nrows(); r.n_cols=df.ncols(); r.column_names=df.column_names(); r.column_types=df.column_types(); r.duplicate_rows=0;
    for (const auto& c:df.columns) {
        r.null_counts.push_back(c.count_nulls()); r.null_percentages.push_back(c.size()>0?100.0f*c.count_nulls()/c.size():0);
        if (c.dtype!=DType::STRING) { r.means.push_back(c.mean()); r.stds.push_back(c.stddev()); r.mins.push_back(c.min_val()); r.maxs.push_back(c.max_val()); r.skewnesses.push_back(c.skewness()); r.kurtoses.push_back(c.kurtosis()); }
        else { r.means.push_back(0); r.stds.push_back(0); r.mins.push_back(0); r.maxs.push_back(0); r.skewnesses.push_back(0); r.kurtoses.push_back(0); }
        std::set<std::string> u; for (size_t i=0;i<c.size();++i) u.insert(c.get_string(i)); r.unique_counts.push_back(u.size());
    }
    r.memory_usage_bytes=r.n_rows*r.n_cols*4.0f; return r;
}
std::string DataQualityReport::summary() const {
    std::ostringstream o; o << "=== Data Quality ===\n" << n_rows << " rows x " << n_cols << " cols\n";
    for (int i=0;i<n_cols;++i) o << column_names[i] << ": nulls=" << null_counts[i] << "(" << null_percentages[i] << "%) unique=" << unique_counts[i] << "\n"; return o.str();
}
FeatureImportance permutation_importance(const std::vector<Tensor1D>& X, const Tensor1D& y, std::function<Tensor1D(const Tensor1D&)> pred_fn, std::function<float(const Tensor1D&,const Tensor1D&)> metric, const std::vector<std::string>& fn, int nr) {
    int n=X.size(),p=X[0].size(); Tensor1D preds(n); for (int i=0;i<n;++i) preds[i]=pred_fn(X[i])[0]; float baseline=metric(y,preds);
    FeatureImportance r; r.feature_names=fn; r.importances.resize(p);
    for (int j=0;j<p;++j) { float td=0; for (int rr=0;rr<nr;++rr) { auto Xp=X; for (int i=n-1;i>0;--i){int k=std::rand()%(i+1);std::swap(Xp[i][j],Xp[k][j]);} Tensor1D pp(n); for (int i=0;i<n;++i) pp[i]=pred_fn(Xp[i])[0]; td+=baseline-metric(y,pp); } r.importances[j]=td/nr; }
    return r;
}
std::string FeatureImportance::to_string() const {
    std::vector<size_t> idx(feature_names.size()); std::iota(idx.begin(),idx.end(),0);
    std::sort(idx.begin(),idx.end(),[this](size_t a,size_t b){return importances[a]>importances[b];});
    std::ostringstream o; for (size_t i:idx) o << std::left << std::setw(20) << feature_names[i] << importances[i] << "\n"; return o.str();
}
ResidualAnalysis analyze_residuals(const Tensor1D& yt, const Tensor1D& yp) {
    int n=yt.size(); ResidualAnalysis r; r.residuals.resize(n); for (int i=0;i<n;++i) r.residuals[i]=yt[i]-yp[i];
    r.mean_residual=stats::mean(r.residuals); r.std_residual=stats::stddev(r.residuals);
    float dn=0,dd=0; for (int i=1;i<n;++i) dn+=(r.residuals[i]-r.residuals[i-1])*(r.residuals[i]-r.residuals[i-1]); for (int i=0;i<n;++i) dd+=r.residuals[i]*r.residuals[i]; r.durbin_watson=dd>0?dn/dd:2;
    r.normality_test=stats::t_test_one_sample(r.residuals,0); r.normality_test.test_name="Residual normality";
    Tensor1D h1(r.residuals.begin(),r.residuals.begin()+n/2),h2(r.residuals.begin()+n/2,r.residuals.end());
    float ratio=stats::variance(h2)/(stats::variance(h1)+1e-10f); r.heteroscedastic=ratio>2||ratio<0.5f;
    return r;
}
std::string ResidualAnalysis::summary() const {
    std::ostringstream o; o << "Residuals: mean=" << mean_residual << " std=" << std_residual << " DW=" << durbin_watson << " hetero=" << (heteroscedastic?"YES":"no"); return o.str();
}
} // namespace reporting

namespace distance {
float euclidean(const Tensor1D& a, const Tensor1D& b) { float s=0; for (size_t i=0;i<a.size();++i){float d=a[i]-b[i];s+=d*d;} return std::sqrt(s); }
float manhattan(const Tensor1D& a, const Tensor1D& b) { float s=0; for (size_t i=0;i<a.size();++i) s+=std::abs(a[i]-b[i]); return s; }
float cosine_similarity(const Tensor1D& a, const Tensor1D& b) { float d=0,na=0,nb=0; for (size_t i=0;i<a.size();++i){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];} return d/(std::sqrt(na)*std::sqrt(nb)+1e-10f); }
float cosine_distance(const Tensor1D& a, const Tensor1D& b) { return 1-cosine_similarity(a,b); }
float minkowski(const Tensor1D& a, const Tensor1D& b, float p) { float s=0; for (size_t i=0;i<a.size();++i) s+=std::pow(std::abs(a[i]-b[i]),p); return std::pow(s,1/p); }
float chebyshev(const Tensor1D& a, const Tensor1D& b) { float mx=0; for (size_t i=0;i<a.size();++i) mx=std::max(mx,std::abs(a[i]-b[i])); return mx; }
Tensor2D pairwise_distances(const std::vector<Tensor1D>& X, const std::string& m) {
    int n=X.size(); Tensor2D D(n,Tensor1D(n,0));
    for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) { float d; if (m=="manhattan") d=manhattan(X[i],X[j]); else if (m=="cosine") d=cosine_distance(X[i],X[j]); else d=euclidean(X[i],X[j]); D[i][j]=D[j][i]=d; }
    return D;
}
} // namespace distance

namespace math_util {
Tensor1D softmax(const Tensor1D& x) { float mx=*std::max_element(x.begin(),x.end()); Tensor1D o(x.size()); float s=0; for (size_t i=0;i<x.size();++i){o[i]=std::exp(x[i]-mx);s+=o[i];} for (auto& v:o) v/=s; return o; }
Tensor1D linspace(float s, float e, int n) { Tensor1D o(n); float st=(n>1)?(e-s)/(n-1):0; for (int i=0;i<n;++i) o[i]=s+i*st; return o; }
Tensor1D arange(float s, float e, float st) { Tensor1D o; for (float v=s;v<e;v+=st) o.push_back(v); return o; }
Tensor1D zeros(int n) { return Tensor1D(n,0); }
Tensor1D ones(int n) { return Tensor1D(n,1); }
Tensor1D random_normal(int n, float mean, float std) {
    Tensor1D o(n); for (int i=0;i<n;++i) { float u1=(float)std::rand()/RAND_MAX,u2=(float)std::rand()/RAND_MAX; o[i]=mean+std*std::sqrt(-2*std::log(u1+1e-10f))*std::cos(6.2832f*u2); } return o;
}
Tensor1D random_uniform(int n, float lo, float hi) { Tensor1D o(n); for (int i=0;i<n;++i) o[i]=lo+(float)std::rand()/RAND_MAX*(hi-lo); return o; }
Tensor2D eye(int n) { Tensor2D I(n,Tensor1D(n,0)); for (int i=0;i<n;++i) I[i][i]=1; return I; }
Tensor2D transpose(const Tensor2D& A) { if (A.empty()) return {}; int m=A.size(),n=A[0].size(); Tensor2D T(n,Tensor1D(m)); for (int i=0;i<m;++i) for (int j=0;j<n;++j) T[j][i]=A[i][j]; return T; }
float dot(const Tensor1D& a, const Tensor1D& b) { float s=0; for (size_t i=0;i<a.size();++i) s+=a[i]*b[i]; return s; }
Tensor1D dot(const Tensor2D& A, const Tensor1D& x) { return matvec(A, x); }
float norm(const Tensor1D& x, float p) { float s=0; for (float v:x) s+=std::pow(std::abs(v),p); return std::pow(s,1/p); }
Tensor1D elementwise_add(const Tensor1D& a, const Tensor1D& b) { Tensor1D o(a.size()); for (size_t i=0;i<a.size();++i) o[i]=a[i]+b[i]; return o; }
Tensor1D elementwise_mul(const Tensor1D& a, const Tensor1D& b) { Tensor1D o(a.size()); for (size_t i=0;i<a.size();++i) o[i]=a[i]*b[i]; return o; }
Tensor1D scalar_mul(const Tensor1D& a, float s) { Tensor1D o(a.size()); for (size_t i=0;i<a.size();++i) o[i]=a[i]*s; return o; }
Tensor2D matrix_add(const Tensor2D& A, const Tensor2D& B) { Tensor2D C(A.size(),Tensor1D(A[0].size())); for (size_t i=0;i<A.size();++i) for (size_t j=0;j<A[0].size();++j) C[i][j]=A[i][j]+B[i][j]; return C; }
Tensor1D solve(const Tensor2D& A, const Tensor1D& b) {
    int n=A.size(); Tensor2D aug(n,Tensor1D(n+1)); for (int i=0;i<n;++i){for (int j=0;j<n;++j) aug[i][j]=A[i][j]; aug[i][n]=b[i];}
    for (int i=0;i<n;++i) { int mr=i; for (int k=i+1;k<n;++k) if (std::abs(aug[k][i])>std::abs(aug[mr][i])) mr=k; std::swap(aug[i],aug[mr]); float piv=aug[i][i]; if (std::abs(piv)<1e-12f) continue; for (int j=0;j<=n;++j) aug[i][j]/=piv; for (int k=0;k<n;++k){if (k==i) continue; float f=aug[k][i]; for (int j=0;j<=n;++j) aug[k][j]-=f*aug[i][j];} }
    Tensor1D x(n); for (int i=0;i<n;++i) x[i]=aug[i][n]; return x;
}
Tensor2D inverse(const Tensor2D& A) {
    int n=A.size(); Tensor2D aug(n,Tensor1D(2*n,0)); for (int i=0;i<n;++i){for (int j=0;j<n;++j) aug[i][j]=A[i][j]; aug[i][n+i]=1;}
    for (int i=0;i<n;++i) { int mr=i; for (int k=i+1;k<n;++k) if (std::abs(aug[k][i])>std::abs(aug[mr][i])) mr=k; std::swap(aug[i],aug[mr]); float piv=aug[i][i]; if (std::abs(piv)<1e-12f) continue; for (int j=0;j<2*n;++j) aug[i][j]/=piv; for (int k=0;k<n;++k){if (k==i) continue; float f=aug[k][i]; for (int j=0;j<2*n;++j) aug[k][j]-=f*aug[i][j];} }
    Tensor2D inv(n,Tensor1D(n)); for (int i=0;i<n;++i) for (int j=0;j<n;++j) inv[i][j]=aug[i][n+j]; return inv;
}
float determinant(const Tensor2D& A) {
    int n=A.size(); if (n==1) return A[0][0]; if (n==2) return A[0][0]*A[1][1]-A[0][1]*A[1][0];
    Tensor2D U=A; float det=1;
    for (int i=0;i<n;++i) { int mr=i; for (int k=i+1;k<n;++k) if (std::abs(U[k][i])>std::abs(U[mr][i])) mr=k; if (mr!=i){std::swap(U[i],U[mr]);det*=-1;} if (std::abs(U[i][i])<1e-12f) return 0; det*=U[i][i]; for (int k=i+1;k<n;++k){float f=U[k][i]/U[i][i]; for (int j=i;j<n;++j) U[k][j]-=f*U[i][j];} }
    return det;
}
EigenResult eigen(const Tensor2D& A, int max_iter) {
    int n=A.size(); EigenResult r; r.eigenvalues.resize(n); r.eigenvectors.resize(n,Tensor1D(n));
    Tensor2D M=A,V=eye(n);
    for (int iter=0;iter<max_iter;++iter) for (int i=0;i<n;++i) {
        Tensor1D v(n); for (int j=0;j<n;++j) v[j]=V[j][i];
        Tensor1D Mv(n,0); for (int j=0;j<n;++j) for (int k=0;k<n;++k) Mv[j]+=M[j][k]*v[k];
        float nm=0; for (float val:Mv) nm+=val*val; nm=std::sqrt(nm); if (nm>0) for (auto& val:Mv) val/=nm;
        for (int j=0;j<n;++j) V[j][i]=Mv[j];
    }
    for (int i=0;i<n;++i) { Tensor1D v(n); for (int j=0;j<n;++j) v[j]=V[j][i]; Tensor1D Av(n,0); for (int j=0;j<n;++j) for (int k=0;k<n;++k) Av[j]+=A[j][k]*v[k]; r.eigenvalues[i]=0; for (int j=0;j<n;++j) r.eigenvalues[i]+=v[j]*Av[j]; r.eigenvectors[i]=v; }
    return r;
}
} // namespace math_util

} // namespace nn
