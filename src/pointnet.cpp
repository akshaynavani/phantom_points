#include "phantom/pointnet.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace phantom {

namespace {

constexpr int kGeometricFeatures = 6;
constexpr uint32_t kVersion = 1;

template <typename T>
T read_pod(std::istream& in) {
    T v{};
    in.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!in) throw std::runtime_error("truncated .ppnt file");
    return v;
}

DenseLayer read_layer(std::istream& in) {
    DenseLayer L;
    L.in_dim = static_cast<int>(read_pod<uint32_t>(in));
    L.out_dim = static_cast<int>(read_pod<uint32_t>(in));
    L.relu = read_pod<uint32_t>(in) != 0;
    if (L.in_dim <= 0 || L.out_dim <= 0 || L.in_dim > 65536 || L.out_dim > 65536)
        throw std::runtime_error("corrupt layer dimensions in .ppnt file");
    L.weight.resize(static_cast<size_t>(L.in_dim) * L.out_dim);
    L.bias.resize(L.out_dim);
    in.read(reinterpret_cast<char*>(L.weight.data()), L.weight.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(L.bias.data()), L.bias.size() * sizeof(float));
    if (!in) throw std::runtime_error("truncated .ppnt layer");
    return L;
}

std::vector<float> run_block(const std::vector<DenseLayer>& block, std::vector<float> x, int n) {
    std::vector<float> y;
    for (const DenseLayer& L : block) {
        y.assign(static_cast<size_t>(n) * L.out_dim, 0.f);
        L.forward(x.data(), n, y.data());
        x.swap(y);
    }
    return x;
}

inline float sigmoid(float z) { return 1.f / (1.f + std::exp(-z)); }

}  // namespace

void DenseLayer::forward(const float* x, int n, float* y) const {
    for (int i = 0; i < n; ++i) {
        const float* xi = x + static_cast<size_t>(i) * in_dim;
        float* yi = y + static_cast<size_t>(i) * out_dim;
        for (int o = 0; o < out_dim; ++o) {
            const float* w = weight.data() + static_cast<size_t>(o) * in_dim;
            float acc = bias[o];
            for (int k = 0; k < in_dim; ++k) acc += w[k] * xi[k];
            yi[o] = relu ? std::max(acc, 0.f) : acc;
        }
    }
}

PointNetRefiner PointNetRefiner::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open refiner weights: " + path);
    char magic[4];
    in.read(magic, 4);
    if (!in || std::memcmp(magic, "PPNT", 4) != 0) throw std::runtime_error("not a .ppnt file: " + path);

    PointNetRefiner r;
    uint32_t version = read_pod<uint32_t>(in);
    if (version != kVersion) throw std::runtime_error("unsupported .ppnt version");
    r.in_features_ = static_cast<int>(read_pod<uint32_t>(in));
    r.num_classes_ = static_cast<int>(read_pod<uint32_t>(in));
    r.sectors_ = static_cast<int>(read_pod<uint32_t>(in));
    r.feature_scale_ = read_pod<float>(in);
    r.max_delta_range_ = read_pod<float>(in);
    r.max_delta_intensity_ = read_pod<float>(in);
    uint32_t n_local = read_pod<uint32_t>(in);
    uint32_t n_global = read_pod<uint32_t>(in);
    uint32_t n_head = read_pod<uint32_t>(in);
    for (uint32_t i = 0; i < n_local; ++i) r.local_.push_back(read_layer(in));
    for (uint32_t i = 0; i < n_global; ++i) r.global_.push_back(read_layer(in));
    for (uint32_t i = 0; i < n_head; ++i) r.head_.push_back(read_layer(in));
    r.validate();
    return r;
}

PointNetRefiner PointNetRefiner::from_layers(std::vector<DenseLayer> local,
                                             std::vector<DenseLayer> global,
                                             std::vector<DenseLayer> head, int num_classes,
                                             int sectors, float feature_scale, float max_dr,
                                             float max_di) {
    PointNetRefiner r;
    r.local_ = std::move(local);
    r.global_ = std::move(global);
    r.head_ = std::move(head);
    r.num_classes_ = num_classes;
    r.in_features_ = kGeometricFeatures + num_classes;
    r.sectors_ = sectors;
    r.feature_scale_ = feature_scale;
    r.max_delta_range_ = max_dr;
    r.max_delta_intensity_ = max_di;
    r.validate();
    return r;
}

void PointNetRefiner::validate() const {
    if (local_.empty() || global_.empty() || head_.empty())
        throw std::runtime_error("refiner needs local, global and head layers");
    if (in_features_ != kGeometricFeatures + num_classes_)
        throw std::runtime_error("refiner feature count does not match class count");
    auto chain = [](const std::vector<DenseLayer>& b, int in) {
        for (const auto& L : b) {
            if (L.in_dim != in) throw std::runtime_error("refiner layer dimensions do not chain");
            in = L.out_dim;
        }
        return in;
    };
    int local_out = chain(local_, in_features_);
    int global_out = chain(global_, local_out);
    int head_out = chain(head_, local_out + global_out);
    if (head_out != 3) throw std::runtime_error("refiner head must output 3 channels");
    if (sectors_ <= 0) throw std::runtime_error("refiner sector count must be positive");
}

void PointNetRefiner::build_features(const LidarPoint& p, int beams, float* out) const {
    const float s = feature_scale_;
    out[0] = p.x / s;
    out[1] = p.y / s;
    out[2] = p.z / s;
    out[3] = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z) / s;
    out[4] = p.intensity;
    out[5] = static_cast<float>(p.ring) / static_cast<float>(std::max(beams - 1, 1));
    for (int c = 0; c < num_classes_; ++c) out[kGeometricFeatures + c] = 0.f;
    int lbl = std::min<int>(p.label, num_classes_ - 1);
    out[kGeometricFeatures + lbl] = 1.f;
}

std::vector<float> PointNetRefiner::forward(const std::vector<float>& feats, int n) const {
    if (n == 0) return {};
    std::vector<float> local = run_block(local_, feats, n);
    const int local_dim = local_.back().out_dim;
    std::vector<float> g = run_block(global_, local, n);
    const int global_dim = global_.back().out_dim;

    // Symmetric max-pool over the sector.
    std::vector<float> pooled(global_dim, -std::numeric_limits<float>::infinity());
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < global_dim; ++d)
            pooled[d] = std::max(pooled[d], g[static_cast<size_t>(i) * global_dim + d]);

    const int cat_dim = local_dim + global_dim;
    std::vector<float> cat(static_cast<size_t>(n) * cat_dim);
    for (int i = 0; i < n; ++i) {
        float* row = cat.data() + static_cast<size_t>(i) * cat_dim;
        std::copy_n(local.data() + static_cast<size_t>(i) * local_dim, local_dim, row);
        std::copy_n(pooled.data(), global_dim, row + local_dim);
    }
    return run_block(head_, std::move(cat), n);
}

RefineStats PointNetRefiner::refine(Scan& scan, Rng& rng, const RefineOptions& opts) const {
    RefineStats stats;
    stats.input_points = scan.points.size();
    if (scan.points.empty()) return stats;

    // Bucket points by azimuth sector (same integer arithmetic as features.sector_ids).
    std::vector<std::vector<size_t>> buckets(sectors_);
    for (size_t i = 0; i < scan.points.size(); ++i) {
        int s = static_cast<int>(static_cast<int64_t>(scan.points[i].column) * sectors_ / scan.columns);
        buckets[std::min(std::max(s, 0), sectors_ - 1)].push_back(i);
    }

    std::vector<uint8_t> keep(scan.points.size(), 1);
    std::vector<float> feats;
    double sum_dr = 0.0, sum_di = 0.0;
    for (const auto& idx : buckets) {
        const int n = static_cast<int>(idx.size());
        if (n == 0) continue;
        feats.assign(static_cast<size_t>(n) * in_features_, 0.f);
        for (int i = 0; i < n; ++i)
            build_features(scan.points[idx[i]], scan.beams, feats.data() + static_cast<size_t>(i) * in_features_);
        std::vector<float> out = forward(feats, n);

        for (int i = 0; i < n; ++i) {
            LidarPoint& p = scan.points[idx[i]];
            const float* o = out.data() + static_cast<size_t>(i) * 3;
            float dr = max_delta_range_ * std::tanh(o[0]);
            float di = max_delta_intensity_ * std::tanh(o[1]);
            float p_drop = sigmoid(o[2]);

            if (opts.allow_dropout) {
                bool drop = opts.sample_dropout ? rng.bernoulli(p_drop) : p_drop > 0.5f;
                if (drop) { keep[idx[i]] = 0; continue; }
            }
            if (opts.refine_range && p.range > 0.f) {
                float r_new = std::max(p.range + dr, 0.05f);
                float k = r_new / p.range;
                p.x *= k; p.y *= k; p.z *= k;
                p.range = r_new;
                sum_dr += std::fabs(dr);
            }
            if (opts.refine_intensity) {
                p.intensity = clamp(p.intensity + di, 0.f, opts.intensity_saturation);
                sum_di += std::fabs(di);
            }
            p.flags |= kFlagRefined;
        }
    }

    size_t w = 0;
    for (size_t i = 0; i < scan.points.size(); ++i)
        if (keep[i]) scan.points[w++] = scan.points[i];
    stats.dropped_points = scan.points.size() - w;
    scan.points.resize(w);

    // Keep per-object surviving point counts consistent with the refined scan.
    for (auto& v : scan.visibility) v.final_points = 0;
    for (const auto& p : scan.points) {
        if (p.instance == 0) continue;
        for (auto& v : scan.visibility)
            if (v.instance_id == p.instance) { ++v.final_points; break; }
    }

    if (w > 0) {
        stats.mean_abs_delta_range = sum_dr / static_cast<double>(w);
        stats.mean_abs_delta_intensity = sum_di / static_cast<double>(w);
    }
    return stats;
}

}  // namespace phantom
