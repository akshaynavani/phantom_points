#pragma once

#include <cstdint>
#include <random>

namespace phantom {

/// Thin wrapper over mt19937_64 so every scene / scan is reproducible from a seed.
class Rng {
public:
    explicit Rng(uint64_t seed = 0) : eng_(seed) {}

    /// Combine a global seed and an index into an independent stream seed (splitmix64).
    static uint64_t derive(uint64_t seed, uint64_t index) {
        uint64_t z = seed + 0x9E3779B97F4A7C15ULL * (index + 1);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    float uniform(float lo = 0.f, float hi = 1.f) {
        return std::uniform_real_distribution<float>(lo, hi)(eng_);
    }
    int uniform_int(int lo, int hi) {  // inclusive
        return std::uniform_int_distribution<int>(lo, hi)(eng_);
    }
    float normal(float mean = 0.f, float stddev = 1.f) {
        if (stddev <= 0.f) return mean;
        return std::normal_distribution<float>(mean, stddev)(eng_);
    }
    bool bernoulli(float p) {
        if (p <= 0.f) return false;
        if (p >= 1.f) return true;
        return uniform() < p;
    }
    int poisson(float mean) {
        if (mean <= 0.f) return 0;
        return std::poisson_distribution<int>(mean)(eng_);
    }

    std::mt19937_64& engine() { return eng_; }

private:
    std::mt19937_64 eng_;
};

}  // namespace phantom
