#pragma once

#include <map>
#include <string>
#include <vector>

namespace phantom {

/// Minimal INI reader: `[section]` headers, `key = value` pairs, `#`/`;` comments.
/// Keys are addressed as "section.key". Missing keys fall back to caller defaults.
class Config {
public:
    Config() = default;

    static Config load(const std::string& path);
    static Config parse(const std::string& text);

    bool has(const std::string& key) const { return values_.count(key) > 0; }
    std::string get_string(const std::string& key, const std::string& def = "") const;
    float get_float(const std::string& key, float def) const;
    int get_int(const std::string& key, int def) const;
    bool get_bool(const std::string& key, bool def) const;
    std::vector<float> get_floats(const std::string& key) const;  // comma separated

    void set(const std::string& key, const std::string& value) { values_[key] = value; }
    const std::map<std::string, std::string>& values() const { return values_; }

private:
    std::map<std::string, std::string> values_;
};

}  // namespace phantom
