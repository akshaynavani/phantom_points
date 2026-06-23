#include "phantom/config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace phantom {

namespace {

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
    return s.substr(b, e - b);
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

}  // namespace

Config Config::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config: " + path);
    std::stringstream ss;
    ss << in.rdbuf();
    return parse(ss.str());
}

Config Config::parse(const std::string& text) {
    Config cfg;
    std::istringstream in(text);
    std::string line, section;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        // Strip inline comments.
        size_t c = line.find_first_of("#;");
        if (c != std::string::npos) line = line.substr(0, c);
        line = trim(line);
        if (line.empty()) continue;

        if (line.front() == '[') {
            if (line.back() != ']')
                throw std::runtime_error("config line " + std::to_string(lineno) + ": bad section");
            section = lower(trim(line.substr(1, line.size() - 2)));
            continue;
        }
        size_t eq = line.find('=');
        if (eq == std::string::npos)
            throw std::runtime_error("config line " + std::to_string(lineno) + ": expected key = value");
        std::string key = lower(trim(line.substr(0, eq)));
        std::string val = trim(line.substr(eq + 1));
        cfg.values_[section.empty() ? key : section + "." + key] = val;
    }
    return cfg;
}

std::string Config::get_string(const std::string& key, const std::string& def) const {
    auto it = values_.find(key);
    return it == values_.end() ? def : it->second;
}

float Config::get_float(const std::string& key, float def) const {
    auto it = values_.find(key);
    if (it == values_.end()) return def;
    try {
        return std::stof(it->second);
    } catch (const std::exception&) {
        throw std::runtime_error("config key " + key + " is not a number: " + it->second);
    }
}

int Config::get_int(const std::string& key, int def) const {
    auto it = values_.find(key);
    if (it == values_.end()) return def;
    try {
        return std::stoi(it->second);
    } catch (const std::exception&) {
        throw std::runtime_error("config key " + key + " is not an integer: " + it->second);
    }
}

bool Config::get_bool(const std::string& key, bool def) const {
    auto it = values_.find(key);
    if (it == values_.end()) return def;
    std::string v = lower(it->second);
    if (v == "1" || v == "true" || v == "yes" || v == "on") return true;
    if (v == "0" || v == "false" || v == "no" || v == "off") return false;
    throw std::runtime_error("config key " + key + " is not a boolean: " + it->second);
}

std::vector<float> Config::get_floats(const std::string& key) const {
    std::vector<float> out;
    auto it = values_.find(key);
    if (it == values_.end()) return out;
    std::stringstream ss(it->second);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        tok = trim(tok);
        if (!tok.empty()) out.push_back(std::stof(tok));
    }
    return out;
}

}  // namespace phantom
