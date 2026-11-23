// Run an exported Inkling draft recurrence with MLX; see bench_native_export.py.
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include "mlx/mlx.h"

namespace mx = mlx::core;
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: bench_native_draft EXPORT_DIR\n";
        return 2;
    }
    std::string dir(argv[1]);
    auto data = mx::load_safetensors(dir + "/inputs.safetensors").first;
    auto fn = mx::import_function(dir + "/draft.mlxfn");
    std::vector<mx::array> inputs{data.at("anchor"), data.at("hidden"), data.at("context")};
    auto output = fn(inputs);
    if (!mx::array_equal(output[0], data.at("expected")).item<bool>()) {
        std::cerr << "output differs from Python\n";
        return 1;
    }
    for (int i = 0; i < 4; ++i) mx::eval(fn(inputs));
    std::vector<double> times;
    for (int i = 0; i < 20; ++i) {
        const auto start = std::chrono::steady_clock::now();
        mx::eval(fn(inputs));
        times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    std::sort(times.begin(), times.end());
    std::cout << "{\"matches\":true,\"median_ms\":" << (times[9]+times[10])/2 << ",\"samples_ms\":[";
    for (size_t i = 0; i < times.size(); ++i) std::cout << (i ? "," : "") << times[i];
    std::cout << "]}\n";
}
