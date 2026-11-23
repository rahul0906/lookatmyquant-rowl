// Run the exported target layer and verify every cache output.
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include "mlx/mlx.h"
namespace mx = mlx::core;
int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: bench_native_layer EXPORT_DIR\n"; return 2; }
    std::string dir(argv[1]);
    auto data = mx::load_safetensors(dir + "/stateful_inputs.safetensors").first;
    auto fn = mx::import_function(dir + "/stateful.mlxfn");
    std::vector<mx::array> inputs;
    for (int i=0; i<6; ++i) inputs.push_back(data.at("input_" + std::to_string(i)));
    auto output = fn(inputs);
    for (size_t i=0; i<output.size(); ++i) {
        auto key = "expected_" + std::to_string(i);
        if (!data.count(key) || !mx::array_equal(output[i], data.at(key)).item<bool>()) {
            std::cerr << "output mismatch at " << i << "\n"; return 1;
        }
    }
    if (output.size() != 6) { std::cerr << "unexpected output count\n"; return 1; }
    for (int i=0; i<4; ++i) mx::eval(fn(inputs));
    std::vector<double> times;
    for (int i=0; i<20; ++i) {
        auto start=std::chrono::steady_clock::now();
        mx::eval(fn(inputs));
        times.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
    }
    std::sort(times.begin(),times.end());
    std::cout << "{\"matches\":true,\"outputs\":" << output.size() << ",\"median_ms\":" << (times[9]+times[10])/2 << ",\"samples_ms\":[";
    for (size_t i=0; i<times.size(); ++i) std::cout << (i ? "," : "") << times[i];
    std::cout << "]}\n";
}
