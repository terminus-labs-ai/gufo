// Routed IQ3_S/IQ4_XS/IQ4_NL projections (decode vector and prefill tiled
// MMQ paths) against an F64 reference over the CPU dequantizers, and the
// Q6_K -> Q8_0 output-head conversion against DequantizeQ6_K.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/mmq/qfn_mmq.h"

namespace fn = gufo::models::qwen38_flash_next;
using gufo::core::GgmlType;
namespace {

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
  }
}

std::uint32_t NextRandom(std::uint32_t* state) noexcept {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

float Uniform(std::uint32_t* state, float scale) {
  return scale *
         static_cast<float>(static_cast<int>(NextRandom(state) & 0xFFFFU) -
                            32768) /
         32768.0F;
}

struct Format {
  const char* name;
  std::size_t block;
  std::size_t bytes;
  std::size_t scale_offset;  ///< Byte offset of the F16 super-block scale.
  void (*dequantize)(const void*, float*, std::size_t);
};

Format FormatOf(GgmlType type) {
  switch (type) {
    case GgmlType::kIQ3_S:
      return {"IQ3_S", 256, 110, 0, gufo::quant::DequantizeIQ3_S};
    case GgmlType::kIQ4_XS:
      return {"IQ4_XS", 256, 136, 0, gufo::quant::DequantizeIQ4_XS};
    case GgmlType::kIQ4_NL:
      return {"IQ4_NL", 32, 18, 0, gufo::quant::DequantizeIQ4_NL};
    case GgmlType::kQ6_K:
      return {"Q6_K", 256, 210, 208, gufo::quant::DequantizeQ6_K};
    default:
      throw std::logic_error("unexpected test format");
  }
}

/// Random blocks with finite F16 scales, and their CPU-dequantized rows.
struct Weights {
  std::vector<std::uint8_t> blocks;
  std::vector<float> values;
};

Weights MakeWeights(GgmlType type, std::size_t rows, std::size_t k,
                    std::uint32_t seed) {
  const Format f = FormatOf(type);
  const std::size_t row_bytes = k / f.block * f.bytes;
  Weights w;
  w.blocks.resize(rows * row_bytes);
  for (auto& b : w.blocks)
    b = static_cast<std::uint8_t>(NextRandom(&seed) >> 7);
  for (std::size_t i = 0; i < w.blocks.size() / f.bytes; ++i) {
    const __half d =
        __float2half(0.002F + 0.008F * (NextRandom(&seed) % 1000) / 1000.0F);
    std::memcpy(w.blocks.data() + i * f.bytes + f.scale_offset, &d, 2);
  }
  w.values.resize(rows * k);
  for (std::size_t r = 0; r < rows; ++r)
    f.dequantize(w.blocks.data() + r * row_bytes, w.values.data() + r * k, k);
  return w;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  T* device = nullptr;
  CheckHip(hipMalloc(&device, host.size() * sizeof(T) + 4096), "hipMalloc");
  CheckHip(hipMemset(device, 0, host.size() * sizeof(T) + 4096), "hipMemset");
  CheckHip(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                     hipMemcpyHostToDevice),
           "upload");
  return device;
}

template<typename T>
std::vector<T> Download(const T* device, std::size_t count) {
  std::vector<T> host(count);
  CheckHip(
      hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost),
      "download");
  return host;
}

enum class Path { kVector, kVectorPair, kTiled };

/// Runs one routed projection of `rows` input rows, each with `used` expert
/// slots, and checks every output against the F64 reference. The tolerance
/// is the one the routed Q4_K tests apply to the tiled tier: 1% of the
/// largest reference magnitude (the 8-bit activation quantization).
bool Check(GgmlType type, Path path, int rows, int used, int experts, int m,
           int k, std::uint32_t seed) {
  const Format f = FormatOf(type);
  const int projections = path == Path::kVectorPair ? 2 : 1;
  std::vector<Weights> w;
  for (int p = 0; p < projections; ++p)
    w.push_back(MakeWeights(type, std::size_t(experts) * m, k, seed + p));
  std::uint32_t state = seed ^ 0x5A5A5A5AU;
  std::vector<std::int32_t> ids(std::size_t(rows) * used);
  for (int t = 0; t < rows; ++t) {
    for (int j = 0; j < used; ++j) {
      std::int32_t e;
      do {
        e = static_cast<std::int32_t>(NextRandom(&state) % experts);
      } while (std::find(ids.begin() + t * used, ids.begin() + t * used + j,
                         e) != ids.begin() + t * used + j);
      ids[t * used + j] = e;
    }
  }
  std::vector<float> x(std::size_t(rows) * k);
  for (float& v : x)
    v = Uniform(&state, 2.0F);

  const std::size_t outputs = std::size_t(rows) * used * m;
  std::vector<void*> dw;
  std::vector<float*> dy;
  for (int p = 0; p < projections; ++p) {
    dw.push_back(Upload(w[p].blocks));
    dy.push_back(Upload(std::vector<float>(outputs, NAN)));
  }
  float* dx = Upload(x);
  std::int32_t* di = Upload(ids);
  int rc = -1;
  if (path == Path::kTiled) {
    const auto raw = type == GgmlType::kIQ3_S    ? qfn_mmq_iq3_s_moe_raw
                     : type == GgmlType::kIQ4_XS ? qfn_mmq_iq4_xs_moe_raw
                                                 : qfn_mmq_iq4_nl_moe_raw;
    rc = raw(dw[0], dx, di, dy[0], m, k, rows, experts, used, nullptr);
  } else {
    rc = qfn_mmq_moe_vec(static_cast<int>(type), dw[0], dx, di, dy[0], m, k,
                         rows, experts, used, nullptr,
                         projections == 2 ? dw[1] : nullptr,
                         projections == 2 ? dy[1] : nullptr);
  }
  if (rc != 0)
    throw std::runtime_error(std::string(f.name) + " routed projection failed");
  CheckHip(hipDeviceSynchronize(), "routed projection");

  double worst = 0.0, scale = 0.0;
  for (int p = 0; p < projections; ++p) {
    const auto out = Download(dy[p], outputs);
    for (std::size_t s = 0; s < ids.size(); ++s) {
      const std::size_t t = s / used;
      for (int row = 0; row < m; ++row) {
        const float* wr =
            w[p].values.data() + (std::size_t(ids[s]) * m + row) * k;
        double ref = 0.0;
        for (int i = 0; i < k; ++i)
          ref += double(wr[i]) * x[t * k + i];
        const float v = out[s * m + row];
        if (!std::isfinite(v))
          throw std::runtime_error(std::string(f.name) +
                                   " routed output is not finite");
        worst = std::max(worst, std::abs(ref - v));
        scale = std::max(scale, std::abs(ref));
      }
    }
  }
  for (void* p : dw)
    (void)hipFree(p);
  for (float* p : dy)
    (void)hipFree(p);
  (void)hipFree(dx);
  (void)hipFree(di);
  const bool ok = worst < 1e-2 * scale;
  std::cout << f.name << ' '
            << (path == Path::kTiled        ? "tiled"
                : path == Path::kVectorPair ? "vector pair"
                                            : "vector")
            << " rows=" << rows << " used=" << used << " m=" << m << " k=" << k
            << ": worst |GPU - F64| " << worst << " (scale " << scale << ")"
            << (ok ? "" : " FAILED") << '\n';
  return ok;
}

/// Decode (1, 4, 8 tokens) and prefill (64, 512 tokens) for one format: the
/// gate/up view routes each token to `used` experts; the down view sees one
/// (token, slot) row per expert.
bool CheckFormat(GgmlType type, std::uint32_t seed) {
  constexpr int experts = 16, used = 4, ff = 128, hidden = 1024;
  // IQ4_NL down projections read the model's 640-wide expert rows.
  const int down_k = type == GgmlType::kIQ4_NL ? 640 : 512;
  bool ok = true;
  for (int tokens : {1, 4, 8}) {
    ok = Check(type, Path::kVectorPair, tokens, used, experts, ff, hidden,
               seed + tokens) &&
         ok;
    ok = Check(type, Path::kVector, tokens * used, 1, experts, 256, down_k,
               seed + 100 + tokens) &&
         ok;
  }
  for (int tokens : {64, 512}) {
    ok = Check(type, Path::kTiled, tokens, used, experts, ff, hidden,
               seed + tokens) &&
         ok;
    ok = Check(type, Path::kTiled, tokens * used, 1, experts, 256, down_k,
               seed + 100 + tokens) &&
         ok;
  }
  return ok;
}

/// Every converted element lies within half its Q8_0 block's stored step of
/// the Q6_K value.
bool CheckHeadConversion() {
  constexpr std::size_t rows = 37, cols = 2560;
  const Weights q6 = MakeWeights(GgmlType::kQ6_K, rows, cols, 0x6A6B6CU);
  void* src = Upload(q6.blocks);
  std::vector<std::uint8_t> q8(rows * cols / 32 * 34);
  void* dst = Upload(q8);
  if (qfn_mmq_q6_K_to_q8_0(src, dst, std::int64_t(rows * cols), nullptr) != 0)
    throw std::runtime_error("Q6_K -> Q8_0 conversion failed");
  CheckHip(hipDeviceSynchronize(), "Q6_K -> Q8_0 conversion");
  q8 = Download(static_cast<const std::uint8_t*>(dst), q8.size());
  std::vector<float> converted(rows * cols);
  gufo::quant::DequantizeQ8_0(q8.data(), converted.data(), converted.size());
  double worst_ratio = 0.0;
  for (std::size_t i = 0; i < converted.size(); ++i) {
    __half d;
    std::memcpy(&d, q8.data() + i / 32 * 34, 2);
    const double step = __half2float(d);
    const double error = std::abs(double(converted[i]) - q6.values[i]);
    if (!std::isfinite(converted[i]) || error > 0.5 * step * (1 + 1e-5)) {
      std::cout << "Q6_K -> Q8_0 element " << i << " differs by " << error
                << " (step " << step << ") FAILED\n";
      return false;
    }
    if (step > 0)
      worst_ratio = std::max(worst_ratio, error / step);
  }
  (void)hipFree(src);
  (void)hipFree(dst);
  std::cout << "Q6_K -> Q8_0 head: worst error " << worst_ratio
            << " of a Q8_0 step\n";
  return true;
}

/// Q6_K stays rejected everywhere but a separate output head.
bool CheckUploadRejectsQ6K() {
  // An empty GGUF: rejection must happen before any tensor is read.
  std::vector<std::uint8_t> gguf(32, 0);  // Header padded to alignment.
  std::memcpy(gguf.data(), "GGUF", 4);
  gguf[4] = 3;
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(gguf.data(), gguf.size(), &error);
  if (!reader)
    throw std::runtime_error("empty GGUF did not open: " + error);
  static const std::uint8_t storage[1] = {};
  const auto tensor = [](GgmlType type, const char* name) {
    fn::TensorRef t;
    t.data = storage;
    t.type = type;
    t.cols = 256;
    t.name = name;
    return t;
  };
  const auto rejected = [&](const fn::ModelWeights& w,
                            const std::string& name) {
    std::string message;
    const bool ok = fn::rocm::DeviceModel::Upload(w, *reader, nullptr, nullptr,
                                                  &message) == nullptr &&
                    message == "unsupported HIP tensor format Q6_K: " + name;
    std::cout << "Q6_K " << name << ": " << (ok ? "rejected" : "FAILED") << " ("
              << message << ")\n";
    return ok;
  };
  bool ok = true;
  // A Q6_K embedding, tied to the head, is not converted.
  fn::ModelWeights tied;
  tied.token_embd = tensor(GgmlType::kQ6_K, "token_embd.weight");
  tied.output = tied.token_embd;
  ok = rejected(tied, "token_embd.weight") && ok;
  // A Q6_K routed tensor next to a convertible Q6_K head.
  fn::ModelWeights routed;
  routed.token_embd = tensor(GgmlType::kQ8_0, "token_embd.weight");
  static const std::uint8_t head_storage[1] = {};
  routed.output = tensor(GgmlType::kQ6_K, "output.weight");
  routed.output.data = head_storage;
  routed.layers.resize(1);
  routed.layers[0].ffn_gate_exps = tensor(GgmlType::kIQ3_S, "gate");
  routed.layers[0].ffn_up_exps = tensor(GgmlType::kIQ3_S, "up");
  routed.layers[0].ffn_down_exps =
      tensor(GgmlType::kQ6_K, "blk.0.ffn_down_exps.weight");
  ok = rejected(routed, "blk.0.ffn_down_exps.weight") && ok;
  return ok;
}

}  // namespace

int main() {
  try {
    bool ok = true;
    ok = CheckFormat(GgmlType::kIQ3_S, 0x1A3500U) && ok;
    ok = CheckFormat(GgmlType::kIQ4_XS, 0x1A4C00U) && ok;
    ok = CheckFormat(GgmlType::kIQ4_NL, 0x1A4E00U) && ok;
    ok = CheckHeadConversion() && ok;
    ok = CheckUploadRejectsQ6K() && ok;
    return ok ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
