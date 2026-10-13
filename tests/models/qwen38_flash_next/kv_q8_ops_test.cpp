// The q8_0 KV cache against the F16 cache: every writer (StoreKv,
// PrepareAttention with and without query heads) and reader (decode
// Attention with and without split partials, WmmaCausalAttention, and the
// MTP draft's K/V-only preparation plus decode) at a short and a 64K+
// position. F16 itself is covered bit-exactly by attention_ops_test.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace q = gufo::models::qwen38_flash_next::rocm;
namespace {

constexpr std::uint32_t kHeads = 24, kKvHeads = 2, kDim = 256, kRatio = 4;
constexpr std::uint32_t kQWidth = kHeads * kDim, kKvWidth = kKvHeads * kDim;
constexpr std::size_t kQ8Row = q::KvRowHalves(q::KvType::kQ8, kKvWidth);
// Stated tolerances. A stored value is within half its block's step (plus
// the F16 scale's own rounding); attention outputs (values in [-1, 1])
// stay within kOutputLimit of the F16 cache's.
constexpr double kStepFraction = 0.51;
constexpr double kOutputLimit = 1e-2;

void Check(hipError_t e, const char* what) {
  if (e != hipSuccess)
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

template<typename T>
struct Dev {
  explicit Dev(std::size_t n) : n(n) {
    Check(hipMalloc(&p, n * sizeof(T)), "hipMalloc");
    Check(hipMemset(p, 0, n * sizeof(T)), "hipMemset");
  }
  ~Dev() { (void)hipFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  void Up(const std::vector<T>& h) {
    Check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice),
          "upload");
  }
  std::vector<T> Down(std::size_t first, std::size_t count) const {
    std::vector<T> h(count);
    Check(hipDeviceSynchronize(), "sync");
    Check(hipMemcpy(h.data(), p + first, count * sizeof(T),
                    hipMemcpyDeviceToHost),
          "download");
    return h;
  }
  T* p{nullptr};
  std::size_t n;
};

std::vector<float> Values(std::size_t n, std::uint32_t s, float scale) {
  std::vector<float> v(n);
  for (auto& x : v) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    x = scale * static_cast<float>(static_cast<int>(s & 0xFFFFU) - 32768) /
        32768.0F;
  }
  return v;
}

/// Decodes cache rows [first, first + rows) of a q8 cache; `steps` receives
/// each value's block scale.
std::vector<float> DecodeQ8(const Dev<__half>& cache, std::size_t first,
                            std::size_t rows, std::vector<float>* steps) {
  const auto raw = cache.Down(first * kQ8Row, rows * kQ8Row);
  std::vector<float> out(rows * kKvWidth);
  steps->resize(out.size());
  for (std::size_t r = 0; r < rows; ++r) {
    const auto* bytes = reinterpret_cast<const std::int8_t*>(&raw[r * kQ8Row]);
    const auto* scales = &raw[r * kQ8Row + kKvWidth / 2];
    for (std::size_t i = 0; i < kKvWidth; ++i) {
      (*steps)[r * kKvWidth + i] = __half2float(scales[i / 32]);
      out[r * kKvWidth + i] = bytes[i] * (*steps)[r * kKvWidth + i];
    }
  }
  return out;
}

void ExpectStored(const std::vector<float>& want, const Dev<__half>& q8,
                  std::size_t first, const char* what) {
  std::vector<float> steps;
  const auto got = DecodeQ8(q8, first, want.size() / kKvWidth, &steps);
  double worst = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double err = std::fabs(double(got[i]) - want[i]);
    const double limit = kStepFraction * steps[i] + std::fabs(want[i]) / 1024;
    if (err > limit)
      throw std::runtime_error(std::string(what) + " q8 value " +
                               std::to_string(i) + " off by " +
                               std::to_string(err));
    worst = std::max(worst, err);
  }
  std::cout << what << ": max error " << worst << '\n';
}

void ExpectClose(const Dev<float>& f16, const Dev<float>& q8, std::size_t n,
                 const std::string& what) {
  const auto a = f16.Down(0, n), b = q8.Down(0, n);
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(b[i]))
      throw std::runtime_error(what + ": non-finite q8 output");
    worst = std::max(worst, std::fabs(double(a[i]) - b[i]));
  }
  std::cout << what << ": max |q8 - f16| " << worst << '\n';
  if (worst > kOutputLimit)
    throw std::runtime_error(what + " exceeds the q8 tolerance");
}

void RunAt(std::uint32_t start) {
  constexpr std::uint32_t kBatch = 64;
  const std::size_t rows = start + kBatch + 1;
  Dev<__half> k16(rows * kKvWidth), v16(rows * kKvWidth);
  Dev<__half> k8(rows * kQ8Row), v8(rows * kQ8Row);
  Dev<float> src(rows * kKvWidth);
  Dev<std::uint32_t> pos(1);
  const std::string at = " start=" + std::to_string(start);

  // StoreKv round trip over the whole prefix.
  pos.Up({0});
  const auto keys = Values(start * std::size_t{kKvWidth}, 7, 1.0F);
  const auto values = Values(start * std::size_t{kKvWidth}, 9, 1.0F);
  for (auto [host, c16, c8] : {std::tuple{&keys, &k16, &k8},
                               std::tuple{&values, &v16, &v8}}) {
    Check(hipMemcpy(src.p, host->data(), host->size() * sizeof(float),
                    hipMemcpyHostToDevice),
          "upload rows");
    q::StoreKv(src.p, c16->p, start, kKvWidth, pos.p, nullptr);
    q::StoreKv(src.p, c8->p, start, kKvWidth, pos.p, nullptr, q::KvType::kQ8);
  }
  ExpectStored(keys, k8, 0, ("StoreKv round trip" + at).c_str());

  // PrepareAttention writes the batch's rows; queries/gates stay identical.
  const std::uint32_t stride = 2 * (kQWidth + kKvWidth);
  Dev<float> packed(std::size_t{kBatch} * stride), gamma(kDim);
  Dev<float> q16(kBatch * kQWidth), g16(kBatch * kQWidth);
  Dev<float> q8(kBatch * kQWidth), g8(kBatch * kQWidth);
  packed.Up(Values(packed.n, 17, 1.0F));
  gamma.Up(std::vector<float>(kDim, 1.0F));
  pos.Up({start});
  for (const bool prefill : {false, true}) {
    if (!q::PrepareAttention(packed.p, stride, gamma.p, gamma.p, q16.p, g16.p,
                             k16.p, v16.p, kBatch, kHeads, kKvHeads, kDim, 64,
                             pos.p, 1e7F, 1e-6F, nullptr, nullptr, prefill) ||
        !q::PrepareAttention(packed.p, stride, gamma.p, gamma.p, q8.p, g8.p,
                             k8.p, v8.p, kBatch, kHeads, kKvHeads, kDim, 64,
                             pos.p, 1e7F, 1e-6F, nullptr, nullptr, prefill,
                             q::KvType::kQ8))
      throw std::runtime_error("PrepareAttention rejected the geometry");
  }
  if (q16.Down(0, q16.n) != q8.Down(0, q8.n) ||
      g16.Down(0, g16.n) != g8.Down(0, g8.n))
    throw std::runtime_error("q8 preparation changed queries or gates");
  std::vector<float> want;
  for (const auto& h : k16.Down(start * std::size_t{kKvWidth},
                                std::size_t{kBatch} * kKvWidth))
    want.push_back(__half2float(h));
  ExpectStored(want, k8, start, ("PrepareAttention keys" + at).c_str());
  want.clear();
  for (const auto& h : v16.Down(start * std::size_t{kKvWidth},
                                std::size_t{kBatch} * kKvWidth))
    want.push_back(__half2float(h));
  ExpectStored(want, v8, start, ("PrepareAttention values" + at).c_str());

  // Readers over the same history: dense and masked, split and unsplit.
  const std::uint32_t words = (rows / kRatio + 31) / 32;
  Dev<std::uint32_t> mask(std::size_t{kBatch} * words);
  std::vector<std::uint32_t> bits(mask.n);
  std::uint32_t s = 5;
  for (auto& w : bits) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    w = s & 0x11111111U;
  }
  mask.Up(bits);
  Dev<float> o16(kBatch * kQWidth), o8(kBatch * kQWidth);
  Dev<float> parts(std::size_t{kBatch} * kHeads * 8 * (kDim + 2));
  for (const bool masked : {false, true}) {
    const auto* m = masked ? mask.p : nullptr;
    const std::string kind = std::string(masked ? " masked" : " dense") + at;
    for (const std::uint32_t splits : {1U, 8U}) {
      float* p = splits > 1 ? parts.p : nullptr;
      q::Attention(q16.p, k16.p, v16.p, m, words, o16.p, p, splits, 3, pos.p,
                   kHeads, kKvHeads, kDim, kRatio, nullptr);
      q::Attention(q16.p, k8.p, v8.p, m, words, o8.p, p, splits, 3, pos.p,
                   kHeads, kKvHeads, kDim, kRatio, nullptr, q::KvType::kQ8);
      ExpectClose(o16, o8, 3 * kQWidth,
                  "decode Attention splits=" + std::to_string(splits) + kind);
    }
    if (!q::WmmaCausalAttention(q16.p, g16.p, k16.p, v16.p, m, words, o16.p,
                                kBatch, start, kHeads, kKvHeads, kDim, kRatio,
                                nullptr) ||
        !q::WmmaCausalAttention(q16.p, g16.p, k8.p, v8.p, m, words, o8.p,
                                kBatch, start, kHeads, kKvHeads, kDim, kRatio,
                                nullptr, false, q::KvType::kQ8))
      throw std::runtime_error("WmmaCausalAttention rejected the geometry");
    ExpectClose(o16, o8, std::size_t{kBatch} * kQWidth,
                "WmmaCausalAttention" + kind);
  }

  // MTP draft: K/V-only preparation (zero query heads), then decode.
  pos.Up({start + kBatch - 3});
  if (!q::PrepareAttention(packed.p, 2 * kKvWidth, nullptr, gamma.p, nullptr,
                           nullptr, k16.p, v16.p, 3, 0, kKvHeads, kDim, 64,
                           pos.p, 1e7F, 1e-6F, nullptr) ||
      !q::PrepareAttention(packed.p, 2 * kKvWidth, nullptr, gamma.p, nullptr,
                           nullptr, k8.p, v8.p, 3, 0, kKvHeads, kDim, 64,
                           pos.p, 1e7F, 1e-6F, nullptr, nullptr, false,
                           q::KvType::kQ8))
    throw std::runtime_error("draft preparation rejected the geometry");
  q::Attention(q16.p, k16.p, v16.p, nullptr, words, o16.p, parts.p, 8, 3,
               pos.p, kHeads, kKvHeads, kDim, kRatio, nullptr);
  q::Attention(q16.p, k8.p, v8.p, nullptr, words, o8.p, parts.p, 8, 3, pos.p,
               kHeads, kKvHeads, kDim, kRatio, nullptr, q::KvType::kQ8);
  ExpectClose(o16, o8, 3 * kQWidth, "MTP draft attention" + at);
}

}  // namespace

int main() {
  try {
    static_assert(kQ8Row * 2 <= 0.55 * kKvWidth * 2, "q8 rows <= 0.55 x F16");
    RunAt(300);
    RunAt(70000);
    std::cout << "kv_q8_ops: q8_0 cache within tolerance\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "kv_q8_ops failed: " << e.what() << '\n';
    return 1;
  }
}
