@AGENTS.md

# gxs local build (Normandy)

Branch `gxs-<upstream-short>` carries the gxs changes (UD-IQ4_XS routed experts, Q6_K -> Q8_0 head, q8_0 KV cache) on upstream `main`. Spec: `/data/obsidian/topics/homelab/normandy-sr2/specs/gxs.md`. The branch lives in `terminus-labs-ai/gufo`; the IQ experts part goes upstream separately.

Build and test with upstream's flake toolchain, inside a `nixos/nix` container on Normandy: the host has no Nix. The wrapper is `normandy-sr2/gufo-nix/run.sh`; Normandy keeps a copy at `/data/worktrees/gufo/nixrun.sh`. It runs as root in rootful Podman, keeps `/nix` in the `gufo-nix` volume, mounts `$GUFO_SRC` at its own path, the models (read-only) and `/data/artifacts/normandy-sr2`, and passes through only the Strix Halo iGPU (gfx1151). `GUFO_SRC` is required.

```sh
RUN="env GUFO_SRC=$PWD /data/worktrees/gufo/nixrun.sh"   # from the checkout

$RUN nix develop -c cmake --preset gpu-test
$RUN nix develop -c cmake --build --preset gpu-test --parallel 24 -- -k 0
$RUN nix develop -c ctest --preset gpu-full -R '^<test-name>$' --output-on-failure
$RUN nix develop -c ctest --preset gpu-fast --output-on-failure        # full suite
$RUN nix shell --inputs-from . nixpkgs#clang-tools nixpkgs#python3 -c python3 tools/ci/check-format.py
$RUN nix build .#release --out-link /data/artifacts/normandy-sr2/<session>/result-release
```

Stop production Gufo before GPU tests or probes (`sudo systemctl stop gufo`) and start it again afterwards; leave it stopped if it was already down. Build output is root-owned.

`gpu-fast` on `9cde650f` + gxs: 262 tests, 253 pass, 6 skipped (three Qwen3-TTS artifact tests, three NPU device tests: the wrapper passes no NPU). Three failures also fail on unmodified upstream with this toolchain and are not gxs regressions: `qwen_image_21_weight_upload_test` (does not compile), `persistent_deltanet_test` (hipMemcpy invalid argument) and `kernel_resources_test` (nine non-gxs kernels over upstream's allowances). Any other failure is a regression.

`tools/ci/kernel-resources.json` is upstream's file plus the 54 IQ `qfn_mmq::mul_mat_q` kernels (ggml types 20, 21 and 23), measured from the release package with `tools/ci/check-kernel-resources.py <result>/bin/.gufo-wrapped --write-baseline`. `bin/gufo` is a wrapper script; the ELF is `.gufo-wrapped`.

## Reference probe

```sh
$RUN nix develop -c cmake --build --preset gpu-test --target qwen38_flash_next_gpu_probe
$RUN nix develop -c build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model <first shard .gguf> \
  --prompt-file tests/models/qwen38_flash_next/data/gxs-probe-prompt.txt \
  --reference [--kv-cache-type q8_0]
```

Models: `/data/ai-stack/llm/models/qwen3.8-flash-next/{UD-IQ4_XS,MTP}/`; production uses the `-of-00003` UD-IQ4_XS shards. The probe compares the last 64 prompt positions against the CPU F32 reference. On `9cde650f` + gxs: f16 58/64 top-1, mean KL 0.0250; q8_0 60/64, mean KL 0.0259.

## Live KV type check

`tests/cli/kv_cache_type_test.py GUFO MODEL MTP_MODEL CACHE_DIR OUT_JSON` serves a ~200K-token needle at 262K context in f16, q8_0, then f16 again, and checks that a disk checkpoint never crosses KV types. Give it an empty cache directory.
