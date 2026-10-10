@AGENTS.md

# gxs local build (Normandy)

Branch `gxs` ports UD-IQ4_XS support onto upstream `main` `fd747a5`
(`v0.9.1` + #485).
Spec: `/data/obsidian/topics/homelab/normandy-sr2/specs/gxs.md`. Never push
this branch to any remote.

Nix is not installed here. Every build and test command runs inside the
`gufo-dev:rocm7.2.3-gcc15.3` container (ROCm 7.2.3, GCC 15.3.0; the host keeps
ROCm 7.2.2). The Dockerfile and wrapper live in
`~/git/normandy-sr2/gufo-dev/`. The wrapper mounts `${GUFO_SRC:-~/git/gufo}`
as its working directory, the Flash-Next models (read-only) and
`/data/artifacts/normandy-sr2/gxs`, and passes through only the Strix Halo
iGPU (gfx1151). It ignores the current directory: for a worktree, run
`GUFO_SRC=<worktree> $RUN ...`, or it silently builds `~/git/gufo`.

```sh
RUN=~/git/normandy-sr2/gufo-dev/run.sh   # run from ~/git/gufo

# Image (only when the Dockerfile changes)
docker build -t gufo-dev:rocm7.2.3-gcc15.3 ~/git/normandy-sr2/gufo-dev

# Configure and build
$RUN cmake --preset gpu-test
$RUN cmake --build --preset gpu-test --parallel 24

# Unit test (one target)
$RUN cmake --build --preset gpu-test --target <test-target>
$RUN ctest --preset gpu-full -R '^<test-name>$' --output-on-failure

# Full suite
$RUN ctest --preset gpu-fast --output-on-failure
```

Stop production Gufo before GPU tests or probes: `docker stop gufo`. Leave it
stopped if it was already down.

`gpu-fast` baseline on `fd747a5` in gufo-dev: 125 tests, 122 pass, 3 skipped
(`qwen3_tts_loader_test`, `qwen3_tts_tokenizer_test`,
`qwen3_tts_reference_artifacts_test`: Qwen3-TTS artifacts absent). Any other
skip or failure is a regression. This needs the 112 B `GdnRowSplitKernel`
scratch allowance in `tools/ci/kernel-resources.json`: upstream made that file
with Nix AMD clang 22, and gufo-dev (ROCm 7.2.3, GCC 15.3, its clang) differs.
`kernel_resources_test` allowances for the gxs IQ kernels also live there.

## Reference probe

```sh
$RUN cmake --build --preset gpu-test --target qwen38_flash_next_gpu_probe
$RUN build/gpu-test/tests/models/qwen38_flash_next/qwen38_flash_next_gpu_probe \
  --model <first shard .gguf> \
  --prompt-file tests/models/qwen38_flash_next/data/gxs-probe-prompt.txt \
  --reference
```

Models: `/data/ai-stack/llm/models/qwen3.8-flash-next/{UD-IQ4_XS,UD-Q4_K_XL,MTP}/`.
The probe compares only the last `min(batch, 64)` prompt positions against
the CPU F32 reference (`max_logit_rows`). The fixed prompt fits one 512-token
batch. Baseline output: `/data/artifacts/normandy-sr2/gxs/baseline/`.
