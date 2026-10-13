"""Live --kv-cache-type check for Qwen3.8-Flash-Next (GPU and model; not ctest).

Each server runs at --sessions=2 --context=262144 on one disk-cache directory
and records its load_completed line (kv_cache_type, session_state_bytes,
device use). It sends the fixed ~200K-token needle prompt
(tests/models/qwen38_flash_next/data/kv-needle.json) greedily and expects the
needle back. Order: f16 stores a checkpoint; q8_0 must not restore it and
prefills the whole prompt; f16 again restores it (positive control).

usage: kv_cache_type_test.py GUFO MODEL MTP_MODEL CACHE_DIR OUT_JSON [PORT]
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

NEEDLE = os.path.join(os.path.dirname(__file__), "..", "models",
                      "qwen38_flash_next", "data", "kv-needle.json")


def needle_prompt():
    spec = json.load(open(NEEDLE))
    at = int(spec["repeat"] * spec["needle_at"])
    body = (spec["filler"] * at + spec["needle"] +
            spec["filler"] * (spec["repeat"] - at))
    return body + spec["question"], spec["answer"]


def http(port, path, body=None, timeout=3600):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=data,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"{}")


def wait_for(log_path, pattern, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        match = re.search(pattern, open(log_path, errors="replace").read())
        if match:
            return match
        time.sleep(2)
    return None


def run(args, kv_type, label, prompt, expect_store):
    gufo, model, mtp, cache, out = args[:5]
    port = int(args[5]) if len(args) > 5 else 8732
    log_path = f"{os.path.splitext(out)[0]}-{label}.log"
    command = [gufo, "serve", "--host=127.0.0.1", f"--port={port}",
               "--sessions=2", "llm", f"--model={model}",
               "--context=262144", "--speculative=mtp", f"--mtp-model={mtp}",
               "--draft-tokens=3", "--cache-ram-bytes=2147483648",
               f"--cache-disk={cache}", "--cache-disk-bytes=68719476736",
               "--cache-disk-staging-bytes=6442450944",
               f"--kv-cache-type={kv_type}"]
    with open(log_path, "w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        loaded = wait_for(log_path, r"event=load_completed.*", 1200)
        assert loaded, f"{label}: server did not load"
        while True:
            try:
                http(port, "/ready", timeout=5)
                break
            except Exception:
                assert server.poll() is None, f"{label}: server exited"
                time.sleep(2)
        begin = time.time()
        reply = http(port, "/v1/completions",
                     {"prompt": prompt, "max_tokens": 16, "temperature": 0})
        elapsed = time.time() - begin
        if expect_store:
            assert wait_for(log_path, r"event=disk_cache action=stored", 600), (
                f"{label}: no checkpoint stored")
        text = open(log_path, errors="replace").read()
        fields = dict(re.findall(r"(\w+)=(\S+)", loaded.group(0)))
        return {
            "label": label, "kv_cache_type": fields.get("kv_cache_type"),
            "session_state_bytes": int(fields.get("session_state_bytes", 0)),
            "gpu_device_used_mib": int(fields.get("gpu_device_used_mib", 0)),
            "text": reply["choices"][0]["text"], "usage": reply["usage"],
            "request_seconds": round(elapsed, 1),
            "restored": "event=disk_cache action=restored" in text,
            "log": log_path,
        }
    finally:
        server.send_signal(signal.SIGTERM)
        server.wait(timeout=300)


def main():
    args = sys.argv[1:]
    prompt, answer = needle_prompt()
    first = run(args, "f16", "f16", prompt, True)
    q8 = run(args, "q8_0", "q8_0", prompt, False)
    control = run(args, "f16", "f16-restart", prompt, False)
    results = {"answer": answer, "runs": [first, q8, control]}
    f16_kv = first["session_state_bytes"]
    results["q8_session_state_fraction"] = q8["session_state_bytes"] / f16_kv
    results["load_footprint_saving_mib"] = (first["gpu_device_used_mib"] -
                                            q8["gpu_device_used_mib"])
    json.dump(results, open(args[4], "w"), indent=1)
    print(json.dumps(results, indent=1))
    for result in (first, q8):
        assert answer in result["text"], f"{result['label']} missed the needle"
    assert q8["kv_cache_type"] == "q8_0" and first["kv_cache_type"] == "f16"
    # AC 24: the f16 checkpoint is not restored by q8_0; it prefills.
    assert not q8["restored"] and q8["usage"]["cached_tokens"] == 0
    assert control["restored"] and control["usage"]["cached_tokens"] > 0
    print("kv_cache_type: needle found, checkpoints stay within their type")


if __name__ == "__main__":
    main()
