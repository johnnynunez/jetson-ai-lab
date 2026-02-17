---
title: "Cosmos Reasoning 2B on Jetson with vLLM"
description: "Run NVIDIA Cosmos Reasoning 2B FP8 Model on Jetson devices using vLLM, and connect it to Live VLM WebUI for real-time vision inference."
category: "Multimodal"
section: "Vision Language Models"
order: 2
tags: ["vlm", "vision", "cosmos", "cosmos-reasoning", "vllm", "fp8", "jetson-orin", "jetson-thor", "ngc", "live-vlm-webui", "multimodal", "reasoning"]
model: "vllm"
isNew: true
---

[NVIDIA Cosmos Reasoning 2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) is a compact vision-language model with built-in chain-of-thought reasoning capabilities. Despite its small 2B parameter size, it can perform spatial reasoning, anomaly detection, and detailed scene analysis, making it well-suited for edge deployment on Jetson.

This tutorial walks through serving **Cosmos Reasoning 2B FP8 Model** with **vLLM** on Jetson, and connecting it to **[Live VLM WebUI](https://github.com/NVIDIA-AI-IOT/live-vlm-webui)** for real-time webcam-based inference.

---

## Prerequisites

**Supported Devices:**
- Jetson AGX Thor Developer Kit
- Jetson AGX Orin (64GB / 32GB)
- Jetson Orin Super Nano

**JetPack Version:**
- JetPack 6 (L4T r36.x) — for Orin devices
- JetPack 7 (L4T r38.x) — for Thor

**Storage:** NVMe SSD **required**
- ~5 GB for the FP8 model weights
- ~8 GB for the vLLM container image

**Accounts:**
- [NVIDIA NGC](https://ngc.nvidia.com/) account (free) — needed for NGC CLI and model download

---

## Overview

| | Jetson AGX Thor | Jetson AGX Orin | Orin Super Nano |
|---|---|---|---|
| **vLLM Container** | `nvcr.io/nvidia/vllm:26.01-py3` | `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04` | `ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04` |
| **Model** | FP8 via NGC (volume mount) | FP8 via NGC (volume mount) | FP8 via NGC (volume mount) |
| **Max Model Length** | 8192 tokens | 8192 tokens | 768 tokens (memory-constrained) |
| **GPU Memory Util** | 0.8 | 0.8 | 0.52 |

The workflow is the same for both devices:

1. **Download** the FP8 model checkpoint via NGC CLI
2. **Pull** the vLLM Docker image for your device
3. **Launch** the container with the model mounted as a volume
4. **Connect** Live VLM WebUI to the vLLM endpoint

---

## Step 1: Install the NGC CLI

The NGC CLI lets you download model checkpoints from the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/?tab=model).

### Download and install

```bash
mkdir -p ~/Projects/CosmosReasoning
cd ~/Projects/CosmosReasoning

# Download the NGC CLI for ARM64
# Get the latest installer URL from: https://org.ngc.nvidia.com/setup/installers/cli
wget -O ngccli_arm64.zip https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/4.13.0/files/ngccli_arm64.zip
unzip ngccli_arm64.zip
chmod u+x ngc-cli/ngc

# Add to PATH
export PATH="$PATH:$(pwd)/ngc-cli"
```

### Configure the CLI

```bash
ngc config set
```

You will be prompted for:
- **API Key** — generate one at [NGC API Key setup](https://org.ngc.nvidia.com/setup/api-key)
- **CLI output format** — choose `json` or `ascii`
- **org** — press Enter to accept the default

---

## Step 2: Download the Model

Download the **FP8 quantized** checkpoint. This is used on all Jetson devices:

```bash
cd ~/Projects/CosmosReasoning
ngc registry model download-version "nim/nvidia/cosmos-reason2-2b:1208-fp8-static-kv8"
```

This creates a directory called `cosmos-reason2-2b_v1208-fp8-static-kv8/` containing the model weights. Note the full path — you will mount it into the Docker container as a volume.

---

## Step 3: Pull the vLLM Docker Image

### For Jetson AGX Thor

```bash
docker pull nvcr.io/nvidia/vllm:26.01-py3
```

### For Jetson AGX Orin / Orin Super Nano

```bash
docker pull ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04
```

---

## Step 4: Serve Cosmos Reasoning 2B with vLLM

### Option A: Jetson AGX Thor

Thor has ample GPU memory and can run the model with generous context length.

Set the path to your downloaded model and free cached memory on the host:

```bash
MODEL_PATH="$HOME/Projects/CosmosReasoning/cosmos-reason2-2b_v1208-fp8-static-kv8"
sudo sysctl -w vm.drop_caches=3
```

**Launch the container with the model mounted:**

```bash
docker run --rm -it \
  --runtime nvidia \
  --network host \
  --ipc host \
  -v "$MODEL_PATH:/models/cosmos-reason2-2b:ro" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  nvcr.io/nvidia/vllm:26.01-py3 \
  bash
```

**Inside the container, activate the environment and serve the model:**

```bash
vllm serve /models/cosmos-reason2-2b \
  --max-model-len 8192 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.8
```

> **Note:** The `--reasoning-parser qwen3` flag enables chain-of-thought reasoning extraction. The `--media-io-kwargs` flag configures video frame handling.

Wait until you see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Option B: Jetson AGX Orin

AGX Orin has enough memory to run the model with the same generous parameters as Thor.

Set the path to your downloaded model and free cached memory on the host:

```bash
MODEL_PATH="$HOME/Projects/CosmosReasoning/cosmos-reason2-2b_v1208-fp8-static-kv8"
sudo sysctl -w vm.drop_caches=3
```

**1. Launch the container:**

```bash
docker run --rm -it \
  --runtime nvidia \
  --network host \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$MODEL_PATH:/models/cosmos-reason2-2b:ro" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04 \
  bash
```

**2. Inside the container, activate the environment and serve:**

```bash
cd /opt/
source venv/bin/activate

vllm serve /models/cosmos-reason2-2b \
  --max-model-len 8192 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.8
```

Wait until you see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Option C: Jetson Orin Super Nano (memory-constrained)

The Orin Super Nano has significantly less RAM, so we need aggressive memory optimization including reducing the model's default image resolution to fit within the available memory.

**1. Reduce the model's image resolution config:**

The default `preprocessor_config.json` allows images up to 16M pixels, which produces too many tokens for the Orin Super Nano's limited context window. Reduce it on the host before launching Docker:

```bash
cd ~/Projects/CosmosReasoning/cosmos-reason2-2b_v1208-fp8-static-kv8
cp preprocessor_config.json preprocessor_config.json.bak

python3 -c "
import json
with open('preprocessor_config.json') as f:
    cfg = json.load(f)
print(f'Old longest_edge: {cfg[\"size\"][\"longest_edge\"]}')
print(f'Old shortest_edge: {cfg[\"size\"][\"shortest_edge\"]}')
cfg['size']['longest_edge'] = 50176
cfg['size']['shortest_edge'] = 3136
with open('preprocessor_config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
print('New longest_edge: 50176')
print('New shortest_edge: 3136')
"
```

This limits input images to ~50K pixels, keeping image tokens small enough to fit in the constrained context window.

**2. Set the model path and free cached memory:**

```bash
MODEL_PATH="$HOME/Projects/CosmosReasoning/cosmos-reason2-2b_v1208-fp8-static-kv8"
sudo sysctl -w vm.drop_caches=3
```

**3. Launch the container:**

```bash
docker run --rm -it \
  --runtime nvidia \
  --network host \
  -v "$MODEL_PATH:/models/cosmos-reason2-2b:ro" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  ghcr.io/nvidia-ai-iot/vllm:r36.4-tegra-aarch64-cu126-22.04 \
  bash
```

**4. Inside the container, activate the environment and serve:**

```bash
cd /opt/
source venv/bin/activate

vllm serve /models/cosmos-reason2-2b \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --enforce-eager \
  --max-model-len 768 \
  --max-num-batched-tokens 768 \
  --gpu-memory-utilization 0.52 \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --limit-mm-per-prompt '{"image":1}'
```

**Key flags explained (Orin Super Nano only):**

| Flag | Purpose |
|------|---------|
| `--enforce-eager` | Disables CUDA graphs to save memory |
| `--max-model-len 768` | Context window sized for image tokens + output |
| `--max-num-batched-tokens 768` | Matches the model length limit |
| `--gpu-memory-utilization 0.52` | Uses most available memory (Orin Super Nano has ~3.9 GiB free of 7.4 GiB) |
| `--max-num-seqs 1` | Single request at a time to minimize memory |
| `--enable-chunked-prefill` | Processes prefill in chunks for memory efficiency |
| `--limit-mm-per-prompt` | Limits to 1 image per prompt (Live VLM WebUI sends frames as images) |
| `VLLM_SKIP_WARMUP=true` | Skips warmup to save time and memory |

Wait until you see the server is ready:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Verify the server is running

From another terminal on the Jetson:

```bash
curl http://localhost:8000/v1/models
```

You should see the model listed in the response.

---

## Step 5: Test with a Quick API Call

Before connecting the WebUI, verify the model responds correctly:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/cosmos-reason2-2b",
    "messages": [
      {
        "role": "user",
        "content": "What capabilities do you have?"
      }
    ],
    "max_tokens": 128
  }' | python3 -m json.tool
```

> **Tip:** The model name used in the API request must match what vLLM reports. Verify with `curl http://localhost:8000/v1/models`.

---

## Step 6: Connect to Live VLM WebUI

[Live VLM WebUI](https://github.com/NVIDIA-AI-IOT/live-vlm-webui) provides a real-time webcam-to-VLM interface. With vLLM serving Cosmos Reasoning 2B, you can stream your webcam and get live AI analysis with reasoning.

### Install Live VLM WebUI

The easiest method is pip (Open another terminal):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd ~/Projects/CosmosReasoning
uv venv .live-vlm --python 3.12
source .live-vlm/bin/activate
uv pip install live-vlm-webui
live-vlm-webui
```

Or use Docker:

```bash
git clone https://github.com/nvidia-ai-iot/live-vlm-webui.git
cd live-vlm-webui
./scripts/start_container.sh
```

### Configure the WebUI

1. Open **`https://localhost:8090`** in your browser
2. Accept the self-signed certificate (click **Advanced** → **Proceed**)
3. In the **VLM API Configuration** section on the left sidebar:
   - Set **API Base URL** to `http://localhost:8000/v1`
   - Click the **Refresh** button to detect the model
   - Select the Cosmos Reasoning 2B model from the dropdown
4. Select your camera and click **Start**

The WebUI will now stream your webcam frames to Cosmos Reasoning 2B and display the model's analysis in real-time.

### Required WebUI settings for Orin Super Nano

> **Important:** On Orin Super Nano, vLLM is configured with `--max-model-len 768`. The WebUI defaults to `max_tokens: 512`, which will cause requests to fail with a `400 Bad Request` error since image tokens consume most of the context window. You **must** lower Max Tokens before starting analysis.

In the WebUI left sidebar, adjust these settings **before clicking Start**:

- **Max Tokens**: Set to **150** (image tokens use ~500-600 of the 768 context, leaving ~150-200 for output)
- **Frame Processing Interval**: Set to **60+** (gives the model time between frames)
- Use **short prompts** — longer prompts consume more input tokens, leaving fewer for the response

---

## Troubleshooting

### Out of memory on Orin

**Problem:** vLLM crashes with CUDA out-of-memory errors.

**Solution:**
1. Free system memory before starting:
   ```bash
   sudo sysctl -w vm.drop_caches=3
   ```
2. Lower `--gpu-memory-utilization` further (try `0.45` or `0.40`)
3. Reduce `--max-model-len` further (try `128`)
4. Make sure no other GPU-intensive processes are running

### "max_tokens is too large" errors on Orin Super Nano

**Problem:** vLLM returns `400 Bad Request` with the error `max_tokens is too large` or `decoder prompt is longer than the maximum model length`.

**Solution:**
- This happens because image tokens consume most of the 768 token context window (~500-600 tokens for a single image), and the WebUI defaults to `max_tokens: 512`
- In the WebUI, set **Max Tokens** to **150** before starting analysis
- Make sure you edited `preprocessor_config.json` to reduce `longest_edge` to `50176` (Step 4, Option C) — without this, images produce too many tokens to fit in the context

### Model not found in WebUI

**Problem:** The model doesn't appear in the Live VLM WebUI dropdown.

**Solution:**
1. Verify vLLM is running: `curl http://localhost:8000/v1/models`
2. Make sure the WebUI API Base URL is set to `http://localhost:8000/v1` (not `https`)
3. If vLLM and WebUI are in separate containers, use `http://<jetson-ip>:8000/v1` instead of `localhost`

### Slow inference on Orin

**Problem:** Each response takes a very long time.

**Solution:**
- This is expected with the memory-constrained configuration. Cosmos Reasoning 2B FP8 on Orin prioritizes fitting in memory over speed.
- Reduce `max_tokens` in the WebUI to get shorter, faster responses
- Increase the frame interval so the model isn't constantly processing new frames

### vLLM fails to load model

**Problem:** vLLM reports the model path doesn't exist or can't be loaded.

**Solution:**
- Verify the NGC download completed successfully: `ls ~/Projects/CosmosReasoning/cosmos-reason2-2b_v1208-fp8-static-kv8/`
- Make sure the volume mount path is correct in your `docker run` command
- Check that the model directory is mounted as read-only (`:ro`) and the path inside the container matches what you pass to `vllm serve`

---

## Summary

You now have **NVIDIA Cosmos Reasoning 2B** running on your Jetson with vLLM, connected to a real-time webcam interface:

- **Jetson AGX Thor**: Full model with 8192 token context, ideal for detailed reasoning tasks
- **Jetson AGX Orin / Orin Super Nano**: FP8 quantized model with optimized memory settings, bringing reasoning capabilities to smaller Jetson devices

The combination of Cosmos Reasoning 2B's chain-of-thought capabilities with Live VLM WebUI's real-time streaming makes it straightforward to prototype and evaluate vision AI applications at the edge.

---

## Additional Resources

- **Cosmos Reasoning 2B on NVIDIA Build**: [https://huggingface.co/nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)
- **NGC Model Catalog**: [https://catalog.ngc.nvidia.com/](https://catalog.ngc.nvidia.com/)
- **Live VLM WebUI**: [https://github.com/NVIDIA-AI-IOT/live-vlm-webui](https://github.com/NVIDIA-AI-IOT/live-vlm-webui)
- **vLLM Containers for Jetson**: [Supported Models](/models)
- **NGC CLI Installers**: [https://org.ngc.nvidia.com/setup/installers/cli](https://org.ngc.nvidia.com/setup/installers/cli)
