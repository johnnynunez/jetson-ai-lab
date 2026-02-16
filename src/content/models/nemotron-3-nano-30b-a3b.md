---
title: "Nemotron3 Nano 30B-A3B"
model_id: "nemotron-3-nano-30b-a3b"
short_description: "NVIDIA's flagship hybrid MoE reasoning model with 30B total / 3.5B active parameters"
family: "NVIDIA Nemotron"
icon: "⚡"
is_new: true
order: 1
type: "Text"
memory_requirements: "32GB RAM"
precision: "FP4"
model_size: "17GB"
hf_checkpoint: "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
huggingface_url: "https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
build_nvidia_url: "https://build.nvidia.com/nvidia/nemotron-3-nano-30b-a3b"
minimum_jetson: "Thor"
supported_inference_engines:
  - engine: "vLLM"
    type: "Container"
    run_command_thor: |
      sudo docker run -it --rm --pull always --runtime=nvidia \
        -p ${VLLM_PORT:-8000}:${VLLM_PORT:-8000} \
        -e HF_TOKEN=$HF_TOKEN -e VLLM_PORT=${VLLM_PORT:-8000} \
        -e VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.8} \
        -e VLLM_USE_FLASHINFER_MOE_FP4=1 -e VLLM_FLASHINFER_MOE_BACKEND=throughput \
        -v $HOME/.cache/huggingface:/root/.cache/huggingface \
        nvcr.io/nvidia/vllm:25.12.post1-py3 \
        bash -c "wget -q -O /tmp/nano_v3_reasoning_parser.py --header=\"Authorization: Bearer \$HF_TOKEN\" \
          https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/resolve/main/nano_v3_reasoning_parser.py && \
          vllm serve nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
            --port \$VLLM_PORT --gpu-memory-utilization \$VLLM_GPU_MEMORY_UTILIZATION \
            --trust-remote-code --enable-auto-tool-choice --tool-call-parser qwen3_coder \
            --reasoning-parser-plugin /tmp/nano_v3_reasoning_parser.py --reasoning-parser nano_v3 --kv-cache-dtype fp8"
---

**Run command options:** Set `VLLM_PORT` (default `8000`) and `VLLM_GPU_MEMORY_UTILIZATION` (default `0.8`) before running. If you see *"Free memory on device ... is less than desired GPU memory utilization"*, lower the latter (e.g. `export VLLM_GPU_MEMORY_UTILIZATION=0.12` when only ~14 GiB GPU is free).

## Architecture

The model employs a hybrid Mixture-of-Experts (MoE) architecture:
- 23 Mamba-2 and MoE layers
- 6 Attention layers
- 128 experts + 1 shared expert per MoE layer
- 6 experts activated per token
- **3.5B active parameters** / **30B total parameters**

## Inputs and Outputs

**Input:** Text

**Output:** Text

## Intended Use Cases

- **AI Agent Systems**: Build autonomous agents with strong reasoning capabilities
- **Chatbots**: General purpose conversational AI
- **RAG Systems**: Retrieval-augmented generation applications
- **Reasoning Tasks**: Complex problem-solving with configurable reasoning traces
- **Instruction Following**: General instruction-following tasks

## Supported Languages

English, Spanish, French, German, Japanese, Italian, and coding languages.

## Reasoning Configuration

The model's reasoning capabilities can be configured through a flag in the chat template:
- **With reasoning traces**: Higher-quality solutions for complex queries
- **Without reasoning traces**: Faster responses with slight accuracy trade-off for simpler tasks

### Skipping reasoning (minimize TTFT)

For low-latency or single-token tasks (e.g. picking a number for a pre-scripted response), disable reasoning so the model does not generate a `<think>` block first:

- **Per request**: Pass `extra_body={"chat_template_kwargs": {"enable_thinking": false}}` in your chat completion call, and use `max_tokens=1` (or 2) if you only need one token.
- **Server default**: Add `--default-chat-template-kwargs '{"enable_thinking": false}'` to the `vllm serve` command so all requests skip reasoning by default and TTFT stays minimal.
