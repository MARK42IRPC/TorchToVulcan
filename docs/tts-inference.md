# TTS 推理端 / TTS Listening Console

`/tts` 是一个独立的 ONNX Runtime CPU 推理端，用于先验证 TTS 的输入约定、音频输出和听感，再接入 Vulkan runtime。它故意不承诺速度：当前阶段的验收标准是“能稳定生成 WAV，并且可以反复听取和下载”。

## 使用流程

1. 启动开发服务：`.\scripts\dev.ps1`（或双击根目录的 `start.bat`）。
2. 打开 `http://127.0.0.1:5173/tts`，或者在图检查器顶部点击 `VOICE LAB`。
3. 上传一个 `.onnx` 文件或包含 ONNX 的 `.zip`。ZIP 中可以有多个 ONNX，页面会列出每个模型的输入输出契约。
4. 输入文本并点击 `Synthesize & Listen`。成功后页面会自动尝试播放 WAV，也可以使用浏览器控件和下载按钮回归检查。

## 输入契约

不同 TTS 导出没有统一的输入名。后端会根据输入名称和 dtype 提供最小默认值：字符串输入得到原文，`input_ids`/`tokens`/`phoneme_ids` 得到确定性的字符 token，`length` 得到 token 长度，speaker/language 默认 0，scale 默认 1.0。

这些默认值只适合冒烟验证，不能代替真实 tokenizer、音素化器或 speaker 配置。模型面板会展示实际输入名、dtype 和 shape；需要精确控制时，在 `INPUT OVERRIDES` 中填写 JSON object，例如：

```json
{
  "input_ids": [[101, 2057, 102]],
  "input_lengths": [3],
  "sid": [0]
}
```

覆盖值会按 ONNX 输入 dtype 和静态尾维度转换、截断或补零。动态 shape、特殊 token、声学模型的额外条件仍应由模型自己的 tokenizer/config 提供。

## 输出与 vocoder

当前版本只把明确的单声道 waveform 当作可播放音频，支持以下常见形状：`[samples]`、`[1, samples]`、`[samples, 1]`、`[1, 1, samples]`。整数和浮点输出都会转换成单声道 16-bit PCM WAV；采样率默认从同目录 `config.json` 的 `sample_rate`/`sampling_rate`/`sr` 读取，否则使用 22050 Hz，也可以在页面覆盖。

如果输出名或形状表明它是 Mel 频谱（例如 `mel`、`spectrogram`，或常见的 `[1, 80, frames]`），后端会明确提示“需要 vocoder”，不会把频谱直接 flatten 成错误的音频。下一阶段可以在同一个 TTS 工作台增加声学模型 -> vocoder 的双模型链路。

## 当前限制

- 仓库内的 `artifacts/live.onnx` 只是 ReLU 测试模型，不是真实语音权重；它只能验证 API/WAV 链路。
- 未接入 Vulkan/GPU，推理 provider 固定为 `CPUExecutionProvider`。
- 未实现 ckpt/pth、专用 tokenizer、speaker 管理、音频波形可视化和 vocoder 链路。
- 音频结果只保留当前进程最近 16 条，重启服务后清空。

真实模型回归时，建议把模型、配置、词表和 tokenizer 导出物一起放进 ZIP，并先在模型面板核对输入输出，再用 `INPUT OVERRIDES` 对齐原项目的输入张量。

## English

`/tts` is a dedicated ONNX Runtime CPU listening console. It is intentionally a correctness and perceptual-regression endpoint, not a performance benchmark. Upload an ONNX model or a ZIP containing one or more ONNX files, inspect the discovered contract, enter text, and synthesize a downloadable WAV.

Because TTS exports use different contracts, the backend exposes JSON input overrides. Defaults are only smoke-test values; a real model still needs its tokenizer, phonemizer, vocabulary, speaker IDs, and configuration. Direct playback accepts mono waveform tensors such as `[samples]`, `[1, samples]`, `[samples, 1]`, and `[1, 1, samples]`. Mel/spectrogram outputs are rejected with a vocoder hint instead of being flattened into invalid audio.
