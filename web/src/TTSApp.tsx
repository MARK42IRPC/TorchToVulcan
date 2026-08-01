import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { AudioLines, Download, FileUp, LoaderCircle, Play, RotateCcw, Upload, Volume2 } from "lucide-react";

type IOInfo = { name: string; data_type: string; shape: string[] };
type ModelInfo = {
  model_id: string;
  name: string;
  path: string;
  source_name: string;
  inputs: IOInfo[];
  outputs: IOInfo[];
  sample_rate: number;
};
type SynthesisResult = {
  audio_url: string;
  audio_id: string;
  sample_rate: number;
  duration_ms: number;
  samples: number;
  output: IOInfo;
  inputs: Record<string, { dtype: string; shape: number[] }>;
};

const API_ROOT = "";

function typeLabel(item: IOInfo): string {
  return `${item.data_type} [${item.shape.join(", ") || "scalar"}]`;
}

function formatDuration(milliseconds: number): string {
  return `${(milliseconds / 1000).toFixed(2)} s`;
}

export default function TTSApp() {
  const fileRef = useRef<HTMLInputElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState("");
  const [text, setText] = useState("请在这里输入一段文字，试听 ONNX TTS 推理结果。");
  const [overrides, setOverrides] = useState("{}");
  const [sampleRate, setSampleRate] = useState("");
  const [result, setResult] = useState<SynthesisResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");

  const selectedModel = useMemo(
    () => models.find((item) => item.model_id === selectedModelId) ?? models[0] ?? null,
    [models, selectedModelId],
  );

  useEffect(() => {
    void fetch(`${API_ROOT}/api/tts/models`)
      .then((response) => response.json())
      .then((payload: { models?: ModelInfo[] }) => {
        const next = payload.models ?? [];
        setModels(next);
        setSelectedModelId(next[0]?.model_id ?? "");
      })
      .catch(() => undefined);
  }, []);

  const upload = async (file: File) => {
    setUploading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${API_ROOT}/api/tts/models`, { method: "POST", body: form });
      const payload = (await response.json()) as { models?: ModelInfo[]; detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "TTS 模型加载失败");
      const next = payload.models ?? [];
      setModels((current) => [...current, ...next]);
      setSelectedModelId(next[0]?.model_id ?? "");
      setResult(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "TTS 模型加载失败");
    } finally {
      setUploading(false);
    }
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) void upload(file);
    event.target.value = "";
  };

  const synthesize = async () => {
    if (!selectedModel) {
      setError("请先加载 ONNX TTS 模型");
      return;
    }
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const form = new FormData();
      form.append("model_id", selectedModel.model_id);
      form.append("text", text);
      form.append("overrides", overrides || "{}");
      if (sampleRate) form.append("sample_rate", sampleRate);
      const response = await fetch(`${API_ROOT}/api/tts/synthesize`, { method: "POST", body: form });
      const payload = (await response.json()) as SynthesisResult & { detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? "TTS 推理失败");
      const next = { ...payload, audio_url: `${API_ROOT}${payload.audio_url}` };
      setResult(next);
      window.setTimeout(() => void audioRef.current?.play().catch(() => undefined), 80);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "TTS 推理失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="tts-shell">
      <header className="tts-topbar">
        <a className="tts-back" href="/">TTV GRAPH</a>
        <div className="tts-brand-mark" />
        <div>
          <strong>VOICE LAB / ONNX LISTENING CONSOLE</strong>
          <span>CPU reference inference · perceptual regression · 0.1</span>
        </div>
        <div className="tts-topbar__status"><span /> ORT CPU READY</div>
      </header>

      <main className="tts-main">
        <section className="tts-hero">
          <div className="tts-eyebrow"><AudioLines size={15} /> DEDICATED TTS INFERENCE ENDPOINT</div>
          <h1>把模型真正说出来。</h1>
          <p>加载 ONNX 或 ZIP，输入文本，直接听回归结果。这里优先保证音频链路和听感审查，不隐藏模型的输入约定。</p>
        </section>

        <section className="tts-grid">
          <aside className="tts-panel tts-model-panel">
            <div className="tts-panel-heading"><span>01 / MODEL BAY</span><b>{models.length} loaded</b></div>
            <input ref={fileRef} hidden type="file" accept=".onnx,.zip" onChange={onFileChange} />
            <button className="tts-upload" onClick={() => fileRef.current?.click()} disabled={uploading}>
              {uploading ? <LoaderCircle className="spin" size={19} /> : <Upload size={19} />}
              {uploading ? "Loading model..." : "Load ONNX / ZIP"}
            </button>
            <div className="tts-model-list">
              {models.length === 0 && <div className="tts-empty"><FileUp size={22} /><span>等待一个 TTS 模型</span><small>模型会保留在当前会话中</small></div>}
              {models.map((model) => (
                <button
                  className={`tts-model-card ${selectedModel?.model_id === model.model_id ? "is-selected" : ""}`}
                  key={model.model_id}
                  onClick={() => setSelectedModelId(model.model_id)}
                >
                  <span className="tts-model-dot" />
                  <span><strong>{model.name}</strong><small>{model.source_name} · {model.sample_rate} Hz</small></span>
                </button>
              ))}
            </div>
            {selectedModel && (
              <div className="tts-io-card">
                <div className="tts-io-title">MODEL CONTRACT</div>
                <div className="tts-io-group"><span>INPUTS</span>{selectedModel.inputs.map((item) => <div key={item.name}><b>{item.name}</b><small>{typeLabel(item)}</small></div>)}</div>
                <div className="tts-io-group"><span>OUTPUTS</span>{selectedModel.outputs.map((item) => <div key={item.name}><b>{item.name}</b><small>{typeLabel(item)}</small></div>)}</div>
              </div>
            )}
          </aside>

          <section className="tts-panel tts-compose-panel">
            <div className="tts-panel-heading"><span>02 / TEXT TO AUDIO</span><b>LISTENING PASS</b></div>
            <label className="tts-label" htmlFor="tts-text">TEXT PROMPT <span>{text.length} chars</span></label>
            <textarea id="tts-text" value={text} onChange={(event) => setText(event.target.value)} rows={7} />
            <div className="tts-control-row">
              <label><span>SAMPLE RATE OVERRIDE</span><input value={sampleRate} onChange={(event) => setSampleRate(event.target.value)} placeholder={String(selectedModel?.sample_rate ?? 22050)} inputMode="numeric" /></label>
              <label><span>RUN MODE</span><div className="tts-readonly"><Volume2 size={16} /> CPU / ONNX Runtime</div></label>
            </div>
            <label className="tts-label" htmlFor="tts-overrides">INPUT OVERRIDES <span>optional JSON</span></label>
            <textarea id="tts-overrides" className="tts-json" value={overrides} onChange={(event) => setOverrides(event.target.value)} rows={4} spellCheck={false} />
            <div className="tts-action-row">
              <button className="tts-primary" onClick={() => void synthesize()} disabled={busy || !selectedModel}>
                {busy ? <LoaderCircle className="spin" size={19} /> : <Play size={19} fill="currentColor" />}
                {busy ? "Running inference..." : "Synthesize & Listen"}
              </button>
              <button className="tts-secondary" onClick={() => { setResult(null); setError(""); }}><RotateCcw size={17} /> Reset result</button>
            </div>
            {error && <div className="tts-error">{error}</div>}
          </section>

          <section className="tts-panel tts-result-panel">
            <div className="tts-panel-heading"><span>03 / AUDIO REVIEW</span><b>{result ? "READY" : "NO TAKE"}</b></div>
            {!result ? (
              <div className="tts-result-empty"><div className="tts-wave-placeholder"><span /><span /><span /><span /><span /><span /><span /><span /><span /></div><strong>Audio output will appear here</strong><p>完成一次推理后自动播放 WAV，并显示采样率、时长和实际输出 tensor。</p></div>
            ) : (
              <div className="tts-result-ready">
                <div className="tts-result-badge"><AudioLines size={20} /> TAKE {result.audio_id.slice(0, 6).toUpperCase()}</div>
                <audio ref={audioRef} className="tts-audio" src={result.audio_url} controls />
                <div className="tts-metrics"><div><span>DURATION</span><strong>{formatDuration(result.duration_ms)}</strong></div><div><span>SAMPLES</span><strong>{result.samples.toLocaleString()}</strong></div><div><span>RATE</span><strong>{result.sample_rate} Hz</strong></div></div>
                <div className="tts-output-note"><span>SELECTED OUTPUT</span><b>{result.output.name}</b><small>{typeLabel(result.output)}</small></div>
                <a className="tts-download" href={result.audio_url} download><Download size={17} /> Download WAV</a>
              </div>
            )}
          </section>
        </section>
        <footer className="tts-footer"><span>LISTENING CONSOLE / {selectedModel ? selectedModel.name : "NO MODEL"}</span><span>Audio is generated locally by the Torch to Vulcan backend.</span></footer>
      </main>
    </div>
  );
}
