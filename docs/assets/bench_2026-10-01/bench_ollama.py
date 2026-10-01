import json, urllib.request
P = "Write a detailed 1000-word essay on how neural networks learn through backpropagation, including the role of gradients and the chain rule."
for m in ["qwen2.5:1.5b", "llama3.2:1b"]:
    out = []
    for i in range(4):
        req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps({"model": m, "prompt": P, "stream": False, "options": {"num_predict": 128, "temperature": 0}}).encode(), headers={"Content-Type": "application/json"})
        d = json.loads(urllib.request.urlopen(req, timeout=300).read())
        out.append(round(d["eval_count"] / (d["eval_duration"] / 1e9), 1))
    print(m, "Ollama decode tok/s (run1 cold-ish):", out)
