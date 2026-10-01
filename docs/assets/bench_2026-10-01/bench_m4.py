import subprocess, sys, time, json, statistics as st
model, tag = sys.argv[1], sys.argv[2]
backends = sys.argv[3:]
P = "Write a detailed 1000-word essay on how neural networks learn through backpropagation, including the role of gradients and the chain rule."
# "metal" = path to an unpacked v0.6.0 -metal release binary (with mlx.metallib beside it)
BIN = {"cpu": "sapient", "metal": "./metal/sapient"}
LO, HI = 16, 400
def run(bk, n):
    t = time.perf_counter()
    r = subprocess.run([BIN[bk], "chat", model, "-p", P, "-n", str(n), "--backend", bk, "--raw"], capture_output=True, text=True)
    dt = time.perf_counter() - t
    assert r.returncode == 0, r.stderr[-400:]
    assert "token cap" in r.stderr, "EOS before cap"
    return dt
def llama(bk):
    args = ["llama-bench", "-m", model, "-p", "0", "-n", "128", "-r", "3", "-o", "json"] + (["-t", "4", "-ngl", "0"] if bk == "cpu" else ["-ngl", "99"])
    d = json.loads(subprocess.run(args, capture_output=True, text=True).stdout)[0]
    return d["avg_ts"], d["stddev_ts"]
for bk in backends:
    run(bk, LO)  # warm page cache
    lo, hi, ll = [], [], []
    for i in range(3):
        lo.append(run(bk, LO)); hi.append(run(bk, HI)); time.sleep(10)
        ll.append(llama(bk)); time.sleep(10)
    per = [(HI-LO)/(h-l) for h, l in zip(hi, lo)]
    print(f"{tag} {bk} SAPIENT decode tok/s per-round={[round(x,1) for x in per]} min-based={(HI-LO)/(min(hi)-min(lo)):.1f}  (T16={[round(x,2) for x in lo]} T400={[round(x,2) for x in hi]})", flush=True)
    print(f"{tag} {bk} llama.cpp tg128 avg±sd per-round={[(round(a,1), round(s,1)) for a,s in ll]}", flush=True)
