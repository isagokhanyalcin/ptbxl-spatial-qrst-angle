"""Downloads Uni-G median beats (PTB-XL+ v1.0.1) for all PTB-XL records. Usage: python3 00_download_median_beats.py <ptbxl_database.csv> <out_dir>"""
import sys, os, time, pandas as pd, concurrent.futures as cf, urllib.request
DB, MB = sys.argv[1], sys.argv[2]
B = "https://physionet-open.s3.amazonaws.com/ptb-xl-plus/1.0.1/median_beats/unig"
ids = pd.read_csv(DB).ecg_id.tolist()
def fetch(i):
    sub = f"{(i // 1000) * 1000:05d}"
    for ext in ["hea", "dat"]:
        p = f"{MB}/{i:06d}_medians.{ext}"
        if os.path.exists(p) and os.path.getsize(p) > 0: continue
        for k in range(4):
            try:
                urllib.request.urlretrieve(f"{B}/{sub}/{i:06d}_medians.{ext}", p + ".part"); os.replace(p + ".part", p); break
            except Exception as e:
                if "404" in str(e): return (i, "404")
                time.sleep(2)
    return (i, "ok")
t0 = time.time(); res = []
with cf.ThreadPoolExecutor(32) as ex:
    for n, r in enumerate(ex.map(fetch, ids)):
        res.append(r)
        if n % 2000 == 0: print(n, round(time.time() - t0), "s", flush=True)
print("done", sum(r[1] == "ok" for r in res), "missing", [r[0] for r in res if r[1] != "ok"][:20], flush=True)
