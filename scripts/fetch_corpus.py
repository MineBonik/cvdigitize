"""Download the open-access test corpus (CV / Pt-electrochemistry papers).

The corpus is git-ignored (large, reproducible). Run this to recreate it:
    .venv\\Scripts\\python.exe scripts\\fetch_corpus.py

Only openly downloadable sources are used (arXiv). Most publisher PDFs (MDPI,
RSC, ACS) block automated download with HTTP 403 bot-protection — for those,
download by hand in a browser and drop the PDF into data/corpus/.
"""
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS = os.path.join(ROOT, "data", "corpus")
os.makedirs(CORPUS, exist_ok=True)

# arXiv id -> local filename. These contain real voltammetry/CV figures
# (vector or rasterized) and make a varied end-to-end test set.
ARXIV = {
    "2012.00801": "arxiv_2012.00801_seven_steps_cv_dlcap.pdf",
    "2504.19892": "arxiv_2504.19892_gold_nanoelectrodes_cv.pdf",
    "2503.14758": "arxiv_2503.14758_ioncoupled_et_cv.pdf",
    "1407.1722":  "arxiv_1407.1722_porous_electrode_voltammetry.pdf",
    "2407.04814": "arxiv_2407.04814_solarcell_cv.pdf",
    "2102.05030": "arxiv_2102.05030_sweep_cv_sim.pdf",
    "2305.06810": "arxiv_2305.06810_photoec_transistor.pdf",
}
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")


def fetch(url, out):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as r, open(out, "wb") as f:
        f.write(r.read())
    return os.path.getsize(out)


def main():
    print(f"Downloading {len(ARXIV)} arXiv PDFs into {CORPUS}\n")
    ok = 0
    for aid, name in ARXIV.items():
        out = os.path.join(CORPUS, name)
        if os.path.exists(out) and os.path.getsize(out) > 10000:
            print(f"  skip (exists)  {name}")
            ok += 1
            continue
        try:
            size = fetch(f"https://arxiv.org/pdf/{aid}", out)
            print(f"  OK  {name}  ({size:,} bytes)")
            ok += 1
        except Exception as e:
            print(f"  FAIL {aid}: {e}")
    print(f"\n{ok}/{len(ARXIV)} available. Also copy real vector Pt-CV papers "
          f"(e.g. from data/in/) into data/corpus/, then run:\n"
          f"  cvdigitize batch data/corpus")


if __name__ == "__main__":
    main()
