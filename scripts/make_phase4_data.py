"""Generate the Phase 4 surrogate dataset (labels from the Phase 1 analytic pricer).

Writes data/surrogate/heston_iv_dataset.npz (not committed: ~60 MB, regenerate with this script,
about 5 minutes on 4 cores). Deterministic given the seed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heston_risk.surrogate.dataset import make_dataset  # noqa: E402

N_SLICES = 2**17       # 131,072 (parameter, maturity) slices
STRIKES = 10           # strikes per slice -> 1,310,720 candidate rows
SEED = 2026

if __name__ == "__main__":
    t0 = time.perf_counter()
    d = make_dataset(N_SLICES, STRIKES, SEED)
    secs = time.perf_counter() - t0
    out = ROOT / "data" / "surrogate"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "heston_iv_dataset.npz", X=d["X"], y=d["y"], slice_id=d["slice_id"],
                        n_dropped=d["n_dropped"], n_total=d["n_total"], seconds=secs)
    print(f"{d['n_total']:,} rows generated, {d['n_dropped']:,} dropped (not invertible), {secs:.0f} s")
