"""Post-process precision_raw.json → precision.tex.

Usage: python precision_table.py <bench_dir>
"""

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np


bench_dir = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("BENCH_DIR", ".")

# Parameter column order mirrors conftest.py rand_<source> fixtures
PARAM_ORDER: dict[str, list[str]] = {
    "tmsv":   ["mean_photon", "detection_efficiency"],
    "spdc":   ["mean_photon", "bsm_efficiency", "outcoupling_efficiency", "detection_efficiency"],
    "zalm":   ["mean_photon", "bsm_efficiency", "outcoupling_efficiency", "detection_efficiency", "dark_counts"],
    "sigsag": ["mean_photon", "bsm_efficiency", "outcoupling_efficiency", "detection_efficiency"],
}

SOURCE_ORDER = ["tmsv", "spdc", "zalm", "sigsag"]

SOURCE_NAMES = {
    "tmsv":   "TMSV",
    "spdc":   "SPDC",
    "zalm":   "ZALM",
    "sigsag": "Sagnac (SIGSAG)",
}

PARAM_DISPLAY: dict[str, str] = {
    "mean_photon":           r"$N_s$",
    "detection_efficiency":  r"$\eta_d$",
    "outcoupling_efficiency": r"$\eta_t$",
    "bsm_efficiency":        r"$\eta_b$",
    "dark_counts":           r"$P_d$",
}


def _decode(v: float | dict) -> float | np.ndarray:
    if isinstance(v, dict):
        return np.array(v["real"]) + 1j * np.array(v["imag"])
    return float(v)


def _errors(py_raw: float | dict, jl_raw: float | dict) -> tuple[float, float]:
    pv, jv = _decode(py_raw), _decode(jl_raw)
    if isinstance(pv, np.ndarray):
        diff = np.abs(jv - pv)  # type: ignore[operator]
        abs_err = float(np.max(diff))
        denom = float(np.max(np.abs(pv)))
    else:
        abs_err = abs(float(jv) - float(pv))  # type: ignore[arg-type]
        denom = abs(float(pv))
    rel_err = abs_err / denom if denom != 0.0 else math.inf
    return abs_err, rel_err


def _sci(v: float) -> str:
    if math.isinf(v):
        return r"$\infty$"
    if math.isnan(v):
        return r"$\mathrm{NaN}$"
    if v == 0.0:
        return r"$0$"
    exp = int(math.floor(math.log10(abs(v))))
    mant = v / 10.0 ** exp
    return rf"${mant:.3f} \times 10^{{{exp}}}$"


def _esc(s: str) -> str:
    return s.replace("_", r"\_")


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def _build_table(header: list[str], rows: list[list[str]], resize: bool) -> str:
    col_spec = "l|" + "r" * (len(header) - 3) + "|rr"
    begin = r"\begin{tabular}{" + col_spec + r"}"
    end = r"\end{tabular}"
    if resize:
        begin = r"\resizebox{\linewidth}{!}{" + begin
        end = end + "}"
    lines = [
        begin,
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", end]
    return "\n".join(lines)


# ---- Load and group raw data ----

with open(os.path.join(bench_dir, "precision_raw.json")) as f:
    raw_rows: list[dict] = json.load(f)

by_source: dict[str, list[dict]] = defaultdict(list)
for row in raw_rows:
    src = row["function"].split(".")[0]
    by_source[src].append(row)

# ---- Build LaTeX table ----

parts = [
    r"% Precision table",
    r"\begin{table*}[htbp]",
    r"\caption{Absolute and relative error between Python and Julia Genqo implementations across source types and parameter configurations.}",
    r"\label{tab:precision}",
    r"\centering",
]

for src in SOURCE_ORDER:
    if src not in by_source:
        continue
    rows = by_source[src]
    pkeys = PARAM_ORDER[src]
    header = ["Toolbox Function"] + [PARAM_DISPLAY.get(k, _esc(k)) for k in pkeys] + ["Absolute Error", "Relative Error"]
    resize = False

    table_rows = []
    for row in rows:
        fn_cell = r"\texttt{" + _esc(row["function"]) + "}"
        if isinstance(row["py"], dict):
            fn_cell += r"$^\dagger$"
        ae, re = _errors(row["py"], row["jl"])
        table_rows.append(
            [fn_cell]
            + [_fmt(row["params"][k]) for k in pkeys]
            + [_sci(ae), _sci(re)]
        )

    parts.append("")
    parts.append(r"\medskip\noindent\textbf{" + SOURCE_NAMES[src] + r"}\par\smallskip")
    parts.append(_build_table(header, table_rows, resize))

parts.append("")
parts.append(r"\end{table*}")
parts.append("")

tex_path = os.path.join(bench_dir, "precision.tex")
with open(tex_path, "w") as f:
    f.write("\n".join(parts))

print(f"Wrote {tex_path}")
