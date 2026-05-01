"""Create box and whisker plots comparing benchmark results between Python and Julia genqo implementations."""

import json
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

bench_dir = sys.argv[1] if len(sys.argv) > 1 else ".benchmarks"

first_run_dots = True # Flag indicating whether to plot blue/orange dots for Python/Julia first-run times

plt.rcParams.update({"font.size": 6})

# Load benchmark results
with open(f"{bench_dir}/py-bench.json") as f:
    py = json.load(f)
with open(f"{bench_dir}/jl-bench.json") as f:
    jl = json.load(f)

py_times = {bm["name"].removeprefix("test_").replace("__", "."): np.array(bm["stats"]["data"]) for bm in py["benchmarks"]}
jl_times = {name: np.array(bm[1]["times"])/1e9 for (name, bm) in jl[1][0][1]["data"].items()}

# Pull out the perfect-matching-permutation benchmarks for a separate plot below.
pmp_pattern = re.compile(r"^pmp\.(old|new)\.N=(\d+)$")
pmp_times = {name: jl_times.pop(name) for name in list(jl_times) if pmp_pattern.match(name)}

# Get all functions and sort by decreasing median time
all_funcs = list(set(py_times.keys()) | set(jl_times.keys()))
all_funcs.sort(key=lambda f: max(np.median(py_times.get(f, 0)), np.median(jl_times.get(f, 0))), reverse=True)

# Create paired boxplots
fig, ax = plt.subplots(figsize=(8.5-.7*2, 3.5))

positions = []
data = []
colors = []
labels = []
for i, func in enumerate(all_funcs):
    if func in py_times:
        positions.append(3*i + 1)
        data.append(py_times[func])
        colors.append('lightblue')
    
    if func in jl_times:
        positions.append(3*i + 2)
        data.append(jl_times[func])
        colors.append('lightcoral')

bp = ax.boxplot(
    data, 
    positions=positions, 
    medianprops=dict(color="black", lw=0.5), 
    boxprops=dict(lw=0.5), 
    whiskerprops=dict(lw=0.5), 
    capprops=dict(lw=0.5), 
    widths=1.0, 
    patch_artist=True, 
    # whis=(0,100),
    showfliers=False
)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add speedup annotations
i = 0
for j, func in enumerate(all_funcs):
    if func in py_times and func in jl_times:
        py_median = np.median(py_times[func])
        jl_median = np.median(jl_times[func])
        speedup = py_median / jl_median
        
        # Position text at the center between the two boxplots
        x_pos = 3*j + 1.5
        
        # Position above the higher boxplot (and first-run dots if enabled)
        py_max = bp['whiskers'][2*i+1].get_data()[1][-1]
        jl_max = bp['whiskers'][2*i+3].get_data()[1][-1]
        top = max(py_max, jl_max)
        if first_run_dots:
            top = max(top, py_times[func][0], jl_times[func][0])
        y_pos = top * 1.8
        
        ax.text(
            x_pos, y_pos, f'{speedup:.1f}×', 
            ha='center', va='bottom', fontsize=5, 
            bbox=dict(
                boxstyle='round,pad=0.3', facecolor='white', 
                edgecolor='gray', alpha=0.8, linewidth=0.5
            )
        )
        i += 2
    elif func in py_times or func in jl_times:
        i += 1

# Add orange dots for Julia first-run times
if first_run_dots:
    py_dot_plotted = False
    jl_dot_plotted = False
    for j, func in enumerate(all_funcs):
        if func in py_times:
            first_run = py_times[func][0]
            ax.plot(3*j + 1, first_run, 'o', color='lightblue', markeredgecolor='black', markeredgewidth=0.3, markersize=3, zorder=5, label='Python first run' if not py_dot_plotted else "")
            py_dot_plotted = True
        if func in jl_times:
            first_run = jl_times[func][0]
            ax.plot(3*j + 2, first_run, 'o', color='orange', markeredgecolor='black', markeredgewidth=0.3, markersize=3, zorder=5, label='Julia first run' if not jl_dot_plotted else "")
            jl_dot_plotted = True

ax.set_xticks([3*i + 1.5 for i in range(len(all_funcs))])
ax.set_xticklabels(all_funcs, rotation=45, ha='right')
ax.set_xlabel("Toolbox Function", fontsize=8)
ax.set_yscale('log')
ax.set_ylabel("Execution Time (s)", fontsize=8)
ax.set_title("Genqo Benchmark Comparison", fontsize=8)
ax.grid(axis='y', alpha=0.3)

# Add legend
legend_elements = [
    Patch(facecolor='lightblue', label='Genqo v0.1.0 (Python)'),
    Patch(facecolor='lightcoral', label='Genqo v1.2.0 (Julia)')
]
if first_run_dots:
    legend_elements += [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markeredgecolor='black', markeredgewidth=0.3, markersize=4, label='Genqo v0.1.0 first run'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markeredgecolor='black', markeredgewidth=0.3, markersize=4, label='Genqo v1.2.0 first run'),
    ]
ax.legend(handles=legend_elements, loc='upper right')

fig.tight_layout()
fig.savefig(f"{bench_dir}/benchmark_comparison.svg")


# Perfect Matching Permutation Algorithm benchmark comparison (old vs new at varying N)
pmp_groups = {}
for name, times in pmp_times.items():
    m = pmp_pattern.match(name)
    variant, N = m.group(1), int(m.group(2))
    pmp_groups.setdefault(N, {})[variant] = times

if pmp_groups:
    Ns = sorted(pmp_groups.keys())

    fig, ax = plt.subplots(figsize=(8.5/2-.7, 3.5))

    pmp_positions = []
    pmp_data = []
    pmp_colors = []
    for i, N in enumerate(Ns):
        if "old" in pmp_groups[N]:
            pmp_positions.append(3*i + 1)
            pmp_data.append(pmp_groups[N]["old"])
            pmp_colors.append('lightgreen')
        if "new" in pmp_groups[N]:
            pmp_positions.append(3*i + 2)
            pmp_data.append(pmp_groups[N]["new"])
            pmp_colors.append('lightcoral')

    bp = ax.boxplot(
        pmp_data,
        positions=pmp_positions,
        medianprops=dict(color="black", lw=0.5),
        boxprops=dict(lw=0.5),
        whiskerprops=dict(lw=0.5),
        capprops=dict(lw=0.5),
        widths=1.0,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp['boxes'], pmp_colors):
        patch.set_facecolor(color)

    ax.set_xticks([3*i + 1.5 for i in range(len(Ns))])
    ax.set_xticklabels([str(N) for N in Ns])
    ax.set_xlabel("N (number of modes to partition)", fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel("Execution Time (s)", fontsize=8)
    ax.set_title("Perfect Pairing Algorithm Benchmark Comparison", fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    ax.legend(handles=[
        Patch(facecolor='lightgreen', label='Old algorithm'),
        Patch(facecolor='lightcoral', label='New algorithm'),
    ], loc='upper left')

    fig.tight_layout()
    fig.savefig(f"{bench_dir}/pmp_benchmark_comparison.svg")
