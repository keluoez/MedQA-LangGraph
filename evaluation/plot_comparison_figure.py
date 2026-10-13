#!/usr/bin/env python3
"""
Top-conference-style comprehensive biomedical QA comparison figure.
6 datasets, 3 evaluators, sorted by Evidence Grounding (high->low).

Style: Colorblind-safe, minimal grid, subfigure labels, Nature/ACL aesthetic.
Output: comparison_data/fig_comprehensive_comparison.pdf (.png)
"""
import json, os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Styling ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# ── Colorblind-safe palette (Wong + custom) ───────────────────────────
# Ours gets a distinct highlight color
DATASET_COLORS = {
    'MedQA-LangGraph\n(Ours)':        '#0072B2',  # strong blue
    'ReasonMed':                       '#E69F00',  # orange
    'BioGraphletQA':                   '#009E73',  # bluish green
    'MedQuAD\n(NIH Expert)':           '#CC79A7',  # reddish purple
    'medical-o1\n(GPT-4o+Verifier)':   '#F0E442',  # yellow
    'ChatDoctor\n(Patient Dialogues)': '#D55E00',  # vermillion
}

EVAL_STYLES = {
    'DeepSeek-V3':    dict(alpha=1.0,  hatch='',   edgecolor='white', linewidth=0.6),
    'Qwen2.5-72B':    dict(alpha=0.72, hatch='..', edgecolor='white', linewidth=0.6),
    'GPT-4o':         dict(alpha=0.55, hatch='//', edgecolor='white', linewidth=0.6),
}

# ── Data loading ───────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
COMP_DIR = BASE / 'comparison_data'
BIO_DIR  = Path('c:/Users/Jiang/Desktop/run-bdi/experiments/biographletqa/output')

def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def compute_stats(results):
    n = len(results)
    return {
        'F':  sum(r['factuality']  for r in results) / n,
        'Gr': sum(r['granularity'] for r in results) / n,
        'Gnd': sum(r['grounding']   for r in results) / n,
        'Tot': sum(r['total']       for r in results) / n,
    }

# Dataset definitions — sorted by Grounding (high -> low)
DATASETS = [
    ('MedQA-LangGraph\n(Ours)', {
        'DeepSeek-V3':  BASE / 'our_data_100_evaluated_deepseek.json',
        'Qwen2.5-72B':  BASE / 'our_data_100_evaluated_qwen.json',
        'GPT-4o':       BASE / 'our_data_100_evaluated_deepseek.json',  # placeholder, use all_stats
    }),
    ('ReasonMed', {
        'DeepSeek-V3':  BASE / 'reasonmed_100_evaluated_deepseek.json',
        'Qwen2.5-72B':  BASE / 'reasonmed_100_evaluated_final.json',
        'GPT-4o':       COMP_DIR / 'reasonmed_evaluated_gpt4o.json',
    }),
    ('BioGraphletQA', {
        'DeepSeek-V3':  BIO_DIR / 'biographletqa_results_deepseek.json',
        'Qwen2.5-72B':  BIO_DIR / 'biographletqa_results_qwen.json',
        'GPT-4o':       BIO_DIR / 'biographletqa_results_gpt4o.json',
    }),
    ('MedQuAD\n(NIH Expert)', {
        'DeepSeek-V3':  COMP_DIR / 'medquad_evaluated_deepseek.json',
        'Qwen2.5-72B':  COMP_DIR / 'medquad_evaluated_qwen.json',
        'GPT-4o':       COMP_DIR / 'medquad_evaluated_gpt4o.json',
    }),
    ('medical-o1\n(GPT-4o+Verifier)', {
        'DeepSeek-V3':  COMP_DIR / 'medical_o1_evaluated_deepseek.json',
        'Qwen2.5-72B':  COMP_DIR / 'medical_o1_evaluated_qwen.json',
        'GPT-4o':       COMP_DIR / 'medical_o1_evaluated_gpt4o.json',
    }),
    ('ChatDoctor\n(Patient Dialogues)', {
        'DeepSeek-V3':  COMP_DIR / 'chatdoctor_evaluated_deepseek.json',
        'Qwen2.5-72B':  COMP_DIR / 'chatdoctor_evaluated_qwen.json',
        'GPT-4o':       COMP_DIR / 'chatdoctor_evaluated_gpt4o.json',
    }),
]

# ── Load from all_stats.json for Ours GPT-4o (no separate file) ──────
ALL_STATS = load_json(COMP_DIR / 'all_stats.json')

def get_ours_gpt4o():
    s = ALL_STATS['ours_gpt4o']
    return {'F': s['factuality'], 'Gr': s['granularity'], 'Gnd': s['grounding'], 'Tot': s['total']}


def main():
    # Load data
    data = {}
    labels = []
    for name, paths in DATASETS:
        labels.append(name)
        data[name] = {}
        for evaluator, path in paths.items():
            try:
                if evaluator == 'GPT-4o' and 'Ours' in name:
                    data[name][evaluator] = get_ours_gpt4o()
                elif path.exists():
                    results = load_json(path)
                    data[name][evaluator] = compute_stats(results)
                else:
                    print(f'  MISSING: {path}')
                    data[name][evaluator] = {'F': 0, 'Gr': 0, 'Gnd': 0, 'Tot': 0}
            except Exception as e:
                print(f'  ERROR loading {name}/{evaluator}: {e}')
                data[name][evaluator] = {'F': 0, 'Gr': 0, 'Gnd': 0, 'Tot': 0}

    # ── Build plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.8))
    # Reserve top space for suptitle + shared legend row
    fig.subplots_adjust(hspace=0.34, wspace=0.22, top=0.88, bottom=0.09, left=0.07, right=0.97)

    # Metrics in correct reading order: a(top-left), b(top-right), c(bot-left), d(bot-right)
    metrics = [
        ('Gnd', 'Evidence Grounding (0-3)',    axes[0, 0], 3.3,  '(a)'),
        ('F',   'Factuality (0-4)',            axes[0, 1], 4.5,  '(b)'),
        ('Gr',  'Mechanism Granularity (0-3)', axes[1, 0], 3.3,  '(c)'),
        ('Tot', 'Total Score (0-10)',          axes[1, 1], 10.5, '(d)'),
    ]

    evaluators = ['DeepSeek-V3', 'Qwen2.5-72B', 'GPT-4o']
    x = np.arange(len(labels))
    n_ev = len(evaluators)
    bar_w = 0.22
    offsets = [(i - (n_ev - 1) / 2) * bar_w for i in range(n_ev)]

    for mk, mlabel, ax, ymax, sub_lbl in metrics:
        means = []
        for name, _ in DATASETS:
            vals = [data[name][ev][mk] for ev in evaluators]
            means.append(np.mean(vals))

        for i, ev in enumerate(evaluators):
            vals = [data[name][ev][mk] for name, _ in DATASETS]
            bars = ax.bar(
                x + offsets[i], vals, bar_w,
                label=ev,
                color=[DATASET_COLORS[name] for name, _ in DATASETS],
                **EVAL_STYLES[ev],
            )
            for b, v in zip(bars, vals):
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2,
                            b.get_height() + ymax * 0.012,
                            f'{v:.2f}', ha='center', va='bottom',
                            fontsize=6.2, fontweight='bold', color='#333333')

        ax.scatter(x, means, marker='D', s=32, c='#333333', zorder=5,
                   edgecolors='white', linewidths=0.5, label='Mean')

        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, linespacing=1.1)
        ax.set_ylabel(mlabel, fontsize=10, fontweight='bold')
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(ymax / 6 if ymax > 5 else 0.5))
        ax.grid(axis='y', alpha=0.18, linestyle='-', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(axis='both', length=3, width=0.5)

        # Subfigure label at top-right to avoid blocking bar values
        ax.text(0.98, 0.96, sub_lbl, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none', alpha=0.85))

    # ── Shared legend at top-right of figure (above subplots) ─────────
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#888888', alpha=1.0,  edgecolor='white', linewidth=0.6,
              label='DeepSeek-V3'),
        Patch(facecolor='#888888', alpha=0.72, edgecolor='white', linewidth=0.6,
              hatch='..', label='Qwen2.5-72B'),
        Patch(facecolor='#888888', alpha=0.55, edgecolor='white', linewidth=0.6,
              hatch='//', label='GPT-4o'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#333333',
                   markersize=8, markeredgecolor='white', markeredgewidth=0.5,
                   label='Mean'),
    ]
    fig.legend(handles=legend_elements, loc='upper right',
              fontsize=8, framealpha=0.9, edgecolor='#cccccc',
              ncol=4, handlelength=1.8, columnspacing=0.8,
              bbox_to_anchor=(0.985, 0.965))

    # Suptitle (below legend row to avoid overlap)
    fig.suptitle('Comprehensive Biomedical QA Quality Comparison\n'
                 '(Sorted by Evidence Grounding)',
                 fontsize=14, fontweight='bold', y=1.005)

    # ── Save ──────────────────────────────────────────────────────────
    out_base = COMP_DIR / 'fig_comprehensive_comparison'
    for fmt, dpi in [('pdf', None), ('png', 300)]:
        fname = f'{out_base}.{fmt}'
        fig.savefig(fname, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        size_kb = os.path.getsize(fname) / 1024
        print(f'Saved: {fname} ({size_kb:.0f} KB)')

    plt.close()
    print('Done.')


if __name__ == '__main__':
    main()
