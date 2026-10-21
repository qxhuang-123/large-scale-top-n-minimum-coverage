"""Portable, saved-data-only figures and booktabs tables for TPCA / DGCA.

No solver imports or experimental execution. Inputs live in ../data.
Python 3.9+; see requirements.txt for the versions used for delivery.
"""


from pathlib import Path


from datetime import datetime, timezone


import argparse


import csv


import hashlib


import json


import platform


import textwrap


import warnings


import numpy as np


import pandas as pd


import matplotlib


matplotlib.use('Agg')


import matplotlib.pyplot as plt


from matplotlib.backends.backend_pdf import PdfPages


from matplotlib.lines import Line2D


from matplotlib.ticker import ScalarFormatter, LogLocator, NullLocator


from matplotlib.transforms import Bbox


from PIL import Image, ImageOps, ImageDraw


ROOT = Path(__file__).resolve().parents[1]


DS = ['OP', 'YELP', 'VG', 'TG', 'SO']


METHODS = ['MCF', 'TPCAR', 'BGCR-D']


LABEL = {'YELP': 'Yelp', 'TPCAR': 'TPCA', 'BGCR-D': 'DGCA'}


STYLE = {
    'MCF': ('#D55E00', 's', '--'),
    'TPCAR': ('#0072B2', 'o', '-'),
    'BGCR-D': ('#009E73', '^', '-.'),
}


DCOLOR = ['#0072B2', '#009E73', '#D55E00', '#B58A00', '#555555']


DMARK = ['o', '^', 's', 'P', 'D']


DLINE = ['-', '--', '-.', ':', '-']


WIDTH = (210 - 2 * 18.1) / 25.4


STAMP = datetime(2026, 10, 21, tzinfo=timezone.utc)


PDFMETA = {'Creator': 'TPCA-DGCA reproducible plotting package',
           'CreationDate': STAMP, 'ModDate': STAMP}


QA = {}


ASSETS = []


BOOK = None


def label(x):
    return LABEL.get(x, x)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=lambda x: x.item() if hasattr(x, 'item') else str(x)),
                          encoding='utf8')


def style():
    # STIXGeneral is distributed with Matplotlib: no system font is required.
    plt.rcParams.update({
        'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 9,
        'axes.linewidth': .7, 'axes.spines.top': False, 'axes.spines.right': False,
        'lines.linewidth': 1.2, 'lines.markersize': 4,
        'axes.grid': True, 'axes.axisbelow': True,
        'grid.color': '#DDDDDD', 'grid.linewidth': .45,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path',
        'svg.hashsalt': 'TPCA_DGCA_PREVIEW_20261021',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def read_data():
    expected = json.loads((ROOT / 'data/SOURCE_SHA256.json').read_text(encoding='utf8'))
    for relative, digest in expected.items():
        actual = hashlib.sha256((ROOT / 'data' / relative).read_bytes()).hexdigest()
        assert actual == digest, f'Input changed: {relative}'
    a = pd.read_csv(ROOT / 'data/all_runs.csv')
    m = pd.read_csv(ROOT / 'data/main_comparison.csv')
    n = pd.read_csv(ROOT / 'data/N_comparison.csv')
    s = pd.read_csv(ROOT / 'data/strategy_runs.csv')
    assert len(a) == 479 and len(m) == 294 and len(n) == 25 and len(s) == 76
    assert set(s.dataset) == set(DS)
    np.testing.assert_allclose(n.Exact_loss_pct, 100*(n.Top_APS-n.Exact_APS)/n.Top_APS,
                               atol=1e-10, rtol=0)
    shared = m[m.method.isin(METHODS)]
    for (ds, rho), g in shared.groupby(['dataset', 'percentage']):
        for key in ['graph_id', 'N', 'D', 'K0', 'initial_q']:
            assert g[key].nunique() == 1, (ds, rho, key)
    assert set(shared.status) == {'OPTIMAL', 'INFEASIBLE', 'FEASIBLE',
                                  'TARGET_UNREACHED', 'SYSTEM_MEMORY_GUARD'}
    assert not shared.duplicated(['dataset', 'method', 'percentage', 'repeat']).any()
    good = shared[shared.status != 'SYSTEM_MEMORY_GUARD']
    assert (good.groupby(['dataset', 'method', 'percentage']).size() == 3).all()
    assert (good.total > 0).all() and (good.peak_gib > 0).all()
    # Match every main-comparison DGCA objective with the saved exact trajectory at its K.
    references = {}
    quality = []
    for ds in DS:
        r = n[(n.dataset == ds) & (n.N == 10)].iloc[0]
        q0 = int(shared[shared.dataset == ds].initial_q.iloc[0])
        for method, filename in [('TPCAR', f'main_{ds}_TPCAR.json'),
                                 ('BGCR-D', f'BGCR_{ds}_dynamic_user_heap.json')]:
            seq = {}
            for row in json.loads((ROOT / 'data/trajectories' / filename).read_text(encoding='utf8')):
                k = row.get('coverage', row.get('diversity'))
                q = row.get('integer_objective', row.get('objective'))
                if k is None or q is None:
                    continue
                k, q = int(k), int(q)
                assert 0 <= q < 2**53
                assert k not in seq or seq[k] == q, (ds, method, k, 'conflicting duplicate')
                seq[k] = q
            assert min(seq) == r.K0 and seq[int(r.K0)] == q0
            assert max(seq) <= r.Kmax and len(seq) == max(seq) - min(seq) + 1
            assert all(seq[k] >= seq[k+1] for k in range(min(seq), max(seq)))
            if method == 'TPCAR':
                assert max(seq) == r.Kmax
                references[ds] = seq
            last = shared[(shared.dataset == ds) & (shared.method == method) &
                          (shared.percentage == 100)]
            assert (last.K == max(seq)).all() and (last.q == seq[max(seq)]).all()
            for k, q in sorted(seq.items()):
                quality.append({'dataset': ds, 'method': label(method), 'K': k, 'q': q,
                                'K0': int(r.K0), 'Kmax': int(r.Kmax), 'initial_q': q0,
                                'coverage_increase_pct': 100 * (k-r.K0)/(r.Kmax-r.K0),
                                'score_loss_pct': 100 * (q0-q)/q0})
    for _, row in shared[shared.method == 'BGCR-D'].iterrows():
        eq = references[row.dataset][int(row.K)]
        assert eq == int(row.exact_q_same_K)
        assert int(row.q) <= eq
        np.testing.assert_allclose(row.gap_same_K_pct, 100*(eq-row.q)/eq, atol=1e-11)
    # The independent gap runs have one observation per target, not three repeats.
    p = a[a.batch == 'progressive'].copy()
    assert len(p) == 32 and (p.status == 'OPTIMAL').all()
    for (ds, eta), g in p.groupby(['dataset', 'gap_percentage']):
        for key in ['D', 'graph_id', 'K0', 'initial_q', 'K', 'q']:
            assert g[key].nunique() == 1, (ds, eta, key)
        r = n[(n.dataset == ds) & (n.N == 10)].iloc[0]
        assert int(g.D.iloc[0]) == int(r.K0 + np.ceil(eta/100*(r.Kmax-r.K0)))
    # Strategy controls use the same initial state and candidate graph.
    for ds, g in s.groupby('dataset'):
        for key in ['graph_id', 'N', 'D', 'initial_q', 'initial_coverage']:
            assert g[key].nunique() == 1, (ds, key)
        count_r = 1 if ds in ['VG', 'TG'] else 10
        count_d = 1 if ds in ['VG', 'TG'] else 3
        assert len(g[g.variant == 'R']) == count_r
        for v in ['A', 'B', 'C', 'D']:
            h = g[g.variant == v]
            assert len(h) == count_d and h.q.nunique() == h.K.nunique() == 1
        for v, w in [('A', 'B'), ('C', 'D')]:
            h, j = g[g.variant == v].iloc[0], g[g.variant == w].iloc[0]
            assert h.q == j.q and h.K == j.K
    qdf = pd.DataFrame(quality)
    qdf.to_csv(ROOT / 'data/quality_plot_points.csv', index=False, encoding='utf-8-sig')
    dump(ROOT / 'qa/data_audit.json', {
        'frozen_source_files_verified': len(expected), 'source_rows': 479+len(s),
        'main_rows_used': len(shared), 'main_completed_or_diagnosed_rows': len(good),
        'main_guard_stops': len(shared)-len(good), 'N_configurations': len(n),
        'progressive_independent_runs': len(p), 'strategy_runs': len(s),
        'quality_points': len(qdf), 'shared_initial_state_checks': 'passed',
        'duplicate_trajectory_objectives': 'consistent', 'integer_reference_checks': 'passed',
        'solver_execution': False,
    })
    return a, m, n, s, qdf


def axis_style(ax):
    ax.xaxis.grid(False)
    ax.tick_params(direction='out', length=2.5, width=.65)


def five_panels():
    fig = plt.figure(figsize=(WIDTH, 4.55))
    positions = [(.080, .590), (.405, .590), (.730, .590),
                 (.2425, .145), (.5675, .145)]
    axs = []
    for i, (ds, (left, bottom)) in enumerate(zip(DS, positions)):
        ax = fig.add_axes([left, bottom, .235, .275])
        ax.set_title(f'({chr(97+i)}) {label(ds)}', loc='left', pad=7)
        axis_style(ax)
        axs.append(ax)
    return fig, axs


def method_handles():
    return [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], ls=STYLE[m][2],
                   markerfacecolor='white', label=label(m)) for m in METHODS]


def check_layout(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    excluded = set()
    overlaps = []
    for ax in fig.axes:
        if not ax.axison:
            for axis in [ax.xaxis, ax.yaxis]:
                for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                    excluded.update([id(tick.label1), id(tick.label2)])
            continue
        for axis, lim in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            for tick in list(axis.get_major_ticks()) + list(axis.get_minor_ticks()):
                if not min(lim) <= tick.get_loc() <= max(lim):
                    excluded.update([id(tick.label1), id(tick.label2)])
        for ticks in [ax.get_xticklabels(), ax.get_yticklabels()]:
            boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in ticks
                     if t.get_visible() and t.get_text() and id(t) not in excluded]
            for (a, ba), (b, bb) in zip(boxes, boxes[1:]):
                if ba.overlaps(bb):
                    overlaps.append([a, b])
    outside = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text() or id(t) in excluded:
            continue
        box = t.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or
                                        box.x1 > width+1 or box.y1 > height+1):
            outside.append(t.get_text())
    assert not outside, ('outside canvas', outside)
    assert not overlaps, ('overlapping tick labels', overlaps)
    return {'outside_canvas': outside, 'tick_label_overlaps': overlaps}


def finish(fig, name, title, caption, kind='figures'):
    global BOOK
    target = ROOT / kind / name
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = check_layout(fig)
        fig.savefig(str(target)+'.png', dpi=600)
        fig.savefig(str(target)+'.pdf', metadata=PDFMETA)
        fig.savefig(str(target)+'.svg', metadata={'Date': '2026-10-21'})
    glyph = [str(w.message) for w in caught if 'Glyph' in str(w.message)]
    assert not glyph, glyph
    result.update({'missing_glyphs': glyph, 'size_inches': list(fig.get_size_inches()),
                   'png_dpi': 600, 'font': 'STIXGeneral (bundled with Matplotlib)'})
    QA[name] = result
    with Image.open(str(target)+'.png') as im:
        im.convert('L').save(ROOT / 'qa' / (name+'_gray.png'), dpi=(600, 600))
    ASSETS.append({'name': name, 'kind': kind, 'title': title, 'caption': caption})
    if BOOK is not None:
        # Extend the PDF page below the figure. The reusable figure itself has no caption.
        wrapped = textwrap.fill(caption, width=115)
        lines = len(wrapped.splitlines())
        extra = .38 + .14*lines
        h = fig.get_figheight()
        title_artist = fig.text(.045, -.16/h, title, ha='left', va='top',
                                fontsize=10, fontweight='bold')
        cap_artist = fig.text(.045, -.37/h, wrapped, ha='left', va='top',
                              fontsize=8, linespacing=1.25)
        BOOK.savefig(fig, bbox_inches=Bbox.from_extents(0, -extra, WIDTH, h))
        title_artist.remove()
        cap_artist.remove()
    plt.close(fig)


def quality_figure(q):
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.985, bottom=.22, top=.78, wspace=.25)
    handles = []
    for a, (method, ax) in enumerate(zip(['TPCA', 'DGCA'], axs)):
        for i, ds in enumerate(DS):
            g = q[(q.dataset == ds) & (q.method == method)].sort_values('K')
            inds = np.unique(np.linspace(0, len(g)-1, 8, dtype=int)).tolist()
            line, = ax.plot(g.coverage_increase_pct, g.score_loss_pct, color=DCOLOR[i],
                            ls=DLINE[i], marker=DMARK[i], ms=4, markerfacecolor='white',
                            markeredgewidth=.85, markevery=inds, lw=1.25, label=label(ds))
            if a == 0:
                handles.append(line)
        ax.set_title(f'({chr(97+a)}) {method}', loc='left', pad=8)
        ax.set_xlim(0, 101)
        ax.set_ylim(0, .55)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks([0, .1, .2, .3, .4, .5])
        ax.set_xlabel('Attained coverage increase (%)')
        axis_style(ax)
    axs[0].set_ylabel('Predicted-score loss (%)')
    fig.legend(handles=handles, ncol=5, loc='upper center', bbox_to_anchor=(.52, .995),
               frameon=False, handlelength=2.3, columnspacing=1.6)
    finish(fig, 'F04_coverage_quality', 'Figure 4. Coverage and predicted-score trade-off',
           'Predicted-score loss along the largest-target runs (N = 10). Coverage increase is '
           '100(K - K0)/(Kmax - K0); score loss is relative to the initial Top-N plan using the rounded scores. '
           'Both axes are percentages and both panels share their scales. DGCA curves end at their attained coverage.')


def progressive_figure(a):
    p = a[a.batch == 'progressive']
    modes = ['state_progress_v1', 'state_progress_v2', 'state_source_only', 'state_source']
    colors = ['#555555', '#D55E00', '#0072B2', '#009E73']
    fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.05))
    fig.subplots_adjust(left=.10, right=.985, bottom=.23, top=.78, wspace=.29)
    handles = []
    for i, (ds, ax) in enumerate(zip(['OP', 'YELP'], axs)):
        for j, mode in enumerate(modes):
            g = p[(p.dataset == ds) & (p['mode'] == mode)].sort_values('gap_percentage')
            assert len(g) == 4
            line, = ax.plot(g.gap_percentage, g.core, color=colors[j],
                            marker=['s', 'D', 'o', '^'][j], ls=[':', '--', '-.', '-'][j],
                            markerfacecolor='white', markeredgewidth=.9,
                            label=f'V{j+1}' + (' (TPCA)' if j == 3 else ''))
            if i == 0:
                handles.append(line)
        ax.set_title(f'({chr(97+i)}) {label(ds)}', loc='left', pad=8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(.8, 130)
        ax.set_xticks([1, 10, 50, 100])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r'Available-gap fraction $\eta$ (%)')
        ax.set_ylabel('Core time (s)')
        axis_style(ax)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .995), ncol=4,
               frameon=False, handlelength=2.3, columnspacing=1.8)
    p[['dataset', 'mode', 'gap_percentage', 'D', 'K0', 'K', 'q', 'core', 'build', 'total']].to_csv(
        ROOT/'data/progressive_plot_points.csv', index=False, encoding='utf-8-sig')
    finish(fig, 'F05_progressive_ablation', 'Figure 5. Progressive TPCA ablation',
           'Core time for independently started targets at 1%, 10%, 50% and 100% of the available coverage gap. '
           'Each point is one run. V1 retains local repair with boundary scans; V2 adds persistent boundary heaps; '
           'V3 adds donor aggregation; V4 adds terminal aggregation and is the full TPCA. '
           'All versions return the same exact solution value at each target. Core time includes native initialization.')


TASKS = ['runtime', 'memory', 'quality', 'ablation', 'tables']


