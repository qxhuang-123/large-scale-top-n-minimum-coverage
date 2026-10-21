"""Derive a seven-array binding; keep the original F project byte-unchanged.

Reverse arc weights are obtained from user_scores[item_edges[p]]. The F
state_source branch already uses this lookup; its search/repair logic is intact.
"""
import hashlib
from pathlib import Path
import shutil
from bench_common import ROOT, F_ROOT, write_json


def main():
    target = ROOT / 'native'; target.mkdir(exist_ok=True)
    originals = {}
    for name in ('state_variants.cpp', 'state_repair_impl.inc', 'local_repair_impl.inc', 'msvc_compat.h'):
        path = F_ROOT / 'src' / name
        originals[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        text = path.read_text(encoding='utf-8')
        text = text.replace('st.item_scores[p]', 'st.edge_score(st.item_edges[p])')
        if name == 'state_variants.cpp':
            replacements = {
                '    const double* item_scores = nullptr;\n': '',
                '    py::array_t<double, py::array::c_style | py::array::forcecast> item_scores,\n': '',
                '    auto is = item_scores.request();\n': '',
                ' || is.ndim != 1': '',
                ' || iu.shape[0] != is.shape[0]': '',
                '    st.item_scores = static_cast<const double*>(is.ptr);\n': '',
                'PYBIND11_MODULE(state_variants,m)': 'PYBIND11_MODULE(state_variants_bench,m)',
                '"item_edges"_a,"item_scores"_a,"N"_a': '"item_edges"_a,"N"_a',
            }
            for before, after in replacements.items():
                if text.count(before) != 1:
                    raise ValueError('Source changed; review compact binding derivation: %r' % before)
                text = text.replace(before, after)
        if 'st.item_scores' in text:
            raise ValueError('Unconverted reverse-score use')
        (target / name).write_text(text, encoding='utf-8')
    write_json(ROOT / 'compact_backend_provenance.json', dict(
        original_project_unchanged=True, original_source_sha256=originals,
        changes=['Separate module name', 'Remove redundant item_scores input/storage',
                 'Read reverse weights through original edge IDs'],
        state_source_algorithm='same original coverage compression, heaps, local repair and augmentation',
        derived_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir() if p.is_file()}))
    print('Derived seven-array backend sources; original F sources unchanged.')


if __name__ == '__main__': main()
