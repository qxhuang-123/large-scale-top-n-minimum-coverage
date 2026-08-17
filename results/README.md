# Reproducibility results

The JSON files in this directory are the result artifacts used to reproduce
the reported BGCR and TPCAR summaries. All current BGCR runs use the same
candidate cache, candidate fraction `alpha=0.40`, seed `20260704`, and the
deterministic tie rule: score descending, then item id ascending.

## BGCR

`bgcr/OP/`, `bgcr/Yelp/`, `bgcr/VG/`, and `bgcr/TG/` each contain one complete
JSON artifact for every `N` in `{10,15,20,25,30}`. `bgcr/SO/SO_N10_to_N30.json`
contains the corresponding five `N` values in one file. These artifacts report
the initial Top-N solution and the BGCR result at each nominal coverage target.

## TPCAR at N=10

`tpcar/n10/shared_initial/` contains the authoritative shared Top-N initial
solution for every dataset. These files are the exact initialization records
that match BGCR: they use the same score matrix, candidate cache, and tie rule.

`tpcar/n10/archived_history/` retains the previous full TPCAR histories used
for post-initial augmentation results. They are deliberately labelled
`previous`: the historical runners predate the shared-initialization record on
some datasets. Use the shared-initial files for the coverage-relaxed baseline
and consult the archived histories only for the retained augmentation traces.
