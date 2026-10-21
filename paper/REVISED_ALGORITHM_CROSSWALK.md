# Revised implementation crosswalk (22 October 2026)

| Paper name | Current code | Internal mode |
|---|---|---|
| TPCA | current/TPCA/native | state_source |
| DGCA | current/DGCA/native | dynamic_user_heap |
| R | current/DGCA | random_source |
| A | current/DGCA | original |
| B | current/DGCA | fixed_source |
| C | current/DGCA | dynamic_item_heap |
| D | current/DGCA | dynamic_user_heap |

Original COR code remains in methods/ and src/ as a historical baseline.
TPCA leave-one-module-out and V1--V4 all retain local shortest-path repair.
V0 was not rerun under the revised shared-input protocol. R changes only the
initially uncovered-item visit order; C/D implement the same dynamic rule.
An exact same-coverage trajectory is needed to interpret DGCA score gaps.
