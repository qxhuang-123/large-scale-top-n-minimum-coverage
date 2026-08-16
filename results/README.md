# Results policy

The previous archived JSON/XLSX/CSV files were removed because their dataset
shapes and initial coverages do not match Tables 2--5 of the submitted COR
manuscript. Keeping them would make the repository appear reproducible while
silently using a different preprocessing snapshot.

Generate replacement results only after confirming the manuscript dataset
shapes: OP 4,905 x 2,420; Yelp 9,464 x 11,197; VG 24,303 x 10,672;
TG 19,412 x 11,924; SO 35,598 x 18,357. All five methods must use the same
candidate cache, alpha=0.40, N=10, candidate universe I^alpha, and
D_target=ceil(|I^alpha|*p/100). Results should include the metadata fields
`dataset_shape`, `items_with_candidates`, `candidate_fraction`,
`D_target_base`, and `D_target_rounding`.