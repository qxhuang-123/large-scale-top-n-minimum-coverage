# Latest MCF N=10 rerun

The explicit-network MCF was rerun with the shared score/candidate inputs and
`D = ceil(q * |I^alpha|)` for `q in {0,20,40,60,80,100}`.

| Dataset | D=0 K | D=0 objective | Times at 0/20/40/60/80/100% (s) |
| --- | ---: | ---: | --- |
| OP | 780 | 242825.25429558754 | 0.2, 4.1, 8.0, 11.9, 15.8, 19.9 |
| Yelp | 352 | 453219.4965727329 | 1.9, 27.9, 53.4, 79.2, 104.3, infeasible |
| VG | 1,251 | 1184136.1799118519 | 5.5, 86.2, 167.8, 250.2, 336.0, 424.2 |
| TG | 2,426 | 849787.793035984 | 5.2, 146.5, 211.6, 276.9, 345.0, 412.7 |

The separate SO invocation used a transposed matrix (18,357 x 35,598 rather
than the shared 35,598 x 18,357 user-item matrix) and then exhausted memory
while building the explicit network at 20%. It is excluded from the reported
results; Table 4 retains `Mem. infeas.` for SO MCF.
