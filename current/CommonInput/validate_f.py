"""Independent LP checks of the delivered F API, including infeasible targets."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from f_algorithm import CandidateGraph, solve_graph


def lp(graph, N, D):
    U,I=graph.n_users,graph.n_items
    _,items,scores,users,*_=graph.arrays; E=len(items)
    eq=np.zeros((U,E+I)); eq[users,np.arange(E)]=1
    ub=np.zeros((I+1,E+I)); ub[items,np.arange(E)]=-1
    ub[np.arange(I),E+np.arange(I)]=1; ub[I,E:]=-1
    rhs=np.zeros(I+1); rhs[-1]=-D
    return linprog(np.r_[-scores,np.zeros(I)],A_eq=eq,b_eq=np.full(U,N),
                   A_ub=ub,b_ub=rhs,bounds=(0,1),method="highs")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--instances",type=int,default=40)
    p.add_argument("--output",type=Path,default=Path("validation/current_validation.json"))
    args=p.parse_args()
    stats=dict(instances=args.instances,seed_base=20261017,targets_checked=0,
               lp_feasible=0,lp_infeasible=0,maximum_coverage_certificates=0,
               max_lp_abs_error=0.,source_dirty_max=0,source_class_changes=0,
               source_preserved_tree_arcs=0,terminal_refreshes=0,heap_refreshes=0)
    for seed in range(args.instances):
        rng=np.random.default_rng(20261017+seed)
        U=int(rng.integers(3,10)); I=int(rng.integers(4,16)); N=int(rng.integers(1,min(4,I)))
        R=(rng.integers(-2,6,size=(U,I)) if seed%2==0 else rng.uniform(-2,5,(U,I))).astype(float)
        eligible=rng.random((U,I))<rng.uniform(.25,.95)
        if seed%4==0:
            N=1; R[:,0]=10; eligible[:,0]=True
        if seed%4==1:
            eligible[:,-1]=False
        for u in range(U):
            if eligible[u].sum()<N:
                eligible[u,rng.choice(I,N,replace=False)]=True
        g=CandidateGraph.from_matrix(R,eligible)
        full=solve_graph(g,N,I); K=full["coverage"]
        levels=sorted(set([0,I,*range(full["initial_coverage"],I+1)]))
        best_at_max=lp(g,N,K); assert best_at_max.success
        for D in levels:
            r=solve_graph(g,N,D); oracle=lp(g,N,D)
            stats["targets_checked"]+=1
            if oracle.success:
                assert r["target_feasible"] and r["coverage"]>=D
                error=abs(r["score"]+oracle.fun); stats["lp_feasible"]+=1
            else:
                assert oracle.status==2 and not r["target_feasible"]
                assert r["coverage"]==K and r["maximum_coverage_certified"]
                error=abs(r["score"]+best_at_max.fun)
                stats["lp_infeasible"]+=1; stats["maximum_coverage_certificates"]+=1
            assert error<1e-7,(seed,D,error)
            stats["max_lp_abs_error"]=max(stats["max_lp_abs_error"],error)
            assert r.get("source_pointer_advances",0)<=U*N
            assert r.get("terminal_pointer_advances",0)<=len(g.arrays[1])
            assert r.get("source_dirty_max",0)<=2
            assert r.get("full_searches",0)<=1
        for key in ("source_class_changes","source_preserved_tree_arcs","terminal_refreshes","heap_refreshes"):
            stats[key]+=full.get(key,0)
        stats["source_dirty_max"]=max(stats["source_dirty_max"],full.get("source_dirty_max",0))
    # The documented alternating-chain example must actually use two exchanges.
    g=CandidateGraph.from_matrix(np.array([[5.,4.9,0.],[5.2,0.,0.],[0.,4.8,4.7]]))
    r=solve_graph(g,1,3)
    assert r["coverage"]==3 and abs(r["score"]-14.8)<1e-10
    assert r["augmentations"]==1 and r["swaps"]==2
    g.save(args.output.parent/"demo_csr.npz") if args.output.parent.exists() else None
    stats["two_user_chain_checked"]=True
    # Reject invalid native inputs before reaching the C++ pointer interface.
    rejected=0
    for fn in [lambda:CandidateGraph.from_edges(1,2,[0,0],[0,0],[1.,2.]),
               lambda:solve_graph(g,4,3),lambda:solve_graph(g,1,4),
               lambda:CandidateGraph.from_edges(1,2,[0],[1],[float("nan")])]:
        try: fn()
        except ValueError: rejected+=1
    assert rejected==4; stats["invalid_inputs_rejected"]=rejected
    stats["passed"]=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(stats,indent=2),encoding="utf-8")
    print(json.dumps(stats,indent=2))


if __name__=="__main__": main()
