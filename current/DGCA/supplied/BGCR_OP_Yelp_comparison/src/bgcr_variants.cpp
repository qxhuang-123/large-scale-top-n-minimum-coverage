#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
namespace py=pybind11;
using namespace pybind11::literals;
using Clock=std::chrono::steady_clock;
constexpr double INF=std::numeric_limits<double>::infinity();

// All modes use exactly the same sorted candidate CSR and initial Top-N.
// Tie rule: loss, new item, user, selected slot (matches public BGCR).
struct Choice {
    double loss=INF; int j=-1,u=-1,pos=-1,edge=-1;
    auto key() const { return std::make_tuple(loss,j,u,pos); }
    bool valid() const { return u>=0 && std::isfinite(loss); }
};
struct Greater { bool operator()(const Choice&a,const Choice&b)const{return a.key()>b.key();} };
using Heap=std::priority_queue<Choice,std::vector<Choice>,Greater>;
struct Stats {
    uint64_t column_edges=0, donor_slot_checks=0, source_refreshes=0;
    uint64_t source_pointer_advances=0, terminal_edge_checks=0, terminal_pointer_advances=0;
    uint64_t heap_pops=0, heap_reinserts=0, item_queries=0, source_dirty_max=0;
};
struct History {int k; double s,t;};
struct Event {int u,pos,oldedge,newedge;};

struct State {
    int U,I,N,D,K=0,K0=0; double score=0,score0=0;
    const int32_t *up,*items,*ip,*iu,*ie; const double *scores,*column_scores;
    std::string mode;
    bool aggregate;
    std::vector<int> selected,selected_items,counts,owner_xor,pool,pool_ptr,source_slot,terminal_ptr;
    std::vector<double> selected_scores,source_value;
    std::vector<History> history; std::vector<Event> events;
    Stats stats; Clock::time_point start;
    double initial_seconds=0, strategy_setup_seconds=0;
    double elapsed()const{return std::chrono::duration<double>(Clock::now()-start).count();}
    State(int u,int i,int n,int d,const int32_t*p,const int32_t*x,const double*s,
          const int32_t*q,const int32_t*v,const int32_t*e,const double*cs,std::string m):
        U(u),I(i),N(n),D(d),up(p),items(x),ip(q),iu(v),ie(e),scores(s),column_scores(cs),mode(m),
        aggregate(m!="original"),start(Clock::now()) {
        selected.resize(U*N);selected_items.resize(U*N);selected_scores.resize(U*N);
        counts.assign(I,0);owner_xor.assign(I,0);
        history.reserve(I+1);events.reserve(I);
        for(int u=0;u<U;++u) {
            if(up[u+1]-up[u]<N)throw std::runtime_error("user has fewer than N candidates");
            for(int p=0;p<N;++p){int slot=u*N+p,e=up[u]+p,j=items[e];
                selected[slot]=e;selected_items[slot]=j;selected_scores[slot]=scores[e];
                ++counts[j];owner_xor[j]^=(slot+1);score+=scores[e];}
        }
        for(int c:counts)K+=(c>0);
        score0=score;K0=K;initial_seconds=elapsed();history.push_back({K,score,initial_seconds});
    }
    void init_sources(){
        pool.resize(U*N);pool_ptr.assign(U,0);source_slot.assign(U,-1);source_value.assign(U,INF);
        // Only initially selected donor edges can ever be donors in direct swaps.
        for(int u=0;u<U;++u){
            auto b=pool.begin()+u*N;std::iota(b,b+N,0);
            std::sort(b,b+N,[&](int a,int c){double sa=scores[up[u]+a],sc=scores[up[u]+c];
                return sa!=sc?sa<sc:a<c;});
            refresh_source(u);
        }
    }
    void refresh_source(int u){
        ++stats.source_refreshes;int &p=pool_ptr[u];
        while(p<N){int pos=pool[u*N+p],e=up[u]+pos;
            if(selected[u*N+pos]==e && counts[items[e]]>=2)break;
            ++p;++stats.source_pointer_advances;
        }
        source_slot[u]=(p<N?pool[u*N+p]:-1);
        source_value[u]=(p<N?selected_scores[u*N+source_slot[u]]:INF);
    }
    Choice item_query(int j){
        ++stats.item_queries;Choice best;
        if(counts[j])return best;
        for(int p=ip[j];p<ip[j+1];++p){
            ++stats.column_edges;int u=iu[p];double add=column_scores[p];
            if(aggregate){int pos=source_slot[u];if(pos<0)continue;
                double loss=source_value[u]-add;
                // Columns are ordered by user; source ties use the first slot.
                if(loss<best.loss)best={loss,j,u,pos,ie[p]};
            }else{
                for(int pos=0;pos<N;++pos){++stats.donor_slot_checks;int slot=u*N+pos;
                    if(counts[selected_items[slot]]<2)continue;
                    double loss=selected_scores[slot]-add;
                    if(loss<best.loss)best={loss,j,u,pos,ie[p]};
                }
            }
        }
        return best;
    }
    Choice user_query(int u){
        int pos=source_slot[u];if(pos<0)return {};
        int &e=terminal_ptr[u];
        while(e<up[u+1]){
            ++stats.terminal_edge_checks;if(counts[items[e]]==0)break;
            ++e;++stats.terminal_pointer_advances;
        }
        if(e==up[u+1])return {};
        return {source_value[u]-scores[e],items[e],u,pos,e};
    }
    void swap(const Choice& c){
        int slot=c.u*N+c.pos,old=selected[slot],i=items[old];
        if(counts[c.j]!=0 || counts[i]<2 || items[c.edge]!=c.j)
            throw std::runtime_error("invalid direct swap");
        --counts[i];++counts[c.j];owner_xor[i]^=(slot+1);owner_xor[c.j]^=(slot+1);
        selected[slot]=c.edge;selected_items[slot]=c.j;selected_scores[slot]=scores[c.edge];
        score+=scores[c.edge]-scores[old];++K;
        events.push_back({c.u,c.pos,old,c.edge});
        if(aggregate){
            refresh_source(c.u);uint64_t dirty=1;
            if(counts[i]==1){int remaining=(owner_xor[i]-1)/N;
                if(remaining!=c.u){refresh_source(remaining);++dirty;}}
            stats.source_dirty_max=std::max(stats.source_dirty_max,dirty);
        }
        history.push_back({K,score,elapsed()});
    }
    void run(uint64_t seed){
        if(K>=D)return;
        if(aggregate)init_sources();
        if(mode=="original" || mode=="fixed_source"){
            std::vector<Choice> order;
            for(int j=0;j<I;++j)if(!counts[j]){auto c=item_query(j);if(c.valid())order.push_back(c);}
            std::sort(order.begin(),order.end(),[](const Choice&a,const Choice&b){return a.key()<b.key();});
            strategy_setup_seconds=elapsed()-initial_seconds;
            for(auto old:order){auto c=item_query(old.j);if(c.valid())swap(c);if(K>=D)break;}
        }else if(mode=="random_source" || mode=="id_source"){
            std::vector<int> order;for(int j=0;j<I;++j)if(!counts[j] && ip[j]<ip[j+1])order.push_back(j);
            if(mode=="random_source"){std::mt19937_64 rng(seed);std::shuffle(order.begin(),order.end(),rng);}
            strategy_setup_seconds=elapsed()-initial_seconds;
            for(int j:order){auto c=item_query(j);if(c.valid())swap(c);if(K>=D)break;}
        }else if(mode=="dynamic_item_heap"){
            Heap h;for(int j=0;j<I;++j)if(!counts[j]){auto c=item_query(j);if(c.valid())h.push(c);}
            strategy_setup_seconds=elapsed()-initial_seconds;
            while(!h.empty() && K<D){auto old=h.top();h.pop();++stats.heap_pops;
                auto c=item_query(old.j);if(!c.valid())continue;
                if(c.key()<old.key())throw std::runtime_error("non-monotone item lower bound");
                if(h.empty() || c.key()<=h.top().key())swap(c);
                else{h.push(c);++stats.heap_reinserts;}
            }
        }else if(mode=="dynamic_user_heap"){
            terminal_ptr.assign(up,up+U);Heap h;
            for(int u=0;u<U;++u){auto c=user_query(u);if(c.valid())h.push(c);}
            strategy_setup_seconds=elapsed()-initial_seconds;
            while(!h.empty() && K<D){auto old=h.top();h.pop();++stats.heap_pops;
                auto c=user_query(old.u);if(!c.valid())continue;
                if(c.key()<old.key())throw std::runtime_error("non-monotone user lower bound");
                if(h.empty() || c.key()<=h.top().key()){
                    swap(c);auto next=user_query(c.u);if(next.valid())h.push(next);
                }else{h.push(c);++stats.heap_reinserts;}
            }
        }else throw std::runtime_error("unknown mode");
    }
};

py::dict solve(py::array_t<int32_t,py::array::c_style> up,
    py::array_t<int32_t,py::array::c_style> items,py::array_t<double,py::array::c_style> scores,
    py::array_t<int32_t,py::array::c_style> ip,py::array_t<int32_t,py::array::c_style> iu,
    py::array_t<int32_t,py::array::c_style> ie,py::array_t<double,py::array::c_style> column_scores,
    int N,int D,std::string mode,uint64_t seed){
    int U=up.size()-1,I=ip.size()-1;
    if(U<=0 || I<=0 || N<=0 || D<0 || D>I || items.size()!=scores.size() || iu.size()!=ie.size() || column_scores.size()!=ie.size())
        throw std::runtime_error("invalid input shape or parameter");
    State s(U,I,N,D,up.data(),items.data(),scores.data(),ip.data(),iu.data(),ie.data(),column_scores.data(),mode);
    s.run(seed);double core_seconds=s.elapsed();
    py::array_t<int32_t> selected({U,N});std::copy(s.selected.begin(),s.selected.end(),selected.mutable_data());
    py::array_t<double> history({int(s.history.size()),3});auto hp=history.mutable_data();
    for(auto r:s.history){*hp++=r.k;*hp++=r.s;*hp++=r.t;}
    py::array_t<int32_t> events({int(s.events.size()),4});auto ep=events.mutable_data();
    for(auto e:s.events){*ep++=e.u;*ep++=e.pos;*ep++=e.oldedge;*ep++=e.newedge;}
    auto &t=s.stats;
    py::dict stats("column_edges"_a=t.column_edges,"donor_slot_checks"_a=t.donor_slot_checks,
        "source_refreshes"_a=t.source_refreshes,"source_pointer_advances"_a=t.source_pointer_advances,
        "terminal_edge_checks"_a=t.terminal_edge_checks,"terminal_pointer_advances"_a=t.terminal_pointer_advances,
        "heap_pops"_a=t.heap_pops,"heap_reinserts"_a=t.heap_reinserts,"item_queries"_a=t.item_queries,
        "source_dirty_max"_a=t.source_dirty_max);
    return py::dict("mode"_a=mode,"initial_coverage"_a=s.K0,"initial_objective"_a=s.score0,
        "coverage"_a=s.K,"objective"_a=s.score,"target_reached"_a=(s.K>=D),
        "swaps"_a=s.events.size(),"selected"_a=selected,"history"_a=history,"events"_a=events,
        "stats"_a=stats,"core_seconds"_a=core_seconds,"initial_seconds"_a=s.initial_seconds,
        "strategy_setup_seconds"_a=s.strategy_setup_seconds);
}
PYBIND11_MODULE(bgcr_variants,m){m.def("solve",&solve,"user_indptr"_a,"user_items"_a,"user_scores"_a,
    "item_indptr"_a,"item_users"_a,"item_edges"_a,"item_scores"_a,"N"_a,"D"_a,"mode"_a,"seed"_a=0);}
