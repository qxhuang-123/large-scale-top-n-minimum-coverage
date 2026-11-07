// Only DGCA maintenance is changed. The original constructor, Top-N selection,
// score representation, donor maintenance, accepted swap and trace are reused.
#include "vendor/reference_core.cpp"

static inline bool earlier(const Choice& a, const Choice& b) { return a.key() < b.key(); }

// Keys stored in this heap are snapshots, separate from pending group minima.
struct GroupQueue {
    std::vector<Choice> h;
    std::vector<int> position;
    explicit GroupQueue(int I) : position(I, -1) { h.reserve(I); }
    void put(int p, const Choice& c) { h[p] = c; position[c.j] = p; }
    void down(int p) {
        Choice c = h[p];
        for (;;) {
            int child = 2 * p + 1;
            if (child >= int(h.size())) break;
            if (child + 1 < int(h.size()) && earlier(h[child+1], h[child])) ++child;
            if (!earlier(h[child], c)) break;
            put(p, h[child]); p = child;
        }
        put(p, c);
    }
    void up(int p) {
        Choice c = h[p];
        while (p) {
            int parent = (p-1)/2;
            if (!earlier(c, h[parent])) break;
            put(p, h[parent]); p = parent;
        }
        put(p, c);
    }
    void update(int j, const Choice& c) {
        int p = position[j];
        if (!c.valid()) {
            if (p < 0) return;
            Choice last = h.back(); h.pop_back(); position[j] = -1;
            if (p == int(h.size())) return;
            put(p, last);
        } else if (p < 0) {
            p = int(h.size()); h.push_back(c); position[j] = p;
        } else put(p, c);
        if (p > 0 && earlier(h[p], h[(p-1)/2])) up(p); else down(p);
    }
    void build(const std::vector<Choice>& minima) {
        for (auto c : minima) if (c.valid()) {
            position[c.j] = int(h.size()); h.push_back(c);
        }
        for (int p = int(h.size())/2-1; p >= 0; --p) down(p);
    }
};

struct Stage2State : State {
    using State::State;
    uint64_t queries=0, cached=0, root_updates=0, transfers=0, rescans=0, scanned_users=0;
    Choice query(int u) { ++queries; return user_query(u); }
    bool fresh(const Choice& c) const {
        // Original State::swap eagerly updates the moving user's donor and the
        // remaining owner if a donor becomes singly covered. The cached pair
        // remains optimal exactly when its source is unchanged and target free.
        return source_slot[c.u] == c.pos && !counts[c.j];
    }
    void accept(const Choice& c) {
        State::swap(c);
        // This edge has just become covered, so it need not be checked again.
        terminal_ptr[c.u] = c.edge + 1;
        ++stats.terminal_pointer_advances;
    }
    static void sift(std::vector<Choice>& h, int p) {
        Choice c = h[p];
        for (;;) {
            int child=2*p+1;
            if (child>=int(h.size())) break;
            if (child+1<int(h.size()) && earlier(h[child+1],h[child])) ++child;
            if (!earlier(h[child],c)) break;
            h[p]=h[child]; p=child;
        }
        h[p]=c;
    }
    void replace_root(std::vector<Choice>& h, const Choice& c) {
        ++root_updates;
        if (c.valid()) h[0]=c; else {h[0]=h.back();h.pop_back();}
        if (!h.empty()) sift(h,0);
    }
    void user_heap() {
        std::vector<Choice> heap; heap.reserve(U);
        for (int u=0;u<U;++u) {auto c=query(u);if(c.valid())heap.push_back(c);}
        for (int p=int(heap.size())/2-1;p>=0;--p) sift(heap,p);
        strategy_setup_seconds=elapsed()-initial_seconds;
        while (K<D && !heap.empty()) {
            Choice old=heap[0],c;
            if (fresh(old)) {c=old;++cached;} else c=query(old.u);
            if (!c.valid()) {replace_root(heap,{});continue;}
            if (earlier(c,old)) throw std::logic_error("decreasing user key");
            bool first=(heap.size()<2 || !earlier(heap[1],c)) &&
                       (heap.size()<3 || !earlier(heap[2],c));
            if (first) {
                accept(c);
                if (K>=D) break;
                replace_root(heap,query(c.u));
            } else replace_root(heap,c);
        }
    }
    void grouped_heap() {
        std::vector<int> head(I,-1),next(U,-1),touched;
        std::vector<unsigned char> dirty(I,0);
        std::vector<Choice> best(I);
        GroupQueue heap(I);
        auto attach=[&](const Choice& c,bool mark) {
            if(!c.valid())return;
            next[c.u]=head[c.j];head[c.j]=c.u;++transfers;
            if(!best[c.j].valid() || earlier(c,best[c.j])) {
                best[c.j]=c;
                if(mark && !dirty[c.j]) {dirty[c.j]=1;touched.push_back(c.j);}
            }
        };
        for(int u=0;u<U;++u)attach(query(u),false);
        heap.build(best);strategy_setup_seconds=elapsed()-initial_seconds;
        while(K<D && !heap.h.empty()) {
            Choice c=heap.h[0];int j=c.j;
            if(!fresh(c)) {
                ++rescans;Choice replacement;
                int *link=&head[j];
                while(*link>=0) {
                    int u=*link;++scanned_users;auto v=query(u);
                    if(!v.valid()){*link=next[u];next[u]=-1;}
                    else {
                        if(v.j!=j)throw std::logic_error("uncovered group changed endpoint");
                        if(!replacement.valid() || earlier(v,replacement))replacement=v;
                        link=&next[u];
                    }
                }
                best[j]=replacement;heap.update(j,replacement);continue;
            }
            ++cached;accept(c);
            if(K>=D)break;
            best[j]={};heap.update(j,{});
            int u=head[j];head[j]=-1;
            while(u>=0) {
                int following=next[u];next[u]=-1;
                attach(query(u),true);u=following;
            }
            for(int item:touched){heap.update(item,best[item]);dirty[item]=0;}
            touched.clear();
        }
    }
    void run() {
        if(K>=D)return;
        init_sources(); // The original donor pools and useful score caches.
        terminal_ptr.assign(up,up+U);
        // These N edges are already covered after the identical Top-N phase.
        for(int u=0;u<U;++u)terminal_ptr[u]+=N;
        if(mode=="dynamic_user_heap_cached")user_heap();else grouped_heap();
    }
};

py::dict solve_stage2(py::array_t<int32_t,py::array::c_style> up,
                     py::array_t<int32_t,py::array::c_style> items,
                     py::array_t<double,py::array::c_style> scores,
                     int I,int N,int D,std::string mode,uint64_t seed) {
    int U=up.size()-1;
    if(U<=0||I<=0||N<=0||D<0||D>I||items.size()!=scores.size())
        throw std::runtime_error("invalid forward input");
    if(mode!="dynamic_user_heap_cached" && mode!="dynamic_user_grouped_cached")
        throw std::runtime_error("unknown stage2 mode");
    Stage2State s(U,I,N,D,up.data(),items.data(),scores.data(),nullptr,nullptr,nullptr,nullptr,mode);
    s.run();double core=s.elapsed();
    py::array_t<int32_t> selected({U,N});std::copy(s.selected.begin(),s.selected.end(),selected.mutable_data());
    py::array_t<double> history({int(s.history.size()),3});auto hp=history.mutable_data();
    for(auto r:s.history){*hp++=r.k;*hp++=r.s;*hp++=r.t;}
    py::array_t<int32_t> events({int(s.events.size()),4});auto ep=events.mutable_data();
    for(auto e:s.events){*ep++=e.u;*ep++=e.pos;*ep++=e.oldedge;*ep++=e.newedge;}
    const auto& t=s.stats;
    return py::dict("mode"_a=mode,"initial_coverage"_a=s.K0,"initial_objective"_a=s.score0,
        "coverage"_a=s.K,"objective"_a=s.score,"target_reached"_a=(s.K>=D),
        "swaps"_a=s.events.size(),"selected"_a=selected,"events"_a=events,"history"_a=history,
        "core_seconds"_a=core,"initial_seconds"_a=s.initial_seconds,
        "strategy_setup_seconds"_a=s.strategy_setup_seconds,
        "stats"_a=py::dict("candidate_queries"_a=s.queries,"cache_hits"_a=s.cached,
            "root_updates"_a=s.root_updates,"bucket_transfers"_a=s.transfers,
            "bucket_rescans"_a=s.rescans,"bucket_scan_users"_a=s.scanned_users,
            "terminal_edge_checks"_a=t.terminal_edge_checks,"terminal_pointer_advances"_a=t.terminal_pointer_advances,
            "source_refreshes"_a=t.source_refreshes,"source_pointer_advances"_a=t.source_pointer_advances));
}
PYBIND11_MODULE(dgca_stage2,m) {
    m.def("solve_user",&solve_stage2,"user_indptr"_a,"user_items"_a,"user_scores"_a,
          "num_items"_a,"N"_a,"D"_a,"mode"_a="dynamic_user_heap_cached","seed"_a=0);
}
