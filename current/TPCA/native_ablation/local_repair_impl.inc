// Exact shortest-path-tree repair in either exchange-graph orientation.
// Forward follows the manuscript: donor item -> user -> uncovered item.
// Reverse follows the published C++ code: uncovered item -> user -> donor item.
// Included after the original core, in the same translation unit.
namespace {
struct LocalRepair {
    RunState& st;
    bool forward;
    DijkstraScratch sc;
    std::vector<int> child, next, prev;
    std::vector<uint8_t> active, inside;
    // Selected incidences use permanent user slots, requiring O(U*N), not O(E).
    std::vector<int> slot_head, slot_next, slot_prev;
    std::vector<int> versions;
    using Target = std::tuple<double,int,int>;
    std::priority_queue<Target,std::vector<Target>,std::greater<Target>> targets;
    uint64_t scans=0, repairs=0, total_affected=0, max_affected=0;
    std::vector<int> repair_sizes;

    // Exact query accelerators. Stale boundary keys are lower bounds because
    // forward local repair only increases user potentials.
    bool boundary_oracle, intersection_oracle;
    struct BoundaryEntry { double key; int user, edge; };
    struct BoundaryGreater {
        bool operator()(const BoundaryEntry& a,const BoundaryEntry& b) const {
            return a.key!=b.key ? a.key>b.key : a.user>b.user;
        }
    };
    std::vector<std::vector<BoundaryEntry>> boundary_heaps;
    std::vector<uint8_t> boundary_built;
    std::vector<BoundaryEntry> excluded;
    int words=0;
    std::vector<uint64_t> candidate_bits, affected_bits;
    std::vector<int> candidate_rank, candidate_sorted_edges, affected_words, matching_edges;
    uint64_t heap_build_edges=0, heap_queries=0, heap_top_checks=0,
        heap_refreshes=0, heap_exclusions=0, intersection_build_edges=0,
        intersection_word_checks=0, intersection_matches=0,
        boundary_scan_checks=0, internal_scan_checks=0;
    uint64_t positive_user_price_updates=0;
    double intersection_build_seconds=0;

    explicit LocalRepair(RunState& s,bool f=false,bool bh=false,bool ix=false): st(s),forward(f), child(s.nodes,-1),next(s.nodes,-1),prev(s.nodes,-1),
        active(s.nodes,0),inside(s.nodes,0),slot_head(s.I,-1),
        slot_next(s.U*s.N,-1),slot_prev(s.U*s.N,-1),versions(s.I,0),
        boundary_oracle(bh),intersection_oracle(ix) {
        if((bh||ix)&&!f)throw std::runtime_error("edge oracles require forward orientation");
        if(bh){boundary_heaps.resize(st.I);boundary_built.assign(st.I,0);}
        for(int u=0;u<st.U;++u)
            for(int p=0;p<(int)st.user_list_edges[u].size();++p)
                slot_link(u*st.N+p,st.edge_item(st.user_list_edges[u][p]));
    }
    int boundary_min(int item) {
        ++heap_queries;
        auto& h=boundary_heaps[item];
        const BoundaryGreater greater;
        if(!boundary_built[item]) {
            h.reserve(st.item_indptr[item+1]-st.item_indptr[item]);
            for(int p=st.item_indptr[item];p<st.item_indptr[item+1];++p) {
                int u=st.item_users[p],e=st.item_edges[p];
                h.push_back({st.potential[st.I+u]-st.edge_score(e),u,e});
                ++heap_build_edges;
            }
            std::make_heap(h.begin(),h.end(),greater);boundary_built[item]=1;
        }
        excluded.clear();
        int result=-1;
        while(!h.empty()) {
            ++heap_top_checks;
            const auto top=h.front();int a=st.I+top.user;
            if(inside[a]||!active[a]||st.selected_edge[top.edge]) {
                std::pop_heap(h.begin(),h.end(),greater);h.pop_back();
                excluded.push_back(top);++heap_exclusions;continue;
            }
            double key=st.potential[a]-st.edge_score(top.edge);
            if(key!=top.key) {
                if(key<top.key)throw std::runtime_error("boundary key decreased");
                std::pop_heap(h.begin(),h.end(),greater);
                h.back().key=key;std::push_heap(h.begin(),h.end(),greater);
                ++heap_refreshes;continue;
            }
            result=top.edge;break;
        }
        // Restore exclusions: membership of the repair set and selection can
        // change in later iterations. No edge is permanently discarded.
        for(const auto& entry:excluded) {
            h.push_back(entry);std::push_heap(h.begin(),h.end(),greater);
        }
        return result;
    }
    void build_intersection_index() {
        if(words)return;
        auto start=std::chrono::steady_clock::now();
        words=(st.I+63)/64;
        candidate_bits.assign((size_t)st.U*words,0);
        candidate_rank.resize((size_t)st.U*words);
        candidate_sorted_edges.resize(st.E);
        affected_bits.assign(words,0);
        std::vector<int> cursor(st.user_indptr,st.user_indptr+st.U);
        // Item-major traversal constructs ascending-item lists without sorting.
        for(int j=0;j<st.I;++j)
            for(int p=st.item_indptr[j];p<st.item_indptr[j+1];++p) {
                int u=st.item_users[p];
                candidate_sorted_edges[cursor[u]++]=st.item_edges[p];
                candidate_bits[(size_t)u*words+j/64]|=uint64_t(1)<<(j%64);
                ++intersection_build_edges;
            }
        for(int u=0;u<st.U;++u) {
            int rank=0;
            for(int w=0;w<words;++w) {
                size_t pos=(size_t)u*words+w;
                candidate_rank[pos]=rank;rank+=__builtin_popcountll(candidate_bits[pos]);
            }
        }
        intersection_build_seconds=std::chrono::duration<double>(
            std::chrono::steady_clock::now()-start).count();
    }
    void relax_intersection(int u,int v) {
        matching_edges.clear();
        for(int w:affected_words) {
            ++intersection_word_checks;size_t pos=(size_t)u*words+w;
            uint64_t row=candidate_bits[pos],match=row&affected_bits[w];
            while(match) {
                int bit=__builtin_ctzll(match);match&=match-1;
                int rank=candidate_rank[pos]+__builtin_popcountll(row&((uint64_t(1)<<bit)-1));
                matching_edges.push_back(candidate_sorted_edges[st.user_indptr[u]+rank]);
            }
        }
        // Preserve the original row's relaxation order, including all ties.
        std::sort(matching_edges.begin(),matching_edges.end());
        for(int e:matching_edges) {
            ++scans;++intersection_matches;
            if(!st.selected_edge[e])relax(st,v,st.edge_item(e),-st.edge_score(e),e,sc);
        }
    }
    void slot_link(int slot,int item) {
        slot_prev[slot]=-1; slot_next[slot]=slot_head[item];
        if(slot_head[item]>=0)slot_prev[slot_head[item]]=slot;
        slot_head[item]=slot;
    }
    void slot_unlink(int slot,int item) {
        if(slot_prev[slot]>=0)slot_next[slot_prev[slot]]=slot_next[slot];
        else slot_head[item]=slot_next[slot];
        if(slot_next[slot]>=0)slot_prev[slot_next[slot]]=slot_prev[slot];
        slot_prev[slot]=slot_next[slot]=-1;
    }
    void link(int v,int p) {
        next[v]=child[p];prev[v]=-1;
        if(child[p]>=0)prev[child[p]]=v;
        child[p]=v;
    }
    void unlink(int v) {
        int p=sc.pred[v];
        if(p<0)return;
        if(prev[v]>=0)next[prev[v]]=next[v];else child[p]=next[v];
        if(next[v]>=0)prev[next[v]]=prev[v];
        next[v]=prev[v]=-1;
    }
    void target_push(int j) {
        if(active[j] && (forward ? st.counts[j]==0 : st.counts[j]>=2))
            targets.emplace(st.potential[j]-st.potential[st.sigma],j,versions[j]);
    }
    int target_pop() {
        while(!targets.empty()) {
            auto [d,j,ver]=targets.top();targets.pop();
            if(active[j]&&(forward ? st.counts[j]==0 : st.counts[j]>=2)&&versions[j]==ver)return j;
        }
        return -1;
    }
    // Same forward graph and selected-incidence index, but recompute the
    // shortest path from scratch each time. This isolates tree reuse.
    bool ssp_once() {
        ++st.full_calls;
        sc.reset(st.nodes);sc.dist[st.sigma]=0;heap_push(sc.heap,{0,st.sigma});
        while(!sc.heap.empty()) {
            auto [dv,v]=heap_pop(sc.heap);
            if(dv!=sc.dist[v])continue;
            if(v==st.tau)break;
            if(v==st.sigma) {
                for(int j=0;j<st.I;++j)if(st.counts[j]>=2) {
                    ++scans;relax(st,v,j,0,-1,sc);
                }
            } else if(v<st.I) {
                if(st.counts[v]==0){++scans;relax(st,v,st.tau,0,-1,sc);}
                for(int slot=slot_head[v];slot>=0;slot=slot_next[slot]) {
                    ++scans;int u=slot/st.N,e=st.user_list_edges[u][slot%st.N];
                    relax(st,v,st.I+u,st.edge_score(e),e,sc);
                }
            } else {
                int u=v-st.I;
                for(int e=st.user_indptr[u];e<st.user_indptr[u+1];++e) {
                    ++scans;if(!st.selected_edge[e])relax(st,v,st.edge_item(e),-st.edge_score(e),e,sc);
                }
            }
        }
        if(!std::isfinite(sc.dist[st.tau]))return false;
        std::vector<int> edges;
        for(int v=st.tau;v!=st.sigma;v=sc.pred[v]) {
            if(v<0)throw std::runtime_error("bad forward SSP path");
            if(sc.pred_edge[v]>=0)edges.push_back(sc.pred_edge[v]);
        }
        const double cap=sc.dist[st.tau];
        for(int v=0;v<st.nodes;++v)st.potential[v]+=std::min(sc.dist[v],cap);
        for(size_t p=0;p<edges.size();p+=2) {
            int add=edges[p],rem=edges[p+1];
            int slot=st.edge_user(rem)*st.N+st.selected_edge_pos[rem];
            slot_unlink(slot,st.edge_item(rem));slot_link(slot,st.edge_item(add));
        }
        apply_path(st,edges);return true;
    }
    void full_search() {
        sc.reset(st.nodes);sc.dist[st.sigma]=0;
        heap_push(sc.heap,{0,st.sigma});
        while(!sc.heap.empty()) {
            auto [dv,v]=heap_pop(sc.heap);
            if(dv!=sc.dist[v])continue;
            if(v==st.sigma) {
                for(int j=0;j<st.I;++j)if(forward ? st.counts[j]>=2 : st.counts[j]==0) {
                    ++scans;relax(st,v,j,0,-1,sc);
                }
            } else if(v<st.I) {
                if(forward) {
                    for(int slot=slot_head[v];slot>=0;slot=slot_next[slot]) {
                        ++scans;int u=slot/st.N,e=st.user_list_edges[u][slot%st.N];
                        relax(st,v,st.I+u,st.edge_score(e),e,sc);
                    }
                } else {
                    for(int p=st.item_indptr[v];p<st.item_indptr[v+1];++p) {
                        ++scans;int e=st.item_edges[p];
                        if(!st.selected_edge[e])relax(st,v,st.I+st.item_users[p],-st.edge_score(st.item_edges[p]),e,sc);
                    }
                }
            } else if(v<st.sigma) {
                int u=v-st.I;
                if(forward) {
                    for(int e=st.user_indptr[u];e<st.user_indptr[u+1];++e) {
                        ++scans;if(!st.selected_edge[e])relax(st,v,st.edge_item(e),-st.edge_score(e),e,sc);
                    }
                } else {
                    for(int e:st.user_list_edges[u]) {
                        ++scans;relax(st,v,st.edge_item(e),st.edge_score(e),e,sc);
                    }
                }
            }
        }
        for(int v=0;v<st.nodes;++v)if(std::isfinite(sc.dist[v])) {
            st.potential[v]+=sc.dist[v];active[v]=1;
            if(sc.pred[v]>=0)link(v,sc.pred[v]);
        }
        for(int j=0;j<st.I;++j)target_push(j);
    }
    // Return the subtree rooted at the first disappearing shortest-tree edge.
    std::vector<int> augment(int terminal) {
        std::vector<int> edges,path;
        for(int v=terminal;v!=st.sigma;v=sc.pred[v]) {
            if(v<0 || path.size()>(size_t)st.nodes)throw std::runtime_error("bad tree path");
            path.push_back(v);
            if(sc.pred_edge[v]>=0)edges.push_back(sc.pred_edge[v]);
        }
        if(path.empty() || edges.empty() || edges.size()%2)throw std::runtime_error("bad augmentation");
        if(!forward)std::reverse(edges.begin(),edges.end());
        int root=path.back();
        if(forward) {
            if(st.counts[root]<2)throw std::runtime_error("source donor is not redundant");
            if(st.counts[root]>2)root=path[path.size()-2];
        } else if(st.counts[root]!=0)throw std::runtime_error("source item is already covered");
        std::vector<int> affected{root};
        // No repair/tree traversal is needed after the final augmentation.
        if(st.coverage+1<st.D)
            for(size_t p=0;p<affected.size();++p)
                for(int v=child[affected[p]];v>=0;v=next[v])affected.push_back(v);
        for(size_t p=0;p<edges.size();p+=2) {
            int add=edges[p],rem=edges[p+1];
            int slot=st.edge_user(rem)*st.N+st.selected_edge_pos[rem];
            slot_unlink(slot,st.edge_item(rem));slot_link(slot,st.edge_item(add));
        }
        int old=st.coverage;
        apply_path(st,edges);
        if(st.coverage!=old+1)throw std::runtime_error("coverage increment is not one");
        return affected;
    }
    void repair(const std::vector<int>& affected) {
        ++repairs;total_affected+=affected.size();
        max_affected=std::max<uint64_t>(max_affected,affected.size());
        repair_sizes.push_back((int)affected.size());
        for(int v:affected)inside[v]=1;
        if(intersection_oracle) {
            build_intersection_index();affected_words.clear();
            std::fill(affected_bits.begin(),affected_bits.end(),0);
            for(int v:affected)if(v<st.I)affected_bits[v/64]|=uint64_t(1)<<(v%64);
            for(int w=0;w<words;++w)if(affected_bits[w])affected_words.push_back(w);
        }
        for(int v:affected)unlink(v);
        for(int v:affected){sc.dist[v]=INF;sc.pred[v]=sc.pred_edge[v]=-1;}
        sc.heap.clear();
        // Outside vertices retain their exact old distance, so correction is zero.
        auto boundary=[&](int a,int b,double c,int e) {
            sc.dist[a]=0;relax(st,a,b,c,e,sc);
        };
        for(int v:affected) {
            if(v<st.I) {
                if(forward) {
                    if(boundary_oracle) {
                        int e=boundary_min(v);
                        if(e>=0){++scans;boundary(st.I+st.edge_user(e),v,-st.edge_score(e),e);}
                    } else {
                        for(int p=st.item_indptr[v];p<st.item_indptr[v+1];++p) {
                            ++scans;++boundary_scan_checks;int e=st.item_edges[p],a=st.I+st.item_users[p];
                            if(!st.selected_edge[e]&&!inside[a]&&active[a])boundary(a,v,-st.edge_score(st.item_edges[p]),e);
                        }
                    }
                } else {
                    for(int slot=slot_head[v];slot>=0;slot=slot_next[slot]) {
                        ++scans;int u=slot/st.N,a=st.I+u;
                        int e=st.user_list_edges[u][slot%st.N];
                        if(!inside[a]&&active[a])boundary(a,v,st.edge_score(e),e);
                    }
                }
                if(forward ? st.counts[v]>=2 : st.counts[v]==0) {++scans;boundary(st.sigma,v,0,-1);}
            } else {
                int u=v-st.I;
                if(forward) {
                    for(int e:st.user_list_edges[u]) {
                        ++scans;int a=st.edge_item(e);
                        if(!inside[a]&&active[a])boundary(a,v,st.edge_score(e),e);
                    }
                } else {
                    for(int e=st.user_indptr[u];e<st.user_indptr[u+1];++e) {
                        ++scans;int a=st.edge_item(e);
                        if(!st.selected_edge[e]&&!inside[a]&&active[a])boundary(a,v,-st.edge_score(e),e);
                    }
                }
            }
        }
        while(!sc.heap.empty()) {
            auto [dv,v]=heap_pop(sc.heap);
            if(dv!=sc.dist[v])continue;
            if(v<st.I) {
                if(forward) {
                    for(int slot=slot_head[v];slot>=0;slot=slot_next[slot]) {
                        ++scans;int u=slot/st.N,b=st.I+u;
                        int e=st.user_list_edges[u][slot%st.N];
                        if(inside[b])relax(st,v,b,st.edge_score(e),e,sc);
                    }
                } else {
                    for(int p=st.item_indptr[v];p<st.item_indptr[v+1];++p) {
                        ++scans;int e=st.item_edges[p],b=st.I+st.item_users[p];
                        if(inside[b]&&!st.selected_edge[e])relax(st,v,b,-st.edge_score(st.item_edges[p]),e,sc);
                    }
                }
            } else {
                int u=v-st.I;
                if(forward) {
                    if(intersection_oracle)relax_intersection(u,v);
                    else for(int e=st.user_indptr[u];e<st.user_indptr[u+1];++e) {
                        ++scans;++internal_scan_checks;int b=st.edge_item(e);
                        if(!st.selected_edge[e]&&inside[b])relax(st,v,b,-st.edge_score(e),e,sc);
                    }
                } else {
                    for(int e:st.user_list_edges[u]) {
                        ++scans;int b=st.edge_item(e);
                        if(inside[b])relax(st,v,b,st.edge_score(e),e,sc);
                    }
                }
            }
        }
        for(int v:affected) {
            if(std::isfinite(sc.dist[v])) {
                if(v>=st.I && v<st.sigma && sc.dist[v]>0)++positive_user_price_updates;
                st.potential[v]+=sc.dist[v];active[v]=1;link(v,sc.pred[v]);
            } else {active[v]=0;sc.pred[v]=sc.pred_edge[v]=-1;}
            if(v<st.I){++versions[v];target_push(v);}
            inside[v]=0;
        }
    }
};
} // namespace
