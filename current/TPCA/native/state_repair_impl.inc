// Coverage-state compression; the old LocalRepair implementation is unchanged.
// Preconditions: each user's candidate row is score-descending, item-id-ascending.
// Terminal rows are queried lazily; source pools contain initial selected donors.
namespace {
struct StateRepair : LocalRepair {
    bool source_compressed, activation_index, terminal_compressed;
    std::vector<int> terminal_pos, source_pos, source_edge;
    std::vector<std::vector<int>> source_pool, activated_rows;
    std::vector<uint8_t> activated_items;
    // Same terminal tie rule as the baseline: loss, then item id.
    using UserTarget = std::tuple<double,int,int>;
    std::vector<UserTarget> user_targets;
    uint64_t terminal_queries=0, terminal_checks=0, terminal_refreshes=0,
        terminal_pointer_advances=0, terminal_exhaustions=0,
        source_refresh_calls=0, source_pointer_advances=0, source_pool_checks=0,
        activated_edges=0, activated_columns=0, source_class_changes=0,
        source_dirty_max=0, repair_components_max=0, source_cost_increases=0,
        source_equal_witness_changes=0, source_preserved_tree_arcs=0,
        direct_augmentations=0;
    double activation_seconds=0;

    StateRepair(RunState& s,bool bh,bool source,bool activation,bool terminal=true):
        LocalRepair(s,true,bh,false),source_compressed(source),activation_index(activation),terminal_compressed(terminal),
        terminal_pos(s.user_indptr,s.user_indptr+s.U) {
        if(source_compressed) {
            source_pool.resize(st.U);source_pos.assign(st.U,0);source_edge.assign(st.U,-1);
            for(int u=0;u<st.U;++u) {
                for(int e:st.user_list_edges[u]) {
                    ++source_pool_checks;
                    if(st.counts[st.edge_item(e)]>=2)source_pool[u].push_back(e);
                }
                std::sort(source_pool[u].begin(),source_pool[u].end(),[&](int a,int b) {
                    return st.edge_score(a)!=st.edge_score(b) ? st.edge_score(a)<st.edge_score(b) :
                        st.edge_item(a)<st.edge_item(b);
                });
                refresh_source(u);
            }
        }
        if(activation_index) {
            activated_rows.resize(st.U);activated_items.assign(st.I,0);
            for(int j=0;j<st.I;++j)if(core_item(j))activate(j);
        }
    }
    bool core_item(int j) const {
        return source_compressed ? (st.counts[j]==1 || (!terminal_compressed && st.counts[j]==0)) : st.counts[j]>0;
    }
    void activate(int j) {
        if(!activation_index || activated_items[j])return;
        auto start=std::chrono::steady_clock::now();
        activated_items[j]=1;++activated_columns;
        for(int p=st.item_indptr[j];p<st.item_indptr[j+1];++p) {
            activated_rows[st.item_users[p]].push_back(st.item_edges[p]);++activated_edges;
        }
        activation_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
    int refresh_source(int u) {
        ++source_refresh_calls;
        auto& p=source_pos[u];const auto& pool=source_pool[u];
        while(p<(int)pool.size()) {
            int e=pool[p];
            if(st.selected_edge[e] && st.counts[st.edge_item(e)]>=2)break;
            ++p;++source_pointer_advances;
        }
        return source_edge[u]=p<(int)pool.size() ? pool[p] : -1;
    }
    int terminal_edge(int u) {
        int& e=terminal_pos[u];
        while(e<st.user_indptr[u+1] && st.counts[st.edge_item(e)]>0) {
            ++e;++terminal_pointer_advances;
        }
        return e<st.user_indptr[u+1] ? e : -1;
    }
    void build_targets() {
        for(int u=0;u<st.U;++u)if(active[st.I+u]) {
            int e=terminal_edge(u);
            if(e>=0)user_targets.emplace_back(st.potential[st.I+u]-st.edge_score(e),st.edge_item(e),u);
        }
        std::make_heap(user_targets.begin(),user_targets.end(),std::greater<UserTarget>());
    }
    int choose_terminal() {
        if(!terminal_compressed)return target_pop();
        ++terminal_queries;
        while(!user_targets.empty()) {
            ++terminal_checks;
            auto top=user_targets.front();int u=std::get<2>(top),v=st.I+u;
            int e=active[v] ? terminal_edge(u) : -1;
            if(e<0) {
                std::pop_heap(user_targets.begin(),user_targets.end(),std::greater<UserTarget>());
                user_targets.pop_back();++terminal_exhaustions;continue;
            }
            double key=st.potential[v]-st.edge_score(e);
            UserTarget current{key,st.edge_item(e),u};
            if(current!=top) {
                if(key<std::get<0>(top)-1e-8)throw std::runtime_error("terminal key decreased");
                std::pop_heap(user_targets.begin(),user_targets.end(),std::greater<UserTarget>());
                user_targets.back()=current;
                std::push_heap(user_targets.begin(),user_targets.end(),std::greater<UserTarget>());
                ++terminal_refreshes;continue;
            }
            // Materialize only the winning uncovered item as a virtual SPT leaf.
            int j=st.edge_item(e);
            if(sc.pred[j]>=0 || child[j]>=0)throw std::runtime_error("uncovered terminal in core tree");
            st.potential[j]=key;active[j]=1;sc.pred[j]=v;sc.pred_edge[j]=e;link(j,v);
            return j;
        }
        return -1;
    }
    void relax_row(int u,bool restricted) {
        int v=st.I+u;
        auto visit=[&](int e) {
            ++scans;if(restricted)++internal_scan_checks;
            int j=st.edge_item(e);
            if(st.selected_edge[e])return;
            if(restricted ? !inside[j] : !core_item(j))return;
            relax(st,v,j,-st.edge_score(e),e,sc);
        };
        if(activation_index)for(int e:activated_rows[u])visit(e);
        else for(int e=st.user_indptr[u];e<st.user_indptr[u+1];++e)visit(e);
    }
    void initial_search() {
        sc.reset(st.nodes);sc.dist[st.sigma]=0;heap_push(sc.heap,{0,st.sigma});
        while(!sc.heap.empty()) {
            auto [dv,v]=heap_pop(sc.heap);if(dv!=sc.dist[v])continue;
            if(v==st.sigma) {
                if(source_compressed) {
                    for(int u=0;u<st.U;++u) {
                        int e=source_edge[u];if(e<0)continue;
                        ++scans;relax(st,v,st.I+u,st.edge_score(e),e,sc);
                    }
                } else {
                    for(int j=0;j<st.I;++j)if(st.counts[j]>=2) {
                        ++scans;relax(st,v,j,0,-1,sc);
                    }
                }
            } else if(v<st.I) {
                for(int slot=slot_head[v];slot>=0;slot=slot_next[slot]) {
                    ++scans;int u=slot/st.N,e=st.user_list_edges[u][slot%st.N];
                    relax(st,v,st.I+u,st.edge_score(e),e,sc);
                }
            } else if(v<st.sigma)relax_row(v-st.I,false);
        }
        for(int v=0;v<st.nodes;++v)if(std::isfinite(sc.dist[v])) {
            st.potential[v]+=sc.dist[v];active[v]=1;
            if(sc.pred[v]>=0)link(v,sc.pred[v]);
        }
        if(terminal_compressed)build_targets();
        else for(int j=0;j<st.I;++j)target_push(j);
    }
    std::vector<int> augment_state(int terminal) {
        if(!source_compressed) {
            auto affected=LocalRepair::augment(terminal);
            if(st.coverage<st.D)activate(terminal);
            return affected;
        }
        std::vector<int> edges,path;
        for(int v=terminal;v!=st.sigma;v=sc.pred[v]) {
            if(v<0 || path.size()>(size_t)st.nodes)throw std::runtime_error("bad compressed tree path");
            path.push_back(v);edges.push_back(sc.pred_edge[v]);
        }
        if(edges.empty() || edges.size()%2)throw std::runtime_error("bad compressed augmentation");
        if(edges.size()==2)++direct_augmentations;
        int donor=st.edge_item(edges.back()),first_user=st.edge_user(edges.back());
        int count_before=st.counts[donor];
        if(count_before<2 || path.back()!=st.I+first_user)throw std::runtime_error("invalid donor witness");
        std::vector<int> dirty{first_user};
        if(count_before==2) {
            ++source_class_changes;
            for(int slot=slot_head[donor];slot>=0;slot=slot_next[slot])
                if(slot/st.N!=first_user)dirty.push_back(slot/st.N);
        }
        source_dirty_max=std::max<uint64_t>(source_dirty_max,dirty.size());
        std::vector<double> old_cost;std::vector<int> old_witness;
        for(int u:dirty) {
            old_cost.push_back(source_edge[u]<0 ? INF : st.edge_score(source_edge[u]));
            old_witness.push_back(source_edge[u]);
        }
        for(size_t p=0;p<edges.size();p+=2) {
            int add=edges[p],rem=edges[p+1];
            int slot=st.edge_user(rem)*st.N+st.selected_edge_pos[rem];
            slot_unlink(slot,st.edge_item(rem));slot_link(slot,st.edge_item(add));
        }
        int old=st.coverage;apply_path(st,edges);
        if(st.coverage!=old+1)throw std::runtime_error("coverage increment is not one");
        std::vector<int> affected;
        if(st.coverage>=st.D)return affected;
        // Deleted internal SPT arcs lie on the chosen path. Source weight
        // increases can additionally invalidate one off-path source subtree.
        std::vector<int> roots(path.begin(),path.end()-1);
        for(size_t p=0;p<dirty.size();++p) {
            int u=dirty[p],v=st.I+u,e=refresh_source(u);
            double cost=e<0 ? INF : st.edge_score(e);
            if(cost<old_cost[p]-1e-8)throw std::runtime_error("source weight decreased");
            if(cost>old_cost[p])++source_cost_increases;
            if(cost==old_cost[p] && e!=old_witness[p])++source_equal_witness_changes;
            if(sc.pred[v]==st.sigma) {
                if(cost>old_cost[p])roots.push_back(v);
                else ++source_preserved_tree_arcs;
                sc.pred_edge[v]=e; // equal-cost replacement still needs a current witness
            }
        }
        if(count_before==2) {
            if(sc.pred[donor]>=0 || child[donor]>=0)throw std::runtime_error("donor unexpectedly in compressed tree");
            st.potential[donor]=0;active[donor]=1;roots.push_back(donor);
            activate(donor);
        }
        activate(terminal);
        auto add=[&](int v){if(!inside[v]){inside[v]=1;affected.push_back(v);}};
        for(int v:roots)add(v);
        for(size_t p=0;p<affected.size();++p)
            for(int v=child[affected[p]];v>=0;v=next[v])add(v);
        uint64_t components=0;
        for(int v:affected)if(sc.pred[v]<0 || !inside[sc.pred[v]])++components;
        repair_components_max=std::max(repair_components_max,components);
        return affected;
    }
    void repair_state(const std::vector<int>& affected) {
        ++repairs;total_affected+=affected.size();
        max_affected=std::max<uint64_t>(max_affected,affected.size());
        repair_sizes.push_back((int)affected.size());
        for(int v:affected)inside[v]=1;
        for(int v:affected)unlink(v);
        for(int v:affected){sc.dist[v]=INF;sc.pred[v]=sc.pred_edge[v]=-1;}
        sc.heap.clear();
        auto boundary=[&](int a,int b,double c,int e){sc.dist[a]=0;relax(st,a,b,c,e,sc);};
        for(int v:affected) {
            if(v<st.I) {
                if(!core_item(v))throw std::runtime_error("noncore affected item");
                if(boundary_oracle) {
                    int e=boundary_min(v);
                    if(e>=0){++scans;boundary(st.I+st.edge_user(e),v,-st.edge_score(e),e);}
                } else {
                    for(int p=st.item_indptr[v];p<st.item_indptr[v+1];++p) {
                        ++scans;++boundary_scan_checks;int e=st.item_edges[p],a=st.I+st.item_users[p];
                        if(!st.selected_edge[e]&&!inside[a]&&active[a])boundary(a,v,-st.edge_score(st.item_edges[p]),e);
                    }
                }
                if(!source_compressed && st.counts[v]>=2) {++scans;boundary(st.sigma,v,0,-1);}
            } else {
                int u=v-st.I;
                if(source_compressed && source_edge[u]>=0) {
                    int e=source_edge[u];++scans;boundary(st.sigma,v,st.edge_score(e),e);
                }
                for(int e:st.user_list_edges[u]) {
                    ++scans;int a=st.edge_item(e);
                    if(core_item(a)&&!inside[a]&&active[a])boundary(a,v,st.edge_score(e),e);
                }
            }
        }
        while(!sc.heap.empty()) {
            auto [dv,v]=heap_pop(sc.heap);if(dv!=sc.dist[v])continue;
            if(v<st.I) {
                for(int slot=slot_head[v];slot>=0;slot=slot_next[slot]) {
                    ++scans;int u=slot/st.N,b=st.I+u,e=st.user_list_edges[u][slot%st.N];
                    if(inside[b])relax(st,v,b,st.edge_score(e),e,sc);
                }
            } else relax_row(v-st.I,true);
        }
        for(int v:affected) {
            if(std::isfinite(sc.dist[v])) {
                if(v>=st.I && v<st.sigma && sc.dist[v]>0)++positive_user_price_updates;
                st.potential[v]+=sc.dist[v];active[v]=1;link(v,sc.pred[v]);
            } else {active[v]=0;sc.pred[v]=sc.pred_edge[v]=-1;}
            if(!terminal_compressed && v<st.I){++versions[v];target_push(v);}
            inside[v]=0;
        }
    }
};
} // namespace
