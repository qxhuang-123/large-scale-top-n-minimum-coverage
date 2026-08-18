#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

struct HistoryRow { int iteration; int diversity; double objective; double elapsed; };

py::list history_to_py(const std::vector<HistoryRow>& history) {
    py::list result;
    for (const auto& row : history) {
        result.append(py::dict("iteration"_a=row.iteration, "diversity"_a=row.diversity,
                               "objective"_a=row.objective, "time_sec"_a=row.elapsed));
    }
    return result;
}

struct State {
    int U, I, N;
    const int32_t* user_indptr; const int32_t* user_items; const float* user_scores;
    const int32_t* item_indptr; const int32_t* item_users; const float* item_scores;
    std::vector<std::vector<int>> selected_items;
    std::vector<std::vector<double>> selected_scores;
    std::vector<int> counts;
    double score = 0.0;
    int total_recs = 0, diversity = 0, swaps = 0;
};

void build_initial(State& st) {
    st.selected_items.assign(st.U, std::vector<int>(st.N, -1));
    st.selected_scores.assign(st.U, std::vector<double>(st.N, 0.0));
    st.counts.assign(st.I, 0);
    for (int u=0; u<st.U; ++u) {
        const int begin=st.user_indptr[u], end=st.user_indptr[u+1], len=end-begin;
        const int k=std::min(st.N, len);
        if (!k) continue;
        std::vector<int> edges; edges.reserve(len);
        for(int e=begin;e<end;++e) edges.push_back(e);
        auto better=[&](int a,int b) { double sa=st.user_scores[a], sb=st.user_scores[b];
            return sa != sb ? sa > sb : st.user_items[a] < st.user_items[b]; };
        if(k<len) { std::nth_element(edges.begin(), edges.begin()+k, edges.end(), better); edges.resize(k); }
        std::sort(edges.begin(), edges.end(), better);
        for(int pos=0;pos<k;++pos) { int e=edges[pos], item=st.user_items[e];
            st.selected_items[u][pos]=item; st.selected_scores[u][pos]=st.user_scores[e];
            ++st.counts[item]; st.score+=st.user_scores[e]; ++st.total_recs;
        }
    }
    for(int c:st.counts) if(c>0) ++st.diversity;
}

bool best_swap_for_item(const State& st, int item, double& best_loss, int& best_u, int& best_pos, double& best_add, int& best_remove) {
    best_loss=std::numeric_limits<double>::infinity(); best_u=-1; best_pos=-1; best_add=0.0; best_remove=-1;
    for(int p=st.item_indptr[item]; p<st.item_indptr[item+1]; ++p) {
        int u=st.item_users[p]; double add=st.item_scores[p]; bool already=false;
        for(int pos=0;pos<st.N;++pos) if(st.selected_items[u][pos]==item) {already=true; break;}
        if(already) continue;
        for(int pos=0;pos<st.N;++pos) { int remove=st.selected_items[u][pos];
            if(remove<0 || st.counts[remove]<2) continue;
            double loss=st.selected_scores[u][pos]-add;
            if(loss<best_loss) { best_loss=loss; best_u=u; best_pos=pos; best_add=add; best_remove=remove; }
        }
    }
    return best_u>=0;
}

py::dict run_greedy_csr_impl(int U, int I,
    py::array_t<int32_t, py::array::c_style|py::array::forcecast> user_indptr,
    py::array_t<int32_t, py::array::c_style|py::array::forcecast> user_items,
    py::array_t<float, py::array::c_style|py::array::forcecast> user_scores,
    py::array_t<int32_t, py::array::c_style|py::array::forcecast> item_indptr,
    py::array_t<int32_t, py::array::c_style|py::array::forcecast> item_users,
    py::array_t<float, py::array::c_style|py::array::forcecast> item_scores,
    int N, int D, int history_interval, int max_rounds) {
    auto ui=user_indptr.request(), ux=user_items.request(), us=user_scores.request();
    auto ii=item_indptr.request(), iu=item_users.request(), is=item_scores.request();
    if(ui.ndim!=1||ux.ndim!=1||us.ndim!=1||ii.ndim!=1||iu.ndim!=1||is.ndim!=1 || ui.shape[0]!=U+1 || ii.shape[0]!=I+1 || ux.shape[0]!=us.shape[0] || iu.shape[0]!=is.shape[0]) throw std::runtime_error("bad CSR arrays");
    if(N<=0 || D<0 || D>I || history_interval<=0 || max_rounds<=0) throw std::runtime_error("bad parameters");
    State st{U,I,N,static_cast<const int32_t*>(ui.ptr),static_cast<const int32_t*>(ux.ptr),static_cast<const float*>(us.ptr),static_cast<const int32_t*>(ii.ptr),static_cast<const int32_t*>(iu.ptr),static_cast<const float*>(is.ptr)};
    auto start=std::chrono::steady_clock::now(); build_initial(st);
    const double naive_score=st.score; const int naive_D=st.diversity; const int total_recs=st.total_recs;
    std::vector<HistoryRow> history{{0, st.diversity, st.score, std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
    int rounds=0;
    while(rounds<max_rounds && st.diversity<D) {
        ++rounds; std::vector<std::pair<double,int>> order; order.reserve(I);
        for(int item=0;item<I;++item) if(st.counts[item]==0) { double loss,add; int u,pos,remove; if(best_swap_for_item(st,item,loss,u,pos,add,remove)) order.emplace_back(loss,item); }
        std::sort(order.begin(),order.end(),[](const auto& a,const auto& b){return a.first!=b.first ? a.first<b.first : a.second<b.second;});
        int changed=0;
        for(const auto& pair:order) { int item=pair.second; if(st.counts[item]>0) continue; double loss,add; int u,pos,remove;
            if(!best_swap_for_item(st,item,loss,u,pos,add,remove) || st.counts[remove]<2) continue;
            ++st.counts[item]; --st.counts[remove]; st.score += add-st.selected_scores[u][pos]; st.selected_items[u][pos]=item; st.selected_scores[u][pos]=add; ++st.diversity; ++st.swaps; ++changed;
            if(history_interval<=1 || st.swaps%history_interval==0) history.push_back({st.swaps,st.diversity,st.score,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()});
            if(st.diversity>=D) break;
        }
        if(!changed) break;
    }
    if(history.back().iteration!=st.swaps) history.push_back({st.swaps,st.diversity,st.score,std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()});
    return py::dict("naive_obj"_a=naive_score, "naive_pred"_a=(total_recs?naive_score/total_recs:0.0), "naive_D"_a=naive_D, "tpcar_obj"_a=st.score, "tpcar_pred"_a=(total_recs?st.score/total_recs:0.0), "tpcar_D"_a=st.diversity, "swaps"_a=st.swaps, "rounds"_a=rounds, "total_recs"_a=total_recs, "target_feasible"_a=(st.diversity>=D), "max_reached"_a=st.diversity, "history"_a=history_to_py(history));
}
}

PYBIND11_MODULE(yelp_greedy_core, m) { m.doc()="C++ batch greedy TPCAR core"; m.def("run_greedy_csr", &run_greedy_csr_impl, "num_users"_a,"num_items"_a,"user_indptr"_a,"user_items"_a,"user_scores"_a,"item_indptr"_a,"item_users"_a,"item_scores"_a,"N"_a,"D"_a,"history_interval"_a=50,"max_rounds"_a=1000); }
