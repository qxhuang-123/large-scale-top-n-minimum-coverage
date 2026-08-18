#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

namespace {

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double EPS = 1e-8;

using HeapEntry = std::pair<double, int>;

struct HistoryRow {
    int iteration = 0;
    int diversity = 0;
    double objective = 0.0;
    double elapsed = 0.0;
};

py::list history_to_py(const std::vector<HistoryRow>& history) {
    py::list rows;
    for (const auto& row : history) {
        rows.append(py::dict(
            "iteration"_a = row.iteration,
            "diversity"_a = row.diversity,
            "objective"_a = row.objective,
            "time_sec"_a = row.elapsed
        ));
    }
    return rows;
}

struct DijkstraScratch {
    std::vector<double> dist;
    std::vector<int> pred;
    std::vector<int> pred_edge;
    std::vector<int> path_edges;
    std::vector<HeapEntry> heap;

    void ensure(int nodes, int heap_reserve) {
        if (static_cast<int>(dist.size()) != nodes) {
            dist.resize(nodes);
            pred.resize(nodes);
            pred_edge.resize(nodes);
        }
        if (heap.capacity() < static_cast<size_t>(heap_reserve)) {
            heap.reserve(heap_reserve);
        }
    }

    void reset(int nodes) {
        ensure(nodes, nodes);
        std::fill(dist.begin(), dist.end(), INF);
        std::fill(pred.begin(), pred.end(), -1);
        std::fill(pred_edge.begin(), pred_edge.end(), -1);
        path_edges.clear();
        heap.clear();
    }
};

inline void heap_push(std::vector<HeapEntry>& heap, HeapEntry entry) {
    heap.push_back(entry);
    std::push_heap(heap.begin(), heap.end(), std::greater<HeapEntry>());
}

inline HeapEntry heap_pop(std::vector<HeapEntry>& heap) {
    std::pop_heap(heap.begin(), heap.end(), std::greater<HeapEntry>());
    HeapEntry entry = heap.back();
    heap.pop_back();
    return entry;
}

struct RunState {
    int U = 0;
    int I = 0;
    int E = 0;
    int sigma = 0;
    int tau = 0;
    int nodes = 0;
    int N = 0;
    int D = 0;
    int progress_every = 0;

    const int32_t* user_indptr = nullptr;
    const int32_t* user_items = nullptr;
    const double* user_scores = nullptr;
    const int32_t* edge_users = nullptr;
    const int32_t* item_indptr = nullptr;
    const int32_t* item_users = nullptr;
    const int32_t* item_edges = nullptr;
    const double* item_scores = nullptr;

    std::vector<std::vector<int>> user_list_edges;
    std::vector<uint8_t> selected_edge;
    std::vector<int> selected_edge_pos;
    std::vector<int> counts;
    std::vector<double> potential;
    double initial_score = 0.0;
    double final_score = 0.0;
    int total_recs = 0;
    int naive_D = 0;
    int coverage = 0;
    int augmentations = 0;
    int swaps = 0;

    inline int edge_item(int edge) const { return user_items[edge]; }
    inline double edge_score(int edge) const { return user_scores[edge]; }
    inline int edge_user(int edge) const { return edge_users[edge]; }
};

void build_initial_solution(RunState& st) {
    st.user_list_edges.assign(st.U, {});
    st.selected_edge.assign(st.E, 0);
    st.selected_edge_pos.assign(st.E, -1);
    st.counts.assign(st.I, 0);

    for (int u = 0; u < st.U; ++u) {
        const int begin = st.user_indptr[u];
        const int end = st.user_indptr[u + 1];
        const int len = end - begin;
        if (len <= 0) {
            continue;
        }
        const int k = std::min(st.N, len);
        std::vector<int> edges;
        edges.reserve(len);
        for (int e = begin; e < end; ++e) {
            edges.push_back(e);
        }
        auto better = [&](int a, int b) { return st.edge_score(a) > st.edge_score(b); };
        if (k < len) {
            std::nth_element(edges.begin(), edges.begin() + k, edges.end(), better);
            edges.resize(k);
        }
        std::sort(edges.begin(), edges.end(), better);

        st.user_list_edges[u] = edges;
        for (int pos = 0; pos < static_cast<int>(edges.size()); ++pos) {
            const int e = edges[pos];
            const int item = st.edge_item(e);
            st.selected_edge[e] = 1;
            st.selected_edge_pos[e] = pos;
            st.counts[item] += 1;
            st.initial_score += st.edge_score(e);
            st.total_recs += 1;
        }
    }

    st.final_score = st.initial_score;
    st.naive_D = 0;
    for (int c : st.counts) {
        if (c > 0) {
            st.naive_D += 1;
        }
    }
    st.coverage = st.naive_D;
}

void initialize_potentials(RunState& st) {
    st.potential.assign(st.nodes, 0.0);
    for (int u = 0; u < st.U; ++u) {
        const auto& lst = st.user_list_edges[u];
        if (lst.empty()) {
            continue;
        }
        double min_selected = INF;
        for (int e : lst) {
            min_selected = std::min(min_selected, st.edge_score(e));
        }
        st.potential[st.I + u] = -min_selected;
    }
}

void repair_potentials(RunState& st) {
    std::vector<double> dist(st.nodes, INF);
    std::vector<uint8_t> in_queue(st.nodes, 0);
    std::vector<int> relax_count(st.nodes, 0);
    std::deque<int> q;

    dist[st.sigma] = 0.0;
    q.push_back(st.sigma);
    in_queue[st.sigma] = 1;

    auto raw_relax = [&](int v, int to, double raw_cost) {
        const double nd = dist[v] + raw_cost;
        if (nd + 1e-12 < dist[to]) {
            dist[to] = nd;
            if (!in_queue[to]) {
                q.push_back(to);
                in_queue[to] = 1;
                relax_count[to] += 1;
                if (relax_count[to] > st.nodes) {
                    throw std::runtime_error("negative cycle detected in the exchange graph");
                }
            }
        }
    };

    while (!q.empty()) {
        const int v = q.front();
        q.pop_front();
        in_queue[v] = 0;

        if (v == st.sigma) {
            for (int j = 0; j < st.I; ++j) {
                if (st.counts[j] == 0) {
                    raw_relax(v, j, 0.0);
                }
            }
        } else if (v < st.I) {
            const int item = v;
            if (st.counts[item] >= 2) {
                raw_relax(v, st.tau, 0.0);
            }
            for (int p = st.item_indptr[item]; p < st.item_indptr[item + 1]; ++p) {
                const int e = st.item_edges[p];
                if (!st.selected_edge[e]) {
                    const int u = st.item_users[p];
                    raw_relax(v, st.I + u, -st.item_scores[p]);
                }
            }
        } else if (v < st.sigma) {
            const int u = v - st.I;
            for (int e : st.user_list_edges[u]) {
                raw_relax(v, st.edge_item(e), st.edge_score(e));
            }
        }
    }

    double max_finite = 0.0;
    for (double d : dist) {
        if (std::isfinite(d)) {
            max_finite = std::max(max_finite, d);
        }
    }
    st.potential.assign(st.nodes, max_finite);
    for (int v = 0; v < st.nodes; ++v) {
        if (std::isfinite(dist[v])) {
            st.potential[v] = dist[v];
        }
    }
}

inline void relax(
    const RunState& st,
    int v,
    int to,
    double raw_cost,
    int edge_id,
    DijkstraScratch& sc
) {
    double reduced = raw_cost + st.potential[v] - st.potential[to];
    if (reduced < -EPS) {
        throw std::runtime_error(
            "negative reduced cost " + std::to_string(reduced) +
            " on edge " + std::to_string(v) + "->" + std::to_string(to)
        );
    }
    if (reduced < 0.0) {
        reduced = 0.0;
    }
    const double nd = sc.dist[v] + reduced;
    if (nd + 1e-12 < sc.dist[to]) {
        sc.dist[to] = nd;
        sc.pred[to] = v;
        sc.pred_edge[to] = edge_id;
        heap_push(sc.heap, HeapEntry(nd, to));
    }
}

std::vector<int> shortest_path(RunState& st, DijkstraScratch& sc) {
    sc.reset(st.nodes);

    sc.dist[st.sigma] = 0.0;
    heap_push(sc.heap, HeapEntry(0.0, st.sigma));

    while (!sc.heap.empty()) {
        const auto [dv, v] = heap_pop(sc.heap);
        if (dv != sc.dist[v]) {
            continue;
        }
        if (v == st.tau) {
            break;
        }

        if (v == st.sigma) {
            for (int j = 0; j < st.I; ++j) {
                if (st.counts[j] == 0) {
                    relax(st, v, j, 0.0, -1, sc);
                }
            }
        } else if (v < st.I) {
            const int item = v;
            if (st.counts[item] >= 2) {
                relax(st, v, st.tau, 0.0, -1, sc);
            }
            for (int p = st.item_indptr[item]; p < st.item_indptr[item + 1]; ++p) {
                const int e = st.item_edges[p];
                if (!st.selected_edge[e]) {
                    const int u = st.item_users[p];
                    relax(st, v, st.I + u, -st.item_scores[p], e, sc);
                }
            }
        } else if (v < st.sigma) {
            const int u = v - st.I;
            for (int e : st.user_list_edges[u]) {
                relax(st, v, st.edge_item(e), st.edge_score(e), e, sc);
            }
        }
    }

    if (!std::isfinite(sc.dist[st.tau])) {
        return {};
    }

    sc.path_edges.clear();
    for (int cur = st.tau; cur != -1 && cur != st.sigma; cur = sc.pred[cur]) {
        const int pe = sc.pred_edge[cur];
        if (pe >= 0) {
            sc.path_edges.push_back(pe);
        }
    }
    std::reverse(sc.path_edges.begin(), sc.path_edges.end());
    if (sc.path_edges.size() % 2 != 0) {
        throw std::runtime_error("bad augmenting path edge count");
    }
    return sc.path_edges;
}

void apply_path(RunState& st, const std::vector<int>& path_edges) {
    for (size_t idx = 0; idx + 1 < path_edges.size(); idx += 2) {
        const int add_edge = path_edges[idx];
        const int remove_edge = path_edges[idx + 1];
        const int u = st.edge_user(add_edge);
        if (u < 0 || u >= st.U || remove_edge < st.user_indptr[u] || remove_edge >= st.user_indptr[u + 1]) {
            throw std::runtime_error("bad exchange edge path");
        }
        if (st.selected_edge[add_edge] || !st.selected_edge[remove_edge]) {
            throw std::runtime_error("inconsistent selected state in exchange path");
        }

        const int add_item = st.edge_item(add_edge);
        const int remove_item = st.edge_item(remove_edge);
        st.selected_edge[add_edge] = 1;
        st.selected_edge[remove_edge] = 0;
        if (st.counts[add_item] == 0) {
            st.coverage += 1;
        }
        st.counts[add_item] += 1;
        st.counts[remove_item] -= 1;
        if (st.counts[remove_item] == 0) {
            st.coverage -= 1;
        }
        st.final_score += st.edge_score(add_edge) - st.edge_score(remove_edge);

        const int pos = st.selected_edge_pos[remove_edge];
        if (pos < 0 || pos >= static_cast<int>(st.user_list_edges[u].size()) || st.user_list_edges[u][pos] != remove_edge) {
            throw std::runtime_error("removed edge not found in user list");
        }
        st.user_list_edges[u][pos] = add_edge;
        st.selected_edge_pos[add_edge] = pos;
        st.selected_edge_pos[remove_edge] = -1;
        st.swaps += 1;
    }

    st.augmentations += 1;
}

py::dict run_exact_csr_impl(
    int num_users,
    int num_items,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> user_indptr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> user_items,
    py::array_t<double, py::array::c_style | py::array::forcecast> user_scores,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> edge_users,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> item_indptr,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> item_users,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> item_edges,
    py::array_t<double, py::array::c_style | py::array::forcecast> item_scores,
    int N,
    int D,
    int progress_every
) {
    auto ui = user_indptr.request();
    auto ux = user_items.request();
    auto us = user_scores.request();
    auto eu = edge_users.request();
    auto ii = item_indptr.request();
    auto iu = item_users.request();
    auto ie = item_edges.request();
    auto is = item_scores.request();
    if (ui.ndim != 1 || ux.ndim != 1 || us.ndim != 1 || eu.ndim != 1 ||
        ii.ndim != 1 || iu.ndim != 1 || ie.ndim != 1 || is.ndim != 1) {
        throw std::runtime_error("all CSR arrays must be 1-D");
    }
    if (ui.shape[0] != num_users + 1 || ii.shape[0] != num_items + 1 ||
        ux.shape[0] != us.shape[0] || ux.shape[0] != eu.shape[0] ||
        iu.shape[0] != ie.shape[0] || iu.shape[0] != is.shape[0]) {
        throw std::runtime_error("bad CSR array shapes");
    }
    if (N < 0 || D < 0 || D > num_items) {
        throw std::runtime_error("bad N or D");
    }

    RunState st;
    st.U = num_users;
    st.I = num_items;
    st.E = static_cast<int>(ux.shape[0]);
    st.sigma = st.I + st.U;
    st.tau = st.sigma + 1;
    st.nodes = st.tau + 1;
    st.N = N;
    st.D = D;
    st.progress_every = progress_every;
    st.user_indptr = static_cast<const int32_t*>(ui.ptr);
    st.user_items = static_cast<const int32_t*>(ux.ptr);
    st.user_scores = static_cast<const double*>(us.ptr);
    st.edge_users = static_cast<const int32_t*>(eu.ptr);
    st.item_indptr = static_cast<const int32_t*>(ii.ptr);
    st.item_users = static_cast<const int32_t*>(iu.ptr);
    st.item_edges = static_cast<const int32_t*>(ie.ptr);
    st.item_scores = static_cast<const double*>(is.ptr);

    auto run_start = std::chrono::steady_clock::now();
    std::vector<HistoryRow> history;

    build_initial_solution(st);
    const double naive_pred = st.total_recs > 0 ? st.initial_score / st.total_recs : 0.0;
    auto after_initial = std::chrono::steady_clock::now();
    history.push_back(HistoryRow{
        0,
        st.coverage,
        st.final_score,
        std::chrono::duration<double>(after_initial - run_start).count()
    });

    if (D <= st.coverage) {
        return py::dict(
            "naive_pred"_a = naive_pred,
            "tpcar_pred"_a = naive_pred,
            "naive_obj"_a = st.initial_score,
            "tpcar_obj"_a = st.final_score,
            "naive_D"_a = st.naive_D,
            "tpcar_D"_a = st.coverage,
            "target_feasible"_a = true,
            "max_reached"_a = st.coverage,
            "augmentations"_a = 0,
            "swaps"_a = 0,
            "total_recs"_a = st.total_recs,
            "history"_a = history_to_py(history)
        );
    }

    initialize_potentials(st);
    DijkstraScratch scratch;
    scratch.ensure(st.nodes, st.nodes);
    auto start = std::chrono::steady_clock::now();

    while (st.coverage < D) {
        if (progress_every > 0 && st.augmentations > 0 && st.augmentations % progress_every == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            py::print(
                "    augmentation", st.augmentations, "/", D - st.naive_D,
                "D=", st.coverage, "swaps=", st.swaps, "elapsed=", elapsed
            );
        }

        std::vector<int> path_edges;
        bool ok = false;
        for (int attempt = 0; attempt < 2; ++attempt) {
            try {
                path_edges = shortest_path(st, scratch);
                ok = true;
                break;
            } catch (const std::runtime_error& e) {
                const std::string msg = e.what();
                if (attempt == 1 || msg.find("negative reduced cost") == std::string::npos) {
                    throw;
                }
                repair_potentials(st);
            }
        }
        if (!ok) {
            throw std::runtime_error("failed to find augmenting path");
        }
        if (path_edges.empty()) {
            py::print(
                "    No augmenting path remains. Maximum reachable coverage=", st.coverage,
                ", target D=", D
            );
            break;
        }

        double max_finite = 0.0;
        for (int v = 0; v < st.nodes; ++v) {
            if (std::isfinite(scratch.dist[v])) {
                max_finite = std::max(max_finite, scratch.dist[v]);
            }
        }
        for (int v = 0; v < st.nodes; ++v) {
            st.potential[v] += std::isfinite(scratch.dist[v]) ? scratch.dist[v] : max_finite;
        }
        apply_path(st, path_edges);

        auto now = std::chrono::steady_clock::now();
        history.push_back(HistoryRow{
            st.augmentations,
            st.coverage,
            st.final_score,
            std::chrono::duration<double>(now - run_start).count()
        });
    }

    const double tpcar_pred = st.total_recs > 0 ? st.final_score / st.total_recs : 0.0;
    return py::dict(
        "naive_pred"_a = naive_pred,
        "tpcar_pred"_a = tpcar_pred,
        "naive_obj"_a = st.initial_score,
        "tpcar_obj"_a = st.final_score,
        "naive_D"_a = st.naive_D,
        "tpcar_D"_a = st.coverage,
        "target_feasible"_a = st.coverage >= D,
        "max_reached"_a = st.coverage,
        "augmentations"_a = st.augmentations,
        "swaps"_a = st.swaps,
        "total_recs"_a = st.total_recs,
        "history"_a = history_to_py(history)
    );
}

}  // namespace

#ifndef TPCAR_MODULE_NAME
#define TPCAR_MODULE_NAME tpcar_core
#endif

PYBIND11_MODULE(TPCAR_MODULE_NAME, m) {
    m.doc() = "CSR-only C++ implicit-Dijkstra core for exact TPCAR.";
    m.def(
        "run_exact_csr",
        &run_exact_csr_impl,
        "num_users"_a,
        "num_items"_a,
        "user_indptr"_a,
        "user_items"_a,
        "user_scores"_a,
        "edge_users"_a,
        "item_indptr"_a,
        "item_users"_a,
        "item_edges"_a,
        "item_scores"_a,
        "N"_a,
        "D"_a,
        "progress_every"_a = 200
    );
    // Public Top-N initializer: the same exact C++ path, with D=0.
    m.def(
        "run_initial_csr",
        &run_exact_csr_impl,
        "num_users"_a,
        "num_items"_a,
        "user_indptr"_a,
        "user_items"_a,
        "user_scores"_a,
        "edge_users"_a,
        "item_indptr"_a,
        "item_users"_a,
        "item_edges"_a,
        "item_scores"_a,
        "N"_a,
        "D"_a = 0,
        "progress_every"_a = 0
    );
}

