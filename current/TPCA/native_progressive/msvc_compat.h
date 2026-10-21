#pragma once

// Native Windows equivalents for GCC builtins used by the optional bitset code.
// The state_source (F) algorithm and its search/repair rules are unchanged.
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
inline int trmnl_ctz64(unsigned __int64 value) {
    unsigned long index;
    return _BitScanForward64(&index, value) ? static_cast<int>(index) : 64;
}
#define __builtin_popcountll(value) static_cast<int>(__popcnt64(static_cast<unsigned __int64>(value)))
#define __builtin_ctzll(value) trmnl_ctz64(static_cast<unsigned __int64>(value))
#endif
