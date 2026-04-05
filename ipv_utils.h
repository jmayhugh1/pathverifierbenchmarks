#pragma once

#include "ipv.h"

#include <random>

/// Generate a random query mask where each edge is independently included with
/// probability @p p_query.
Path randomPath(size_t num_edges, std::mt19937 &rng, double p_query = 0.5);
/// Convenience overload that seeds its own PRNG from std::random_device.
Path randomPath(size_t num_edges);

/// Build a random directed graph with @p num_edges distinct (u,v) pairs drawn
/// from [0, num_vertices)^2, each independently hazardous with probability
/// @p hazard_p.
Map randomMap(size_t num_vertices, size_t num_edges, std::mt19937 &rng,
              double hazard_p = 0.5);
/// Convenience overload that seeds its own PRNG from std::random_device.
Map randomMap(size_t num_vertices, size_t num_edges);

/// H(p) = -p log2(p) - (1-p) log2(1-p); returns 0 at the boundaries.
float binary_entropy(float p);

/// Sum of per-edge binary entropies — total uncertainty assuming independence.
float total_entropy(const pMatrix &p);
