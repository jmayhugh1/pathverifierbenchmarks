#pragma once

#include "ipv.h"

#include <random>

namespace ipv_utils {
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
double binary_entropy(double p);

/// Sum of per-edge binary entropies — total uncertainty assuming independence.
double total_entropy(const pMatrix &p);

/// Convert probability in (0,1) to log-odds log(p/(1-p)).
/// Returns ±inf at the boundaries.
double probToLogOdds(double p);

/// Convert log-odds back to probability: 1/(1+exp(-l)).
double logOddsToProb(double l);

/// Convert a vector of log-odds to probabilities.
pMatrix logOddsToProbs(const std::vector<double> &logodds);

/// Convert a vector of probabilities to log-odds.
std::vector<double> probsToLogOdds(const pMatrix &probs);

/// Numerically stable softplus: log(1 + exp(x)).
double softplus(double x);

/// Binary entropy in bits computed directly from log-odds,
/// avoiding intermediate probability conversion.
double binary_entropy_logodds(double l);

/// Sum of per-edge binary entropies from a vector of log-odds.
double total_entropy_logodds(const std::vector<double> &logodds);

/// Numerically stable log-sum-exp: log(Σ exp(x_i)).
double logsumexp(const std::vector<double> &x);

/// log(1 - exp(a)) for a <= 0, numerically stable.
double log1mexp(double a);

} // namespace ipv_utils