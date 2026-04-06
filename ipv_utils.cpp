#include "ipv_utils.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ipv_utils {

Path randomPath(size_t num_edges, std::mt19937 &rng, double p_query) {
  if (p_query < 0.0 || p_query > 1.0) {
    throw std::invalid_argument("p_query must be in [0,1].");
  }
  std::bernoulli_distribution pick(p_query);
  Path path(num_edges);
  for (size_t i = 0; i < num_edges; ++i) {
    path[i] = pick(rng);
  }
  return path;
}

Path randomPath(size_t num_edges) {
  std::random_device rd;
  std::mt19937 rng(rd());
  return randomPath(num_edges, rng);
}

Map randomMap(size_t num_vertices, size_t num_edges, std::mt19937 &rng,
              double hazard_p) {
  if (hazard_p < 0.0 || hazard_p > 1.0) {
    throw std::invalid_argument("hazard_p must be in [0,1].");
  }
  const size_t cap = num_vertices * num_vertices;
  if (num_edges > cap) {
    throw std::invalid_argument(
        "num_edges exceeds num_vertices^2 distinct directed edges.");
  }
  std::vector<std::pair<size_t, size_t>> pairs;
  pairs.reserve(cap);
  for (size_t u = 0; u < num_vertices; ++u) {
    for (size_t v = 0; v < num_vertices; ++v) {
      pairs.emplace_back(u, v);
    }
  }
  std::shuffle(pairs.begin(), pairs.end(), rng);
  std::bernoulli_distribution haz(hazard_p);
  Map map;
  map.reserve(num_edges);
  for (size_t i = 0; i < num_edges; ++i) {
    const auto [u, v] = pairs[i];
    map.push_back({u, v, static_cast<bool>(haz(rng))});
  }
  return map;
}

Map randomMap(size_t num_vertices, size_t num_edges) {
  std::random_device rd;
  std::mt19937 rng(rd());
  return randomMap(num_vertices, num_edges, rng);
}

double binary_entropy(double p) {
  if (p <= 0.f || p >= 1.f) {
    return 0.f;
  }
  return -(p * std::log2(p) + (1.f - p) * std::log2(1.f - p));
}

double total_entropy(const pMatrix &p) {
  double sum = 0.f;
  for (double x : p) {
    sum += binary_entropy(x);
  }
  return sum;
}

double probToLogOdds(double p) {
  if (p <= 0.0) return -std::numeric_limits<double>::infinity();
  if (p >= 1.0) return std::numeric_limits<double>::infinity();
  return std::log(p / (1.0 - p));
}

double logOddsToProb(double l) {
  if (std::isinf(l) && l < 0) return 0.0;
  if (std::isinf(l) && l > 0) return 1.0;
  return 1.0 / (1.0 + std::exp(-l));
}

pMatrix logOddsToProbs(const std::vector<double> &logodds) {
  pMatrix probs(logodds.size());
  for (size_t i = 0; i < logodds.size(); ++i) {
    probs[i] = logOddsToProb(logodds[i]);
  }
  return probs;
}

std::vector<double> probsToLogOdds(const pMatrix &probs) {
  std::vector<double> logodds(probs.size());
  for (size_t i = 0; i < probs.size(); ++i) {
    logodds[i] = probToLogOdds(probs[i]);
  }
  return logodds;
}

double softplus(double x) {
  if (x > 20.0) return x;
  if (x < -20.0) return std::exp(x);
  return std::log1p(std::exp(x));
}

double binary_entropy_logodds(double l) {
  if (std::isinf(l)) return 0.0;
  // H(sigmoid(l)) = sigmoid(l)*softplus(-l) + sigmoid(-l)*softplus(l)
  // all in nats, then convert to bits.
  const double sp_pos = softplus(l);
  const double sp_neg = softplus(-l);
  const double sig_pos = 1.0 / (1.0 + std::exp(-l));
  const double sig_neg = 1.0 - sig_pos;
  return (sig_pos * sp_neg + sig_neg * sp_pos) / std::log(2.0);
}

double total_entropy_logodds(const std::vector<double> &logodds) {
  double sum = 0.0;
  for (double l : logodds) {
    sum += binary_entropy_logodds(l);
  }
  return sum;
}

double logsumexp(const std::vector<double> &x) {
  if (x.empty()) return -std::numeric_limits<double>::infinity();
  const double m = *std::max_element(x.begin(), x.end());
  if (std::isinf(m) && m < 0) return -std::numeric_limits<double>::infinity();
  double sum = 0.0;
  for (double v : x) {
    sum += std::exp(v - m);
  }
  return m + std::log(sum);
}

double log1mexp(double a) {
  if (a >= 0.0) return std::numeric_limits<double>::quiet_NaN();
  if (a > -std::log(2.0)) return std::log(-std::expm1(a));
  return std::log1p(-std::exp(a));
}

} // namespace ipv_utils