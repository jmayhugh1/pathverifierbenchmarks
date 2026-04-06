#include "ipv_utils.h"
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

} // namespace ipv_utils