#include "ipv.h"
#include "ipv_utils.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

approximateIpv::approximateIpv(Map map, float prior) : ipv(std::move(map)) {
  pmat.assign(num_edges, prior);
  confirmed_safe_.assign(num_edges, false);
}

std::tuple<bool, float> approximateIpv::informationGain(Path path) {
  assert(path.size() == num_edges);

  for (size_t i = 0; i < num_edges; ++i) {
    if (confirmed_safe_[i]) {
      pmat[i] = 0.f;
    }
  }

  const float h_before = total_entropy(pmat);

  const bool safe = !collision(path);

  float p_all_safe = 1.f;
  for (size_t i = 0; i < num_edges; ++i) {
    if (confirmed_safe_[i]) {
      continue;
    }
    const float m = path[i] ? 1.f : 0.f;
    p_all_safe *= (1.f - m * pmat[i]);
  }
  const float p_unsafe = 1.f - p_all_safe;

  float alpha = 0.f;
  if (!safe) {
    alpha = 1.f / std::max(p_unsafe, ipv::eps);
  }

  for (size_t i = 0; i < num_edges; ++i) {
    if (confirmed_safe_[i]) {
      continue;
    }
    const float m = path[i] ? 1.f : 0.f;
    float pi = (1.f - m) * pmat[i] + m * (alpha * pmat[i]);
    pmat[i] = std::clamp(pi, 0.f, 1.f);
  }

  if (safe) {
    for (size_t i = 0; i < num_edges; ++i) {
      if (path[i]) {
        confirmed_safe_[i] = true;
        pmat[i] = 0.f;
      }
    }
  }

  const float h_after = total_entropy(pmat);
  const float ig = h_before - h_after;
  return {safe, ig};
}

// --- exactIpv (full joint bitmask posterior) ---

double exactIpv::entropyBits(const std::vector<double> &dist) {
  double h = 0.0;
  for (double p : dist) {
    if (p > 0.0) {
      h -= p * std::log2(p);
    }
  }
  return h;
}

double exactIpv::binaryEntropy(double p) {
  if (p <= 0.0 || p >= 1.0) {
    return 0.0;
  }
  return -p * std::log2(p) - (1.0 - p) * std::log2(1.0 - p);
}

uint64_t exactIpv::pathToMask(const Path &path) const {
  if (path.size() != edges_) {
    throw std::invalid_argument("Path length must equal number of edges.");
  }
  uint64_t mask = 0;
  for (size_t i = 0; i < edges_; ++i) {
    if (path[i]) {
      mask |= (uint64_t{1} << i);
    }
  }
  return mask;
}

bool exactIpv::consistent(uint64_t state, uint64_t query_mask,
                          bool observed_collision) {
  const bool any_hazard_on_path = (state & query_mask) != 0;
  return observed_collision ? any_hazard_on_path : !any_hazard_on_path;
}

void exactIpv::normalize() {
  const double z = std::accumulate(posterior_.begin(), posterior_.end(), 0.0);
  if (z <= kEps) {
    throw std::runtime_error(
        "Posterior became zero: observations are inconsistent.");
  }
  for (double &w : posterior_) {
    w /= z;
  }
}

double exactIpv::predictiveCollisionProbMask(uint64_t query_mask) const {
  double p = 0.0;
  for (uint64_t state = 0; state < posterior_.size(); ++state) {
    if ((state & query_mask) != 0) {
      p += posterior_[state];
    }
  }
  return p;
}

void exactIpv::observeMask(uint64_t query_mask, bool observed_collision) {
  for (uint64_t state = 0; state < posterior_.size(); ++state) {
    if (!consistent(state, query_mask, observed_collision)) {
      posterior_[state] = 0.0;
    }
  }
  normalize();
}

exactIpv::exactIpv(Map map, const pMatrix &priors)
    : ipv(std::move(map)), prior_edge_prob_(priors.begin(), priors.end()),
      edges_(num_edges) {

  if (priors.size() != edges_) {
    throw std::invalid_argument("Need one prior probability per edge.");
  }
  if (edges_ >= 63) {
    throw std::invalid_argument(
        "This exact bitmask version supports at most 62 edges.");
  }
  for (double p : prior_edge_prob_) {
    if (p < 0.0 || p > 1.0) {
      throw std::invalid_argument("Each prior must lie in [0,1].");
    }
  }

  const uint64_t num_states = uint64_t{1} << edges_;
  posterior_.assign(num_states, 0.0);

  for (uint64_t state = 0; state < num_states; ++state) {
    double w = 1.0;
    for (size_t i = 0; i < edges_; ++i) {
      const bool hazardous = ((state >> i) & 1ULL) != 0;
      const double p = prior_edge_prob_[i];
      w *= hazardous ? p : (1.0 - p);
    }
    posterior_[state] = w;
  }
  normalize();
}

void exactIpv::observe(const Path &path, bool observed_collision) {
  observeMask(pathToMask(path), observed_collision);
}

double exactIpv::predictiveCollisionProb(const Path &path) const {
  return predictiveCollisionProbMask(pathToMask(path));
}

double exactIpv::expectedInformationGain(const Path &path) const {
  return binaryEntropy(predictiveCollisionProb(path));
}

std::tuple<bool, float> exactIpv::informationGain(Path path) {
  std::vector<double> prior_marginals = this->marginals();
  observe(path, collision(path));
  const std::vector<double> marginals = this->marginals();
  const bool safe = !collision(path);
  float h_before = 0.0;
  float h_after = 0.0;
  for (size_t i = 0; i < edges_; ++i) {
    h_before += binaryEntropy(prior_marginals[i]);
    h_after += binaryEntropy(marginals[i]);
  }
  const float realized = h_before - h_after;
  return {safe, realized};
}

std::vector<double> exactIpv::marginals() const {
  std::vector<double> m(edges_, 0.0);
  for (uint64_t state = 0; state < posterior_.size(); ++state) {
    const double w = posterior_[state];
    if (w <= 0.0) {
      continue;
    }
    for (size_t i = 0; i < edges_; ++i) {
      if (((state >> i) & 1ULL) != 0) {
        m[i] += w;
      }
    }
  }
  return m;
}
