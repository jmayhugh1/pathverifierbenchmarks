#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

// (u, v, ground_truth_obstructed) per edge slot
using Edge = std::tuple<size_t, size_t, bool>;

// Ordered list of edges (path mask indices align with this order).
using Map = std::vector<Edge>;

// Mask over edges: path[i] == true means edge i is queried this round.
using Path = std::vector<bool>;

// One prior hazard probability per edge (used by approximate / exact
// constructors).
using pMatrix = std::vector<double>;

class Graph {
  Map map;
  int num_nodes;
  int num_edges;
  std::vector<std::tuple<size_t, size_t>> edge_pairs;
  inline static const std::string GRAPH_CONFIGS_PATH = "configs/graphs.json";
  static const nlohmann::json &getGraphConfigs();

public:
  Graph(std::string graph_name);
  Graph(std::string name, int num_nodes, int num_edges,
        const std::vector<std::tuple<size_t, size_t>> &edge_pairs);
  const Map &getMap() const { return map; }
  void randomlyAssignHazards(double hazard_probability);
  Path randomPath(std::mt19937 &rng, double p_query = 0.5);
  Path randomPath(double p_query = 0.5);
};

class ipv {
private:
  Map map;

protected:
  /// Number of edges |E| (path length); not |V|.
  size_t num_edges;
  static constexpr double eps = 1e-12f;

  explicit ipv(Map map) : map(std::move(map)) { num_edges = this->map.size(); }

public:
  virtual ~ipv() = default;
  /// Returns (safe, information_gain_bits): safe iff no queried edge is
  /// obstructed.
  virtual std::tuple<bool, double> informationGain(Path path) = 0;

  /// True iff some queried edge is obstructed in the hidden map (∨_i m_i ∧
  /// Z_i).
  bool collision(Path path) const {
    assert(path.size() == num_edges);
    for (size_t i = 0; i < num_edges; i++) {
      if (path[i] && std::get<2>(map[i])) {
        return true;
      }
    }
    return false;
  }
};

class approximateIpv : public ipv {
  pMatrix pmat;
  /// Edge was on a path observed safe; hazard belief stays 0 permanently.
  std::vector<bool> confirmed_safe_;

public:
  /// Construct with a ground-truth map and a uniform prior hazard probability
  /// per edge (default 0.5).
  explicit approximateIpv(Map map, double prior = 0.5f);

  /// Condition the posterior on a traversal outcome (collision or safe).
  void observe(Path path, bool observed_collision);

  /// Simulate traversing @p path against the hidden map.  On collision the
  /// queried-edge beliefs are scaled up (Bayesian-like conditioning on at
  /// least one hazard); on safe passage the queried edges are zeroed and
  /// permanently locked.  Returns (safe, realized_information_gain_bits).
  std::tuple<bool, double> informationGain(Path path) override;
  /// Current marginal hazard beliefs P(Z_i=1) per edge (updated by
  /// informationGain).
  std::vector<double> marginals() const { return pmat; };
};

/// Full joint over hazard bitmasks Z ∈ {0,1}^{|E|}: eliminate inconsistent
/// states, renormalize; marginals by summing mass where Z_i = 1. Exponential in
/// |E| (≤62 edges).
class exactIpv : public ipv {
private:
  std::vector<double> prior_edge_prob_;
  std::vector<double> posterior_;
  size_t edges_;
  static constexpr double kEps = 1e-12;

  /// Shannon entropy H(dist) in bits over an arbitrary discrete distribution.
  static double entropyBits(const std::vector<double> &dist);

  /// H(p) = -p log2(p) - (1-p) log2(1-p); returns 0 at the boundaries.
  static double binaryEntropy(double p);

  /// Convert a bool-vector path to a compact bitmask (bit i set ⟺ path[i]).
  uint64_t pathToMask(const Path &path) const;

  /// True iff @p state is consistent with the observation: for a collision the
  /// state must have at least one hazard bit under @p query_mask; for safe
  /// passage it must have none.
  static bool consistent(uint64_t state, uint64_t query_mask,
                         bool observed_collision);

  /// Renormalize posterior_ to sum to 1; throws if total mass is ~0.
  void normalize();

  /// P(collision | query_mask) = Σ_{s: s ∧ mask ≠ 0} posterior_[s].
  double predictiveCollisionProbMask(uint64_t query_mask) const;

  /// Zero out posterior entries inconsistent with the observation, then
  /// renormalize.
  void observeMask(uint64_t query_mask, bool observed_collision);

public:
  /// Build the full 2^|E| joint from independent per-edge priors and
  /// normalize.  Throws if |E| > 62 (bitmask overflow).
  explicit exactIpv(Map map, const pMatrix &priors);

  /// Condition the posterior on a traversal outcome (collision or safe).
  void observe(const Path &path, bool observed_collision);

  /// Predictive probability of collision for a candidate path under the
  /// current posterior.
  double predictiveCollisionProb(const Path &path) const;

  /// Binary entropy of the predictive collision probability — an upper bound
  /// on the expected information gain from traversing @p path.
  double expectedInformationGain(const Path &path) const;

  /// Traverse @p path against the hidden map, condition the posterior on the
  /// outcome, and return (safe, realized_information_gain_bits).
  std::tuple<bool, double> informationGain(Path path) override;

  /// Marginal hazard probabilities P(Z_i = 1) for each edge under the current
  /// posterior.
  std::vector<double> marginals() const;
  const std::vector<double> &posterior() const { return posterior_; }
};
