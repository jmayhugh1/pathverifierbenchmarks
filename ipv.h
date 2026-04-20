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
  int start_node = -1;
  int end_node = -1;
  std::vector<std::tuple<size_t, size_t>> edge_pairs;
  /// Cached simple paths (as edge-index sequences) from start_node to end_node.
  /// Invalidated when start_node or end_node changes.
  mutable std::vector<std::vector<size_t>> cached_paths_;
  mutable bool paths_valid_ = false;
  void enumeratePaths() const;
  inline static const std::string GRAPH_CONFIGS_PATH = "configs/graphs.json";
  static const nlohmann::json &getGraphConfigs();

public:
  Graph(std::string graph_name);
  Graph(std::string name, int num_nodes, int num_edges,
        const std::vector<std::tuple<size_t, size_t>> &edge_pairs);
  const Map &getMap() const { return map; }
  void randomlyAssignHazards(double hazard_probability);
  void randomlyAssignHazards(double hazard_probability, std::mt19937 &rng);
  Path randomPath(std::mt19937 &rng, double p_query = 0.5);
  Path randomPath(double p_query = 0.5);
  int setStartNode(int start_node);
  int setEndNode(int end_node);
  int getStartNode() const { return start_node; }
  int getEndNode() const { return end_node; }
  int getNumNodes() const { return num_nodes; }
  /// Return a uniformly random simple path (as an edge mask) from start_node
  /// to end_node.  Throws if start/end are unset or no path exists.
  Path getRandomConnectedPath(std::mt19937 &rng) const;
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
  /// Per-edge hazard beliefs stored as log-odds: log(p/(1-p)).
  /// -inf means confirmed safe (p=0), +inf means certain hazard (p=1).
  std::vector<double> logodds_;
  /// Edge was on a path observed safe; hazard belief stays 0 permanently.
  std::vector<bool> confirmed_safe_;
  /// When true, informationGain uses jointEntropy(); otherwise marginal
  /// entropy.
  bool use_joint_ig_;

public:
  /// Construct with a ground-truth map and a uniform prior hazard probability
  /// per edge (default 0.5).  The prior is converted to log-odds internally.
  /// When @p useJointIG is true, informationGain reports the drop in the
  /// lifted-product joint entropy; otherwise it uses the sum of marginal
  /// entropies (numerically identical for a product distribution, but the flag
  /// keeps the API consistent with exactIpv).
  explicit approximateIpv(Map map, double prior = 0.5, bool useJointIG = false);

  /// Condition the beliefs on a traversal outcome (collision or safe).
  void observe(Path path, bool observed_collision);

  /// Simulate traversing @p path against the hidden map.  On collision the
  /// queried-edge beliefs are scaled up (Bayesian-like conditioning on at
  /// least one hazard); on safe passage the queried edges are zeroed and
  /// permanently locked.  Returns (safe, realized_information_gain_bits).
  std::tuple<bool, double> informationGain(Path path) override;

  /// Current marginal hazard probabilities P(Z_i=1) per edge, converted from
  /// internal log-odds representation.
  std::vector<double> marginals() const;

  /// Lifted independent-product joint over all 2^|E| hazard assignments:
  ///   Q(z) = ∏_i p_i^{z_i} (1-p_i)^{1-z_i}
  /// where p_i are the current approximate marginals.  Returned in bitmask
  /// order consistent with exactIpv::posterior().
  /// Throws if |E| > 62 (bitmask overflow), matching exactIpv's constraint.
  ///
  /// NOTE: This is NOT the true correlated posterior.  The approximate update
  /// treats edges independently, so the best joint representation is the
  /// product of its marginals.  After a collision observation the true
  /// posterior has correlations (e.g. "at least one queried edge is hazardous")
  /// that this factored form cannot capture.
  std::vector<double> posterior() const;

  /// Joint entropy of the lifted product distribution in bits.
  /// For an independent-product Q(z) = ∏ Bernoulli(p_i), the joint entropy
  /// decomposes exactly as H(Q) = Σ H(p_i) — this is the defining property
  /// of independence.  So this method is numerically equivalent to the
  /// marginal-sum entropy, but its semantics are "entropy of the full joint"
  /// for apples-to-apples comparison against exactIpv::jointEntropy().
  double jointEntropy() const;
};

/// Full joint over hazard bitmasks Z ∈ {0,1}^{|E|}: eliminate inconsistent
/// states, renormalize; marginals by summing mass where Z_i = 1. Exponential in
/// |E| (≤62 edges).
class exactIpv : public ipv {
private:
  /// Per-edge prior hazard stored as log-odds: log(p/(1-p)).
  std::vector<double> prior_edge_logodds_;
  /// Log-probability for each of the 2^|E| joint hazard states.
  std::vector<double> log_posterior_;
  size_t edges_;
  /// When true, informationGain uses joint entropy; otherwise marginal entropy.
  bool use_joint_ig_;

  /// Convert a bool-vector path to a compact bitmask (bit i set ⟺ path[i]).
  uint64_t pathToMask(const Path &path) const;

  /// True iff @p state is consistent with the observation: for a collision the
  /// state must have at least one hazard bit under @p query_mask; for safe
  /// passage it must have none.
  static bool consistent(uint64_t state, uint64_t query_mask,
                         bool observed_collision);

  /// Renormalize log_posterior_ so that logsumexp = 0 (i.e. probs sum to 1).
  void normalize();

  /// Log P(collision | query_mask) via logsumexp over matching states.
  double logPredictiveCollisionProbMask(uint64_t query_mask) const;

  /// Set log-prob to -inf for inconsistent states, then renormalize.
  void observeMask(uint64_t query_mask, bool observed_collision);

public:
  /// Build the full 2^|E| joint from independent per-edge priors in log-space
  /// and normalize.  Throws if |E| > 62 (bitmask overflow).
  /// When @p useJointIG is true (default), informationGain reports the drop in
  /// joint entropy; otherwise it uses the sum of marginal entropies.
  explicit exactIpv(Map map, const pMatrix &priors, bool useJointIG = true);

  /// Condition the posterior on a traversal outcome (collision or safe).
  void observe(const Path &path, bool observed_collision);

  /// Predictive probability of collision for a candidate path under the
  /// current posterior.
  double predictiveCollisionProb(const Path &path) const;

  /// Binary entropy of the predictive collision probability — an upper bound
  /// on the expected information gain from traversing @p path.
  double expectedInformationGain(const Path &path) const;

  /// Shannon entropy of the full joint posterior in bits:
  /// H = -Σ_z P(z) log₂ P(z), computed directly from log_posterior_.
  double jointEntropy() const;

  /// Sum of marginal binary entropies in bits: Σ H(p_i).
  double marginalEntropy() const;

  /// Traverse @p path against the hidden map, condition the posterior on the
  /// outcome, and return (safe, realized_information_gain_bits).
  /// Uses joint or marginal entropy according to the useJointIG flag.
  std::tuple<bool, double> informationGain(Path path) override;

  /// Marginal hazard probabilities P(Z_i = 1) per edge, converted from
  /// log-space posterior.
  std::vector<double> marginals() const;

  /// Posterior as probabilities (converted from internal log-space).
  std::vector<double> posterior() const;
};
