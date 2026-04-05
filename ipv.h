#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
using pMatrix = std::vector<float>;

/// Bernoulli mask over edges; each bit true with probability `p_query` (default
/// 0.5).
Path randomPath(size_t num_edges, std::mt19937 &rng, double p_query = 0.5);
Path randomPath(size_t num_edges);

/// `num_edges` distinct directed edges with endpoints in [0, num_vertices);
/// each obstructed independently with probability `hazard_p` (default 0.2).
/// Uses all n² ordered pairs; throws if num_edges > num_vertices *
/// num_vertices.
Map randomMap(size_t num_vertices, size_t num_edges, std::mt19937 &rng,
              double hazard_p = 0.2);
Map randomMap(size_t num_vertices, size_t num_edges);

class ipv {
private:
  Map map;

protected:
  /// Number of edges |E| (path length); not |V|.
  size_t num_edges;
  static constexpr float eps = 1e-12f;

  explicit ipv(Map map) : map(std::move(map)) { num_edges = this->map.size(); }

public:
  virtual ~ipv() = default;
  /// Returns (safe, information_gain_bits): safe iff no queried edge is
  /// obstructed.
  virtual std::tuple<bool, float> informationGain(Path path) = 0;

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
  explicit approximateIpv(Map map, float prior = 0.5f);
  std::tuple<bool, float> informationGain(Path path) override;
  /// Current marginal hazard beliefs P(Z_i=1) per edge (updated by
  /// informationGain).
  const pMatrix &edgeBeliefs() const { return pmat; }
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

  static double entropyBits(const std::vector<double> &dist);
  static double binaryEntropy(double p);
  uint64_t pathToMask(const Path &path) const;
  static bool consistent(uint64_t state, uint64_t query_mask,
                         bool observed_collision);
  void normalize();
  double predictiveCollisionProbMask(uint64_t query_mask) const;
  void observeMask(uint64_t query_mask, bool observed_collision);

public:
  explicit exactIpv(Map map, const pMatrix &priors);

  void observe(const Path &path, bool observed_collision);
  double predictiveCollisionProb(const Path &path) const;
  double expectedInformationGain(const Path &path) const;
  std::tuple<bool, float> informationGain(Path path) override;
  std::vector<double> marginals() const;
  const std::vector<double> &posterior() const { return posterior_; }
};
