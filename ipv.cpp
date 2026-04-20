#include "ipv.h"
#include "ipv_utils.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

const nlohmann::json &Graph::getGraphConfigs() {
  static nlohmann::json configs;
  if (configs.empty()) {
    std::ifstream file(GRAPH_CONFIGS_PATH);
    file >> configs;
  }
  return configs;
}

Graph::Graph(std::string name, int num_nodes, int num_edges,
             const std::vector<std::tuple<size_t, size_t>> &edge_pairs)
    : num_nodes(num_nodes), num_edges(num_edges), edge_pairs(edge_pairs) {
  for (const auto &edge : edge_pairs) {
    map.push_back({std::get<0>(edge), std::get<1>(edge), false});
  }
  assert(map.size() == num_edges);
}

Graph::Graph(std::string graph_name) {
  nlohmann::json configs = getGraphConfigs();
  nlohmann::json graph_config = configs.at(graph_name);
  num_nodes = graph_config.at("num_nodes");
  edge_pairs = graph_config.at("edge_pairs");
  for (const auto &edge : edge_pairs) {
    map.push_back({std::get<0>(edge), std::get<1>(edge), false});
  }
  num_edges = map.size();
  if (graph_config.contains("start_id")) {
    start_node = graph_config.at("start_id");
  }
  if (graph_config.contains("goal_id")) {
    end_node = graph_config.at("goal_id");
  }
}

void Graph::randomlyAssignHazards(double hazard_probability) {
  std::random_device rd;
  std::mt19937 rng(rd());
  randomlyAssignHazards(hazard_probability, rng);
}

void Graph::randomlyAssignHazards(double hazard_probability,
                                  std::mt19937 &rng) {
  std::bernoulli_distribution haz(hazard_probability);
  for (auto &edge : map) {
    std::get<2>(edge) = haz(rng);
  }
}

Path Graph::randomPath(double p_query) {
  std::random_device rd;
  std::mt19937 rng(rd());
  size_t num_edges = map.size();
  return ipv_utils::randomPath(num_edges, rng, p_query);
}

Path Graph::randomPath(std::mt19937 &rng, double p_query) {
  size_t num_edges = map.size();
  return ipv_utils::randomPath(num_edges, rng, p_query);
}

approximateIpv::approximateIpv(Map map, double prior, bool useJointIG)
    : ipv(std::move(map)), use_joint_ig_(useJointIG) {
  const double l0 = ipv_utils::probToLogOdds(prior);
  logodds_.assign(num_edges, l0);
  confirmed_safe_.assign(num_edges, false);
}

int Graph::setStartNode(int start_node) {
  if (start_node < 0 || start_node >= num_nodes) {
    throw std::invalid_argument("Start node must be between 0 and " +
                                std::to_string(num_nodes - 1));
  }
  this->start_node = start_node;
  paths_valid_ = false;
  return 0;
}

int Graph::setEndNode(int end_node) {
  if (end_node < 0 || end_node >= num_nodes) {
    throw std::invalid_argument("End node must be between 0 and " +
                                std::to_string(num_nodes - 1));
  }
  this->end_node = end_node;
  paths_valid_ = false;
  return 0;
}

void Graph::enumeratePaths() const {
  cached_paths_.clear();

  std::vector<std::vector<std::pair<size_t, size_t>>> adj(num_nodes);
  for (size_t i = 0; i < edge_pairs.size(); ++i) {
    auto [u, v] = edge_pairs[i];
    adj[u].emplace_back(v, i);
  }

  std::vector<bool> visited(num_nodes, false);
  std::vector<size_t> current;

  std::function<void(size_t)> dfs = [&](size_t node) {
    if (static_cast<int>(node) == end_node) {
      cached_paths_.push_back(current);
      return;
    }
    for (auto &[nbr, eidx] : adj[node]) {
      if (!visited[nbr]) {
        visited[nbr] = true;
        current.push_back(eidx);
        dfs(nbr);
        current.pop_back();
        visited[nbr] = false;
      }
    }
  };

  visited[start_node] = true;
  dfs(static_cast<size_t>(start_node));
  paths_valid_ = true;
}

Path Graph::getRandomConnectedPath(std::mt19937 &rng) const {
  if (start_node < 0 || end_node < 0) {
    throw std::runtime_error(
        "start_node and end_node must be set before calling "
        "getRandomConnectedPath");
  }

  if (!paths_valid_) {
    enumeratePaths();
  }

  if (cached_paths_.empty()) {
    throw std::runtime_error("No path exists from node " +
                             std::to_string(start_node) + " to node " +
                             std::to_string(end_node));
  }

  std::uniform_int_distribution<size_t> dist(0, cached_paths_.size() - 1);
  const auto &chosen = cached_paths_[dist(rng)];

  Path mask(edge_pairs.size(), false);
  for (size_t eidx : chosen) {
    mask[eidx] = true;
  }
  return mask;
}

std::vector<double> approximateIpv::marginals() const {
  return ipv_utils::logOddsToProbs(logodds_);
}

std::vector<double> approximateIpv::posterior() const {
  if (num_edges >= 63) {
    throw std::invalid_argument(
        "Lifted approximate posterior supports at most 62 edges.");
  }
  const uint64_t num_states = uint64_t{1} << num_edges;
  std::vector<double> post(num_states);

  for (uint64_t state = 0; state < num_states; ++state) {
    double log_w = 0.0;
    for (size_t i = 0; i < num_edges; ++i) {
      const bool hazardous = ((state >> i) & 1ULL) != 0;
      const double l = logodds_[i];
      log_w += hazardous ? -ipv_utils::softplus(-l) : -ipv_utils::softplus(l);
    }
    post[state] = std::exp(log_w);
  }
  return post;
}

double approximateIpv::jointEntropy() const {
  return ipv_utils::total_entropy_logodds(logodds_);
}

void approximateIpv::observe(Path path, bool observed_collision) {
  assert(path.size() == num_edges);
  constexpr double neg_inf = -std::numeric_limits<double>::infinity();

  for (size_t i = 0; i < num_edges; ++i) {
    if (confirmed_safe_[i]) {
      logodds_[i] = neg_inf;
    }
  }

  const bool safe = !observed_collision;

  // log(p_all_safe) = Σ_{queried} log(1-p_i) = Σ -softplus(l_i)
  double log_p_all_safe = 0.0;
  for (size_t i = 0; i < num_edges; ++i) {
    if (confirmed_safe_[i] || !path[i])
      continue;
    log_p_all_safe -= ipv_utils::softplus(logodds_[i]);
  }

  // log(alpha) = -log(p_unsafe) = -log(1 - exp(log_p_all_safe))
  double log_alpha = 0.0;
  if (!safe) {
    log_alpha = -ipv_utils::log1mexp(log_p_all_safe);
  }

  if (!safe) {
    for (size_t i = 0; i < num_edges; ++i) {
      if (confirmed_safe_[i] || !path[i])
        continue;
      // log(alpha * p_i) = log_alpha + log(p_i) = log_alpha - softplus(-l_i)
      const double log_ap = log_alpha - ipv_utils::softplus(-logodds_[i]);
      if (log_ap >= 0.0) {
        logodds_[i] = std::numeric_limits<double>::infinity();
      } else {
        // l' = log(alpha*p / (1 - alpha*p)) = log_ap - log(1 - exp(log_ap))
        logodds_[i] = log_ap - ipv_utils::log1mexp(log_ap);
      }
    }
  }

  if (safe) {
    for (size_t i = 0; i < num_edges; ++i) {
      if (path[i]) {
        confirmed_safe_[i] = true;
        logodds_[i] = neg_inf;
      }
    }
  }
}

std::tuple<bool, double> approximateIpv::informationGain(Path path) {
  assert(path.size() == num_edges);
  constexpr double neg_inf = -std::numeric_limits<double>::infinity();

  for (size_t i = 0; i < num_edges; ++i) {
    if (confirmed_safe_[i]) {
      logodds_[i] = neg_inf;
    }
  }

  const double h_before = use_joint_ig_
                              ? jointEntropy()
                              : ipv_utils::total_entropy_logodds(logodds_);
  const bool observed_collision = collision(path);
  observe(path, observed_collision);

  const double h_after = use_joint_ig_
                             ? jointEntropy()
                             : ipv_utils::total_entropy_logodds(logodds_);
  const double ig = h_before - h_after;
  return {!observed_collision, ig};
}

// --- exactIpv (full joint bitmask posterior, log-space) ---

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
  const double log_z = ipv_utils::logsumexp(log_posterior_);
  if (std::isinf(log_z) && log_z < 0) {
    throw std::runtime_error(
        "Posterior became zero: observations are inconsistent.");
  }
  for (double &lw : log_posterior_) {
    lw -= log_z;
  }
}

double exactIpv::logPredictiveCollisionProbMask(uint64_t query_mask) const {
  std::vector<double> matching;
  for (uint64_t state = 0; state < log_posterior_.size(); ++state) {
    if ((state & query_mask) != 0) {
      matching.push_back(log_posterior_[state]);
    }
  }
  return ipv_utils::logsumexp(matching);
}

void exactIpv::observeMask(uint64_t query_mask, bool observed_collision) {
  constexpr double neg_inf = -std::numeric_limits<double>::infinity();
  for (uint64_t state = 0; state < log_posterior_.size(); ++state) {
    if (!consistent(state, query_mask, observed_collision)) {
      log_posterior_[state] = neg_inf;
    }
  }
  normalize();
}

exactIpv::exactIpv(Map map, const pMatrix &priors, bool useJointIG)
    : ipv(std::move(map)),
      prior_edge_logodds_(ipv_utils::probsToLogOdds(priors)), edges_(num_edges),
      use_joint_ig_(useJointIG) {

  if (priors.size() != edges_) {
    throw std::invalid_argument("Need one prior probability per edge.");
  }
  if (edges_ >= 63) {
    throw std::invalid_argument(
        "This exact bitmask version supports at most 62 edges.");
  }
  for (double p : priors) {
    if (p < 0.0 || p > 1.0) {
      throw std::invalid_argument("Each prior must lie in [0,1].");
    }
  }

  const uint64_t num_states = uint64_t{1} << edges_;
  log_posterior_.assign(num_states, 0.0);

  // Build joint in log-space: log P(state) = Σ log P(edge_i | state_bit_i)
  // log(p) = -softplus(-l), log(1-p) = -softplus(l)
  for (uint64_t state = 0; state < num_states; ++state) {
    double log_w = 0.0;
    for (size_t i = 0; i < edges_; ++i) {
      const bool hazardous = ((state >> i) & 1ULL) != 0;
      const double l = prior_edge_logodds_[i];
      log_w += hazardous ? -ipv_utils::softplus(-l) : -ipv_utils::softplus(l);
    }
    log_posterior_[state] = log_w;
  }
  normalize();
}

void exactIpv::observe(const Path &path, bool observed_collision) {
  observeMask(pathToMask(path), observed_collision);
}

double exactIpv::predictiveCollisionProb(const Path &path) const {
  return std::exp(logPredictiveCollisionProbMask(pathToMask(path)));
}

double exactIpv::expectedInformationGain(const Path &path) const {
  const double p = predictiveCollisionProb(path);
  return ipv_utils::binary_entropy(p);
}

double exactIpv::jointEntropy() const {
  const double log2_e = 1.0 / std::log(2.0);
  double h = 0.0;
  for (double lp : log_posterior_) {
    if (std::isinf(lp) && lp < 0)
      continue;
    h -= std::exp(lp) * lp;
  }
  return h * log2_e;
}

double exactIpv::marginalEntropy() const {
  auto lo = ipv_utils::probsToLogOdds(marginals());
  return ipv_utils::total_entropy_logodds(lo);
}

std::tuple<bool, double> exactIpv::informationGain(Path path) {
  const double h_before = use_joint_ig_ ? jointEntropy() : marginalEntropy();
  observe(path, collision(path));
  const double h_after = use_joint_ig_ ? jointEntropy() : marginalEntropy();
  return {!collision(path), h_before - h_after};
}

std::vector<double> exactIpv::marginals() const {
  // For each edge, logsumexp over states where that bit is set.
  std::vector<double> m(edges_);
  for (size_t i = 0; i < edges_; ++i) {
    std::vector<double> matching;
    const uint64_t bit = uint64_t{1} << i;
    for (uint64_t state = 0; state < log_posterior_.size(); ++state) {
      if ((state & bit) != 0) {
        matching.push_back(log_posterior_[state]);
      }
    }
    m[i] = std::exp(ipv_utils::logsumexp(matching));
  }
  return m;
}

std::vector<double> exactIpv::posterior() const {
  std::vector<double> p(log_posterior_.size());
  for (size_t i = 0; i < log_posterior_.size(); ++i) {
    p[i] = std::exp(log_posterior_[i]);
  }
  return p;
}
