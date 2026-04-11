#include <cmath>
#include <limits>

#include <doctest/doctest.h>

#include "ipv.h"
#include "ipv_utils.h"

TEST_CASE("collision when masked edge is obstructed") {
  Map m = {{0, 0, false}, {0, 1, true}};
  approximateIpv sim(m);
  Path path = {false, true};
  CHECK(sim.collision(path));
}

TEST_CASE("safe path when queried edges are clear") {
  Map m = {{0, 0, false}, {0, 1, false}};
  approximateIpv sim(m);
  Path path = {true, true};
  CHECK_FALSE(sim.collision(path));
}

TEST_CASE("informationGain returns safe=false on obstruction") {
  Map m = {{0, 0, false}, {0, 1, true}};
  approximateIpv sim(m);
  Path path = {false, true};
  auto [safe, ig] = sim.informationGain(path);
  CHECK_FALSE(safe);
  CHECK(ig >= 0.f);
}

TEST_CASE("exactIpv five-edge joint update marginals") {
  Map truth = {{0, 1, false},
               {1, 2, false},
               {2, 3, false},
               {3, 4, false},
               {4, 5, false}};
  exactIpv solver(truth, pMatrix(5, 0.2f));
  solver.observe(Path{true, true, true, false, false}, true);
  solver.observe(Path{false, false, true, true, false}, false);
  solver.observe(Path{false, true, false, false, true}, true);
  auto m = solver.marginals();
  REQUIRE(m.size() == 5);
  CHECK(m[0] == doctest::Approx(9.0 / 29.0));
  CHECK(m[1] == doctest::Approx(25.0 / 29.0));
  CHECK(m[2] == doctest::Approx(0.0));
  CHECK(m[3] == doctest::Approx(0.0));
  CHECK(m[4] == doctest::Approx(9.0 / 29.0));
}

TEST_CASE("exactIpv expectedInformationGain is binary entropy of predictive "
          "collision") {
  Map truth = {{0, 0, false}, {0, 1, false}};
  exactIpv solver(truth, pMatrix(2, 0.5f));
  Path q = {true, false};
  const double p = solver.predictiveCollisionProb(q);
  CHECK(solver.expectedInformationGain(q) ==
        doctest::Approx(-p * std::log2(p) - (1.0 - p) * std::log2(1.0 - p)));
}

TEST_CASE("approximateIpv informationGain mutates edge beliefs") {
  Map m = {{0, 0, false}, {0, 1, true}};
  approximateIpv sim(m, 0.5f);
  CHECK(sim.marginals()[0] == doctest::Approx(0.5f));
  CHECK(sim.marginals()[1] == doctest::Approx(0.5f));

  sim.informationGain(Path{false, true});

  CHECK(sim.marginals()[0] == doctest::Approx(0.5f));
  CHECK(sim.marginals()[1] == doctest::Approx(1.0f));
}

TEST_CASE(
    "approximateIpv safe observation zeros beliefs on queried edges only") {
  Map m = {{0, 0, false}, {0, 1, false}};
  approximateIpv sim(m, 0.5f);
  sim.informationGain(Path{true, false});
  CHECK(sim.marginals()[0] == doctest::Approx(0.0f));
  CHECK(sim.marginals()[1] == doctest::Approx(0.5f));
}

TEST_CASE("approximateIpv confirmed-safe edges stay zero after later updates") {
  Map m = {{0, 0, false}, {0, 1, true}};
  approximateIpv sim(m, 0.5f);
  sim.informationGain(Path{true, false});
  CHECK(sim.marginals()[0] == doctest::Approx(0.f));
  sim.informationGain(Path{true, true});
  CHECK(sim.marginals()[0] == doctest::Approx(0.f));
  CHECK(sim.marginals()[1] > 0.f);
}

TEST_CASE("exactIpv informationGain mutates posterior and marginals") {
  Map m = {{0, 0, false}, {0, 1, true}};
  exactIpv sim(m, pMatrix(2, 0.5f));
  const auto m_before = sim.marginals();
  const double p0_before = sim.posterior()[0];

  sim.informationGain(Path{false, true});

  const auto m_after = sim.marginals();
  double sum = 0.0;
  for (double w : sim.posterior()) {
    sum += w;
  }
  CHECK(sum == doctest::Approx(1.0));
  CHECK(m_after[0] == doctest::Approx(m_before[0]));
  CHECK(m_after[1] > m_before[1]);
  CHECK(sim.posterior()[0] != doctest::Approx(p0_before));
}

TEST_CASE("exactIpv observe safe leaves only states with no hazard on queried "
          "edges") {
  Map m = {{0, 1, false}};
  exactIpv sim(m, pMatrix{0.4f});
  sim.observe(Path{true}, false);
  const auto mar = sim.marginals();
  REQUIRE(mar.size() == 1);
  CHECK(mar[0] == doctest::Approx(0.0));
  CHECK(sim.posterior()[0] == doctest::Approx(1.0));
}

TEST_CASE("exactIpv observe collision conditions on hazard along query mask") {
  Map m = {{0, 1, false}, {1, 2, false}};
  exactIpv sim(m, pMatrix{0.5f, 0.5f});
  sim.observe(Path{true, false}, true);
  const auto mar = sim.marginals();
  REQUIRE(mar.size() == 2);
  CHECK(mar[0] == doctest::Approx(1.0));
  CHECK(mar[1] == doctest::Approx(0.5));
}

TEST_CASE("approxiamteIPV and exactIPV produce the same marginals") {
  Map m = {{0, 1, false}, {1, 2, false}};
  approximateIpv sim(m, 0.5f);
  exactIpv sim_exact(m, pMatrix{0.5f, 0.5f});

  CHECK(sim.marginals() == sim_exact.marginals());

  // now check after one query
  sim.informationGain(Path{true, false});
  sim_exact.informationGain(Path{true, false});
  CHECK(sim.marginals() == sim_exact.marginals());

  sim.informationGain(Path{true, true});
  sim_exact.informationGain(Path{true, true});
  CHECK(sim.marginals() == sim_exact.marginals());
}

// ====================================================================
// Log-odds internal representation tests
// ====================================================================

TEST_CASE("ipv_utils: probToLogOdds / logOddsToProb round-trip") {
  for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
    const double l = ipv_utils::probToLogOdds(p);
    CHECK(ipv_utils::logOddsToProb(l) == doctest::Approx(p));
  }
}

TEST_CASE("ipv_utils: probToLogOdds boundary values") {
  CHECK(ipv_utils::probToLogOdds(0.0) ==
        -std::numeric_limits<double>::infinity());
  CHECK(ipv_utils::probToLogOdds(1.0) ==
        std::numeric_limits<double>::infinity());
  CHECK(ipv_utils::probToLogOdds(0.5) == doctest::Approx(0.0));
}

TEST_CASE("ipv_utils: softplus basic properties") {
  CHECK(ipv_utils::softplus(0.0) == doctest::Approx(std::log(2.0)));
  CHECK(ipv_utils::softplus(100.0) == doctest::Approx(100.0));
  CHECK(ipv_utils::softplus(-100.0) == doctest::Approx(0.0).epsilon(1e-40));
  for (double x : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
    CHECK(ipv_utils::softplus(x) == doctest::Approx(std::log1p(std::exp(x))));
  }
}

TEST_CASE("ipv_utils: binary_entropy_logodds matches binary_entropy") {
  for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
    const double l = ipv_utils::probToLogOdds(p);
    CHECK(ipv_utils::binary_entropy_logodds(l) ==
          doctest::Approx(ipv_utils::binary_entropy(p)).epsilon(1e-10));
  }
  CHECK(ipv_utils::binary_entropy_logodds(
            -std::numeric_limits<double>::infinity()) == 0.0);
  CHECK(ipv_utils::binary_entropy_logodds(
            std::numeric_limits<double>::infinity()) == 0.0);
}

TEST_CASE("ipv_utils: total_entropy_logodds matches total_entropy") {
  pMatrix probs = {0.1, 0.5, 0.9, 0.25};
  std::vector<double> lo = ipv_utils::probsToLogOdds(probs);
  CHECK(ipv_utils::total_entropy_logodds(lo) ==
        doctest::Approx(ipv_utils::total_entropy(probs)).epsilon(1e-10));
}

TEST_CASE("ipv_utils: logsumexp correctness") {
  std::vector<double> vals = {-1.0, -2.0, -3.0};
  double expected = std::log(std::exp(-1.0) + std::exp(-2.0) + std::exp(-3.0));
  CHECK(ipv_utils::logsumexp(vals) == doctest::Approx(expected));

  CHECK(ipv_utils::logsumexp({}) ==
        -std::numeric_limits<double>::infinity());
}

TEST_CASE("ipv_utils: log1mexp basic correctness") {
  CHECK(ipv_utils::log1mexp(-1.0) ==
        doctest::Approx(std::log(1.0 - std::exp(-1.0))));
  CHECK(ipv_utils::log1mexp(-0.01) ==
        doctest::Approx(std::log(1.0 - std::exp(-0.01))).epsilon(1e-10));
}

TEST_CASE("approximateIpv internal log-odds: prior 0.5 yields logodds 0") {
  Map m = {{0, 1, false}, {1, 2, false}};
  approximateIpv sim(m, 0.5);
  auto probs = sim.marginals();
  for (double p : probs) {
    CHECK(p == doctest::Approx(0.5));
  }
  double lo = ipv_utils::probToLogOdds(0.5);
  CHECK(lo == doctest::Approx(0.0));
}

TEST_CASE("approximateIpv: safe observation drives marginals to zero") {
  Map m = {{0, 1, false}, {1, 2, false}};
  approximateIpv sim(m, 0.3);
  sim.observe(Path{true, false}, false);
  auto probs = sim.marginals();
  CHECK(probs[0] == doctest::Approx(0.0));
  CHECK(probs[1] == doctest::Approx(0.3));
}

TEST_CASE("approximateIpv: collision observation increases queried edge "
           "beliefs") {
  Map m = {{0, 1, false}, {1, 2, false}};
  approximateIpv sim(m, 0.3);
  auto probs_before = sim.marginals();
  sim.observe(Path{true, false}, true);
  auto probs_after = sim.marginals();
  CHECK(probs_after[0] > probs_before[0]);
  CHECK(probs_after[1] == doctest::Approx(probs_before[1]));
}

TEST_CASE("exactIpv: log-posterior sums to 1 in probability space") {
  Map m = {{0, 1, false}, {1, 2, false}};
  exactIpv sim(m, pMatrix{0.3, 0.7});

  auto post = sim.posterior();
  double sum = 0.0;
  for (double w : post) {
    sum += w;
  }
  CHECK(sum == doctest::Approx(1.0));

  sim.observe(Path{true, false}, true);
  post = sim.posterior();
  sum = 0.0;
  for (double w : post) {
    sum += w;
  }
  CHECK(sum == doctest::Approx(1.0));
}

TEST_CASE("exactIpv: marginals from log-posterior match hand computation") {
  Map m = {{0, 1, false}};
  exactIpv sim(m, pMatrix{0.4});
  auto mar = sim.marginals();
  REQUIRE(mar.size() == 1);
  CHECK(mar[0] == doctest::Approx(0.4));
}

TEST_CASE("exactIpv: informationGain with marginal entropy flag matches "
           "marginal entropy drop") {
  Map truth = {{0, 1, false}, {1, 2, true}};
  exactIpv sim(truth, pMatrix{0.5, 0.5}, false);

  auto lo_before = ipv_utils::probsToLogOdds(sim.marginals());
  double h_before = ipv_utils::total_entropy_logodds(lo_before);

  auto [safe, ig] = sim.informationGain(Path{false, true});

  auto lo_after = ipv_utils::probsToLogOdds(sim.marginals());
  double h_after = ipv_utils::total_entropy_logodds(lo_after);

  CHECK(ig == doctest::Approx(h_before - h_after).epsilon(1e-10));
}

TEST_CASE("exactIpv: informationGain with joint entropy flag matches "
           "joint entropy drop") {
  Map truth = {{0, 1, false}, {1, 2, true}};
  exactIpv sim(truth, pMatrix{0.5, 0.5}, true);

  double h_before = sim.jointEntropy();
  auto [safe, ig] = sim.informationGain(Path{false, true});
  double h_after = sim.jointEntropy();

  CHECK(ig == doctest::Approx(h_before - h_after).epsilon(1e-10));
}

TEST_CASE("exactIpv: predictiveCollisionProb matches logsumexp derivation") {
  Map m = {{0, 1, false}, {1, 2, false}};
  exactIpv sim(m, pMatrix{0.3, 0.7});
  double p = sim.predictiveCollisionProb(Path{true, true});
  CHECK(p >= 0.0);
  CHECK(p <= 1.0);
  double expected = 1.0 - (1.0 - 0.3) * (1.0 - 0.7);
  CHECK(p == doctest::Approx(expected));
}

// ====================================================================
// Counter-example: joint reasoning captures what marginal-only misses
// ====================================================================
// Graph: v1--e1--v2--e2--v3--e3--v4--e4--v5  (4 edges, uniform prior 0.5)
//
// Observation 1: query {e2,e3,e4}, collision (r=1)
//   => Z_e2 ∨ Z_e3 ∨ Z_e4 = 1
//
// Observation 2: query {e1,e2,e3}, safe (r=0)
//   => Z_e1=0, Z_e2=0, Z_e3=0
//
// Combined: Z_e4 must be 1.  The exact joint posterior captures this;
// the approximate (marginal-only) update cannot.

TEST_CASE("counter-example: exact IPV deduces Z_e4=1 from joint "
           "constraints; approximate IPV cannot") {
  // Ground truth: only e4 is hazardous.
  // q1 = {e2,e3,e4} → collision (e4 hit)
  // q2 = {e1,e2,e3} → safe     (none hazardous)
  Map m = {
      {0, 1, false}, // e1 = (v1,v2)
      {1, 2, false}, // e2 = (v2,v3)
      {2, 3, false}, // e3 = (v3,v4)
      {3, 4, true},  // e4 = (v4,v5) — hazardous
  };

  approximateIpv approx(m, 0.5);
  exactIpv exact(m, pMatrix(4, 0.5));

  auto print_state = [](const std::string &label, const std::vector<double> &mar) {
    MESSAGE(label << " marginals: ["
            << mar[0] << ", " << mar[1] << ", "
            << mar[2] << ", " << mar[3] << "]");
  };

  MESSAGE("=== Prior ===");
  print_state("approx", approx.marginals());
  print_state("exact ", exact.marginals());

  // Observation 1: query {e2,e3,e4} → collision (e4 is hazardous)
  const Path q1 = {false, true, true, true};
  const auto [safe_a1, ig_a1] = approx.informationGain(q1);
  const auto [safe_e1, ig_e1] = exact.informationGain(q1);

  CHECK_FALSE(safe_a1);
  CHECK_FALSE(safe_e1);

  MESSAGE("=== After obs 1: query {e2,e3,e4}, collision ===");
  print_state("approx", approx.marginals());
  print_state("exact ", exact.marginals());
  MESSAGE("approx IG = " << ig_a1 << " bits");
  MESSAGE("exact  IG = " << ig_e1 << " bits");

  // Observation 2: query {e1,e2,e3} → safe (none hazardous)
  const Path q2 = {true, true, true, false};
  const auto [safe_a2, ig_a2] = approx.informationGain(q2);
  const auto [safe_e2, ig_e2] = exact.informationGain(q2);

  CHECK(safe_a2);
  CHECK(safe_e2);

  MESSAGE("=== After obs 2: query {e1,e2,e3}, safe ===");
  print_state("approx", approx.marginals());
  print_state("exact ", exact.marginals());
  MESSAGE("approx IG = " << ig_a2 << " bits");
  MESSAGE("exact  IG = " << ig_e2 << " bits");
  MESSAGE("--- Total ---");
  MESSAGE("approx total IG = " << (ig_a1 + ig_a2) << " bits");
  MESSAGE("exact  total IG = " << (ig_e1 + ig_e2) << " bits");

  const auto exact_m = exact.marginals();
  const auto approx_m = approx.marginals();

  REQUIRE(exact_m.size() == 4);
  REQUIRE(approx_m.size() == 4);

  CHECK(exact_m[0] == doctest::Approx(0.0));
  CHECK(exact_m[1] == doctest::Approx(0.0));
  CHECK(exact_m[2] == doctest::Approx(0.0));
  CHECK(exact_m[3] == doctest::Approx(1.0));

  CHECK(approx_m[0] == doctest::Approx(0.0));
  CHECK(approx_m[1] == doctest::Approx(0.0));
  CHECK(approx_m[2] == doctest::Approx(0.0));
  CHECK(approx_m[3] < 1.0);
}

// ====================================================================
// Joint entropy vs marginal entropy tests
// ====================================================================

TEST_CASE("exactIpv: jointEntropy equals marginalEntropy under independent "
           "prior") {
  Map m = {{0, 1, false}, {1, 2, false}, {2, 3, false}};
  exactIpv sim(m, pMatrix{0.3, 0.5, 0.7});
  CHECK(sim.jointEntropy() ==
        doctest::Approx(sim.marginalEntropy()).epsilon(1e-10));
}

TEST_CASE("exactIpv: jointEntropy <= marginalEntropy after observation "
           "introduces correlations") {
  Map m = {{0, 1, true}, {1, 2, false}, {2, 3, false}};
  exactIpv sim(m, pMatrix{0.5, 0.5, 0.5});
  sim.observe(Path{true, true, false}, true);
  CHECK(sim.jointEntropy() <= sim.marginalEntropy() + 1e-10);
}

TEST_CASE("exactIpv: jointEntropy is zero when posterior is deterministic") {
  Map m = {{0, 1, false}};
  exactIpv sim(m, pMatrix{0.5});
  sim.observe(Path{true}, false);
  CHECK(sim.jointEntropy() == doctest::Approx(0.0).epsilon(1e-12));
}

TEST_CASE("exactIpv: joint IG >= marginal IG for same observation sequence") {
  Map m = {
      {0, 1, false},
      {1, 2, false},
      {2, 3, true},
  };
  exactIpv joint_sim(m, pMatrix{0.5, 0.5, 0.5}, true);
  exactIpv marg_sim(m, pMatrix{0.5, 0.5, 0.5}, false);

  Path q1 = {true, true, true};
  auto [s_j1, ig_j1] = joint_sim.informationGain(q1);
  auto [s_m1, ig_m1] = marg_sim.informationGain(q1);
  CHECK(ig_j1 >= ig_m1 - 1e-10);

  Path q2 = {true, false, false};
  auto [s_j2, ig_j2] = joint_sim.informationGain(q2);
  auto [s_m2, ig_m2] = marg_sim.informationGain(q2);

  double total_joint = ig_j1 + ig_j2;
  double total_marg = ig_m1 + ig_m2;
  CHECK(total_joint >= total_marg - 1e-10);
}

TEST_CASE("exactIpv: useJointIG flag defaults to true") {
  Map m = {{0, 1, true}, {1, 2, false}};
  exactIpv default_sim(m, pMatrix{0.5, 0.5});
  exactIpv joint_sim(m, pMatrix{0.5, 0.5}, true);

  Path q = {true, true};
  auto [s1, ig1] = default_sim.informationGain(q);
  auto [s2, ig2] = joint_sim.informationGain(q);
  CHECK(ig1 == doctest::Approx(ig2).epsilon(1e-12));
}
