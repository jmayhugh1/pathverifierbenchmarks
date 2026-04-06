#include <doctest/doctest.h>

#include "ipv.h"

TEST_CASE("Reading 5v_6e graph config") {
  Graph graph("5v_6e");
  CHECK(graph.getMap().size() == 6);
  CHECK(graph.getMap()[0] == std::make_tuple(0, 3, false));
  CHECK(graph.getMap()[1] == std::make_tuple(0, 1, false));
  CHECK(graph.getMap()[2] == std::make_tuple(0, 2, false));
  CHECK(graph.getMap()[3] == std::make_tuple(2, 4, false));
  CHECK(graph.getMap()[4] == std::make_tuple(2, 3, false));
  CHECK(graph.getMap()[5] == std::make_tuple(3, 4, false));
  Path random_path = graph.randomPath(0.5);
  CHECK(random_path.size() == 6);
}
