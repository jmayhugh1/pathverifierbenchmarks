#pragma once

#include "ipv.h"

#include <random>

Path randomPath(size_t num_edges, std::mt19937 &rng, double p_query = 0.5);
Path randomPath(size_t num_edges);

Map randomMap(size_t num_vertices, size_t num_edges, std::mt19937 &rng,
              double hazard_p = 0.5);
Map randomMap(size_t num_vertices, size_t num_edges);

float binary_entropy(float p);
float total_entropy(const pMatrix &p);
