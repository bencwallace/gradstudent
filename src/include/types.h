/**
 * @file types.h
 * @author Ben Wallace (me@bcwallace.com)
 * @brief Simple type definitions
 * @version 0.1
 * @date 2024-03-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <cstddef>
#include <vector>

namespace gradstudent {

using std::size_t;

/**
 * @brief A sequence of numbers
 *
 * Suitable for indexing tensors
 */
using array_t = std::vector<size_t>;

} // namespace gradstudent
