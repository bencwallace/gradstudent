#pragma once

#include <tuple>
#include <type_traits>

namespace gs {

/* TUPLE CONCATENATOR */

template <typename... Ts>
using tuple_cat_t = decltype(std::tuple_cat(std::declval<Ts>()...));

/* TUPLE GENERATOR */

template <int N, typename T, typename... Ts> struct ntuple {
  using type = typename ntuple<N - 1, T, T, Ts...>::type;
};

template <typename T, typename... Ts> struct ntuple<0, T, Ts...> {
  using type = std::tuple<Ts...>;
};

/**
 * @brief Tuple type with N elements of type T
 */
template <int N, typename T> using ntuple_t = typename ntuple<N, T>::type;

/* BOOLEAN TO CONST */

template <typename NewType, bool...> struct bool_to_const {
  using type = std::tuple<>;
};

template <typename NewType, bool C, bool... Const>
struct bool_to_const<NewType, C, Const...> {
  using type = decltype(std::tuple_cat(
      std::tuple<typename std::conditional_t<C, const NewType, NewType>>{},
      typename bool_to_const<NewType, Const...>::type{}));
};

/**
 * @brief Tuple type with mixed const and non-const elements of type NewType
 *
 * Whether each element is const is determined by the corresponding boolean
 * value in the Const parameter pack.
 */
template <typename NewType, bool... Const>
using bool_to_const_t = typename bool_to_const<NewType, Const...>::type;

/* REFERENCE ADDER */

template <typename...> struct add_ref;

template <typename... Ts> struct add_ref<std::tuple<Ts...>> {
  using type =
      std::tuple<std::conditional_t<std::is_const_v<Ts>, Ts,
                                    std::add_lvalue_reference_t<Ts>>...>;
};

/**
 * @brief Tuple type with mixed reference and non-reference elements
 *
 * An elemenet is either a value type or a non-const reference, depending on
 * whether the corresponding element in the Types parameter pack is const or
 * not.
 */
template <typename... Types> using add_ref_t = typename add_ref<Types...>::type;

} // namespace gs
