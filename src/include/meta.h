#pragma once

#include <tuple>
#include <type_traits>

namespace gradstudent {

/* TUPLE GENERATOR */

template <int N, typename T, typename... Ts> struct ntuple {
  using type = typename ntuple<N - 1, T, T, Ts...>::type;
};

template <typename T, typename... Ts> struct ntuple<0, T, Ts...> {
  using type = std::tuple<Ts...>;
};

template <int N, typename T> using ntuple_t = typename ntuple<N, T>::type;

/* CONST CONVERTER */

template <typename NewType, typename...> struct const_convert {
  using type = std::tuple<>;
};

template <typename NewType, typename T, typename... Types>
struct const_convert<NewType, T, Types...> {
  using type = decltype(std::tuple_cat(
      std::tuple<typename std::conditional_t<std::is_const_v<T>, const NewType,
                                             NewType>>{},
      typename const_convert<NewType, Types...>::type{}));
};

template <typename NewType, typename... Types>
using const_convert_t = typename const_convert<NewType, Types...>::type;

/* REFERENCE ADDER */

template <typename...> struct add_ref;

template <typename... Ts> struct add_ref<std::tuple<Ts...>> {
  using type = std::tuple<typename std::add_lvalue_reference_t<Ts>...>;
};

template <typename... Types> using add_ref_t = typename add_ref<Types...>::type;

} // namespace gradstudent
