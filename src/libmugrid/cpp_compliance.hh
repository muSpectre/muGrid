/**
 * @file   cpp_compliance.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   17 Nov 2017
 *
 * @brief  additions to the standard name space to anticipate C++17 features
 *
 * Copyright © 2017 Till Junge
 *
 * µGrid is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µGrid is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with µGrid; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * * Boston, MA 02111-1307, USA.
 *
 * Additional permission under GNU GPL version 3 section 7
 *
 * If you modify this Program, or any covered work, by linking or combining it
 * with proprietary FFT implementations or numerical libraries, containing parts
 * covered by the terms of those libraries' licenses, the licensors of this
 * Program grant you additional permission to convey the resulting work.
 *
 */

#ifndef SRC_LIBMUGRID_CPP_COMPLIANCE_HH_
#define SRC_LIBMUGRID_CPP_COMPLIANCE_HH_

#include <tuple>

#if __cplusplus < 201703L
#ifdef NO_EXPERIMENTAL
#include <boost/optional.hpp>
#else
#include <experimental/optional>
#endif
#else
#include <optional>
#endif

namespace std_replacement {

  namespace detail {
    template <class T>
    struct is_reference_wrapper : std::false_type {};
    template <class U>
    struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

    //! from cppreference
    template <class Base, class T, class Derived, class... Args>
    auto
    INVOKE(T Base::*pmf, Derived && ref, Args &&... args) noexcept(noexcept(
        (std::forward<Derived>(ref).*pmf)(std::forward<Args>(args)...)))
        -> std::enable_if_t<
            std::is_function<T>::value &&
                std::is_base_of<Base, std::decay_t<Derived>>::value,
            decltype((std::forward<Derived>(ref).*
                      pmf)(std::forward<Args>(args)...))> {
      return (std::forward<Derived>(ref).*pmf)(std::forward<Args>(args)...);
    }

    //! from cppreference
    template <class Base, class T, class RefWrap, class... Args>
    auto INVOKE(T Base::*pmf, RefWrap && ref, Args &&... args) noexcept(
        noexcept((ref.get().*pmf)(std::forward<Args>(args)...)))
        -> std::enable_if_t<
            std::is_function<T>::value &&
                is_reference_wrapper<std::decay_t<RefWrap>>::value,
            decltype((ref.get().*pmf)(std::forward<Args>(args)...))> {
      return (ref.get().*pmf)(std::forward<Args>(args)...);
    }

    //! from cppreference
    template <class Base, class T, class Pointer, class... Args>
    auto
    INVOKE(T Base::*pmf, Pointer && ptr, Args &&... args) noexcept(noexcept(
        ((*std::forward<Pointer>(ptr)).*pmf)(std::forward<Args>(args)...)))
        -> std::enable_if_t<
            std::is_function<T>::value &&
                !is_reference_wrapper<std::decay_t<Pointer>>::value &&
                !std::is_base_of<Base, std::decay_t<Pointer>>::value,
            decltype(((*std::forward<Pointer>(ptr)).*
                      pmf)(std::forward<Args>(args)...))> {
      return ((*std::forward<Pointer>(ptr)).*pmf)(std::forward<Args>(args)...);
    }

    //! from cppreference
    template <class Base, class T, class Derived>
    auto
    INVOKE(T Base::*pmd,
           Derived && ref) noexcept(noexcept(std::forward<Derived>(ref).*pmd))
        -> std::enable_if_t<
            !std::is_function<T>::value &&
                std::is_base_of<Base, std::decay_t<Derived>>::value,
            decltype(std::forward<Derived>(ref).*pmd)> {
      return std::forward<Derived>(ref).*pmd;
    }

    //! from cppreference
    template <class Base, class T, class RefWrap>
    auto INVOKE(T Base::*pmd, RefWrap && ref) noexcept(noexcept(ref.get().*pmd))
        -> std::enable_if_t<
            !std::is_function<T>::value &&
                is_reference_wrapper<std::decay_t<RefWrap>>::value,
            decltype(ref.get().*pmd)> {
      return ref.get().*pmd;
    }

    //! from cppreference
    template <class Base, class T, class Pointer>
    auto INVOKE(T Base::*pmd, Pointer && ptr) noexcept(
        noexcept((*std::forward<Pointer>(ptr)).*pmd))
        -> std::enable_if_t<
            !std::is_function<T>::value &&
                !is_reference_wrapper<std::decay_t<Pointer>>::value &&
                !std::is_base_of<Base, std::decay_t<Pointer>>::value,
            decltype((*std::forward<Pointer>(ptr)).*pmd)> {
      return (*std::forward<Pointer>(ptr)).*pmd;
    }

    //! from cppreference
    template <class F, class... Args>
    auto INVOKE(F && f, Args &&... args) noexcept(
        noexcept(std::forward<F>(f)(std::forward<Args>(args)...)))
        -> std::enable_if_t<
            !std::is_member_pointer<std::decay_t<F>>::value,
            decltype(std::forward<F>(f)(std::forward<Args>(args)...))> {
      return std::forward<F>(f)(std::forward<Args>(args)...);
    }
  }  // namespace detail

  //! from cppreference
  template <class F, class... ArgTypes>
  auto invoke(F && f, ArgTypes &&... args)
      // exception specification for QoI
      noexcept(noexcept(detail::INVOKE(std::forward<F>(f),
                                       std::forward<ArgTypes>(args)...)))
          -> decltype(detail::INVOKE(std::forward<F>(f),
                                     std::forward<ArgTypes>(args)...)) {
    return detail::INVOKE(std::forward<F>(f), std::forward<ArgTypes>(args)...);
  }

  namespace detail {
    //! from cppreference
    template <class F, class Tuple, std::size_t... I>
    constexpr decltype(auto) apply_impl(F && f, Tuple && t,
                                        std::index_sequence<I...>) {
      return std_replacement::invoke(std::forward<F>(f),
                                     std::get<I>(std::forward<Tuple>(t))...);
    }
  }  // namespace detail

  //! from cppreference
  template <class F, class Tuple>
  constexpr decltype(auto) apply(F && f, Tuple && t) {
    return detail::apply_impl(
        std::forward<F>(f), std::forward<Tuple>(t),
        std::make_index_sequence<
            std::tuple_size<std::decay_t<Tuple>>::value>{});
  }

}  // namespace std_replacement

namespace muGrid {

#if __cplusplus < 201703L
  using std_replacement::apply;
  /**
   * emulation `std::optional` (a C++17 feature)
   */
  template <class T>
#ifdef NO_EXPERIMENTAL
  using optional = typename boost::optional<T>;
  using nullopt_t = boost::none_t;
  const nullopt_t nullopt{boost::none};
#else
  using optional = typename std::experimental::optional<T>;
  using nullopt_t = std::experimental::fundamentals_v1::nullopt_t;
  constexpr nullopt_t nullopt{std::experimental::fundamentals_v1::nullopt};
#endif

#else
  using std::apply;
  using std::optional;
#endif

}  // namespace muGrid

#endif  // SRC_LIBMUGRID_CPP_COMPLIANCE_HH_
