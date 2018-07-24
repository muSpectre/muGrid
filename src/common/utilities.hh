/**
 * @file   utilities.hh
 *
 * @author Till Junge <till.junge@epfl.ch>
 *
 * @date   17 Nov 2017
 *
 * @brief  additions to the standard name space to anticipate C++17 features
 *
 * Copyright © 2017 Till Junge
 *
 * µSpectre is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3, or (at
 * your option) any later version.
 *
 * µSpectre is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Emacs; see the file COPYING. If not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */


#ifndef UTILITIES_H
#define UTILITIES_H

#include <boost/tuple/tuple.hpp>

#include <tuple>

#ifdef NO_EXPERIMENTAL
#  if defined(__INTEL_COMPILER)
//#    pragma warning ( disable : 383 )
#  elif defined (__clang__) // test clang to be sure that when we test for gnu it is only gnu
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Weffc++"
#  elif (defined(__GNUC__) || defined(__GNUG__))
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Weffc++"
#  endif
#  include <boost/optional.hpp>
#  if defined(__INTEL_COMPILER)
//#    pragma warning ( disable : 383 )
#  elif defined (__clang__) // test clang to be sure that when we test for gnu it is only gnu
#    pragma clang diagnostic pop
#    pragma clang diagnostic ignored "-Weffc++"
#  elif (defined(__GNUC__) || defined(__GNUG__))
#    pragma GCC diagnostic pop
#    pragma GCC diagnostic ignored "-Weffc++"
#  endif
#else
#  include <experimental/optional>
#endif

namespace std_replacement {

  namespace detail {
    template <class T>
    struct is_reference_wrapper : std::false_type {};
    template <class U>
    struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

    //! from cppreference
    template <class Base, class T, class Derived, class... Args>
    auto INVOKE(T Base::*pmf, Derived&& ref, Args&&... args)
      noexcept(noexcept((std::forward<Derived>(ref).*pmf)(std::forward<Args>(args)...)))
      -> std::enable_if_t<std::is_function<T>::value &&
                          std::is_base_of<Base, std::decay_t<Derived>>::value,
                          decltype((std::forward<Derived>(ref).*pmf)(std::forward<Args>(args)...))>
    {
      return (std::forward<Derived>(ref).*pmf)(std::forward<Args>(args)...);
    }

    //! from cppreference
    template <class Base, class T, class RefWrap, class... Args>
    auto INVOKE(T Base::*pmf, RefWrap&& ref, Args&&... args)
      noexcept(noexcept((ref.get().*pmf)(std::forward<Args>(args)...)))
      -> std::enable_if_t<std::is_function<T>::value &&
                          is_reference_wrapper<std::decay_t<RefWrap>>::value,
                          decltype((ref.get().*pmf)(std::forward<Args>(args)...))>

    {
      return (ref.get().*pmf)(std::forward<Args>(args)...);
    }

    //! from cppreference
    template <class Base, class T, class Pointer, class... Args>
    auto INVOKE(T Base::*pmf, Pointer&& ptr, Args&&... args)
      noexcept(noexcept(((*std::forward<Pointer>(ptr)).*pmf)(std::forward<Args>(args)...)))
      -> std::enable_if_t<std::is_function<T>::value &&
                          !is_reference_wrapper<std::decay_t<Pointer>>::value &&
                          !std::is_base_of<Base, std::decay_t<Pointer>>::value,
                          decltype(((*std::forward<Pointer>(ptr)).*pmf)(std::forward<Args>(args)...))>
    {
      return ((*std::forward<Pointer>(ptr)).*pmf)(std::forward<Args>(args)...);
    }

    //! from cppreference
    template <class Base, class T, class Derived>
    auto INVOKE(T Base::*pmd, Derived&& ref)
      noexcept(noexcept(std::forward<Derived>(ref).*pmd))
      -> std::enable_if_t<!std::is_function<T>::value &&
                          std::is_base_of<Base, std::decay_t<Derived>>::value,
                          decltype(std::forward<Derived>(ref).*pmd)>
    {
      return std::forward<Derived>(ref).*pmd;
    }

    //! from cppreference
    template <class Base, class T, class RefWrap>
    auto INVOKE(T Base::*pmd, RefWrap&& ref)
      noexcept(noexcept(ref.get().*pmd))
      -> std::enable_if_t<!std::is_function<T>::value &&
                          is_reference_wrapper<std::decay_t<RefWrap>>::value,
                          decltype(ref.get().*pmd)>
    {
      return ref.get().*pmd;
    }

    //! from cppreference
    template <class Base, class T, class Pointer>
    auto INVOKE(T Base::*pmd, Pointer&& ptr)
      noexcept(noexcept((*std::forward<Pointer>(ptr)).*pmd))
      -> std::enable_if_t<!std::is_function<T>::value &&
                          !is_reference_wrapper<std::decay_t<Pointer>>::value &&
                          !std::is_base_of<Base, std::decay_t<Pointer>>::value,
                          decltype((*std::forward<Pointer>(ptr)).*pmd)>
    {
      return (*std::forward<Pointer>(ptr)).*pmd;
    }

    //! from cppreference
    template <class F, class... Args>
    auto INVOKE(F&& f, Args&&... args)
      noexcept(noexcept(std::forward<F>(f)(std::forward<Args>(args)...)))
      -> std::enable_if_t<!std::is_member_pointer<std::decay_t<F>>::value,
                          decltype(std::forward<F>(f)(std::forward<Args>(args)...))>
    {
      return std::forward<F>(f)(std::forward<Args>(args)...);
    }
  } // namespace detail

  //! from cppreference
  template< class F, class... ArgTypes >
  auto invoke(F&& f, ArgTypes&&... args)
  // exception specification for QoI
    noexcept(noexcept(detail::INVOKE(std::forward<F>(f), std::forward<ArgTypes>(args)...)))
    -> decltype(detail::INVOKE(std::forward<F>(f), std::forward<ArgTypes>(args)...))
  {
    return detail::INVOKE(std::forward<F>(f), std::forward<ArgTypes>(args)...);
  }

  namespace detail {
    //! from cppreference
    template <class F, class Tuple, std::size_t... I>
    constexpr decltype(auto) apply_impl(F &&f, Tuple &&t, std::index_sequence<I...>)
    {
      return std_replacement::invoke(std::forward<F>(f), std::get<I>(std::forward<Tuple>(t))...);
    }
  }  // namespace detail

  //! from cppreference
  template <class F, class Tuple>
  constexpr decltype(auto) apply(F &&f, Tuple &&t)
  {
    return detail::apply_impl
      (std::forward<F>(f), std::forward<Tuple>(t),
       std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>::value>{});
  }


} //namespace std_replacement


namespace muSpectre {

  namespace internal {

    /**
     * helper struct template to compute the type of a tuple with a
     * given number of entries of the same type
     */
    template <size_t size, typename T, typename... tail>
    struct tuple_array_helper {
      //! underlying tuple
      using type = typename tuple_array_helper<size-1, T, T, tail...>::type;
    };

    /**
     * helper struct template to compute the type of a tuple with a
     * given number of entries of the same type
     */
    template< typename T, typename... tail>
    struct tuple_array_helper<0, T, tail...> {
      //! underlying tuple
      using type = std::tuple<tail...>;
    };

    /**
     * helper struct for runtime index access to
     * tuples. RecursionLevel indicates how much more we can recurse
     * down
     */
    template <class TupArr, size_t Index=0, size_t RecursionLevel=TupArr::Size-1>
    struct Accessor {
      using Stored_t = typename TupArr::Stored_t;

      inline static Stored_t
      get(const size_t & index, TupArr & container) {
        if (index == Index) {
          return std::get<Index>(container);
        } else {
          return Accessor<TupArr, Index+1, RecursionLevel-1>::get(index, container);
        }
      }
      inline static const Stored_t
      get(const size_t & index, const TupArr & container) {
        if (index == Index) {
          return std::get<Index>(container);
        } else {
          return Accessor<TupArr, Index+1, RecursionLevel-1>::get(index, container);
        }
      }
    };

    /**
     * specialisation for recursion end
     */
    template <class TupArr, size_t Index>
    struct Accessor<TupArr, Index, 0> {
      using Stored_t = typename TupArr::Stored_t;

      inline static Stored_t
      get(const size_t & index, TupArr & container) {
        if (index == Index) {
          return std::get<Index>(container);
        } else {
          std::stringstream err{};
          err << "Index " << index << "is out of range.";
          throw std::runtime_error(err.str());
        }
      }

      inline static const Stored_t
      get(const size_t & index, const TupArr & container) {
        if (index == Index) {
          return std::get<Index>(container);
        } else {
          std::stringstream err{};
          err << "Index " << index << "is out of range.";
          throw std::runtime_error(err.str());
        }
      }
    };

    /**
     * helper struct that provides the tuple_array.
     */
    template <typename T, size_t size>
    struct tuple_array_provider {
      //! tuple type that can be used (almost) like an `std::array`
      class type: public tuple_array_helper<size, T>::type {
      public:
        //! short-hand
        using Parent = typename tuple_array_helper<size, T>::type;
        using Stored_t = T;
        constexpr static size_t Size{size};

        //! constructor
        inline type(Parent && parent):Parent{parent}{};

        //! element access
        T operator[] (const size_t & index) {
          return Accessor<type>::get(index, *this);
        }

        //! element access
        const T operator[](const size_t & index) const  {
          return Accessor<type>::get(index, *this);
        }
      protected:
      };
    };
  }  // internal

  /**
   * This is a convenience structure to create a tuple of `nb_elem`
   * entries of type `T`. It is named tuple_array, because it is
   * somewhat similar to an `std::array<T, nb_elem>`. The reason for
   * this structure is that the `std::array` is not allowed by the
   * standard to store references (8.3.2 References, paragraph 5:
   * "There shall be no references to references, no arrays of
   * references, and no pointers to references.") use this, if you
   * want to have a statically known number of references to store,
   * and you wish to do so efficiently.
   */
  template <typename T, size_t nb_elem>
  using tuple_array = typename internal::tuple_array_provider<T, nb_elem>::type;

  using std_replacement::apply;

  /**
   * emulation `std::optional` (a C++17 feature)
   */
  template <class T>
#ifdef NO_EXPERIMENTAL
  using optional = typename boost::optional<T>;
#else
  using optional = typename std::experimental::optional<T>;
#endif

  /* ---------------------------------------------------------------------- */
  /**
   * conversion helper from `boost::tuple` to `std::tuple`
   */
  template <typename BoostTuple, std::size_t... Is>
  auto asStdTuple(BoostTuple&& boostTuple, std::index_sequence<Is...>) {
    return std::tuple<typename boost::tuples::element<Is, std::decay_t<BoostTuple>>::type...>
      (boost::get<Is>(std::forward<BoostTuple>(boostTuple))...);
  }
  /**
   * conversion from `boost::tuple` to `std::tuple`
   */
  template <typename BoostTuple>
  auto asStdTuple(BoostTuple&& boostTuple) {
    return asStdTuple(std::forward<BoostTuple>(boostTuple),
                      std::make_index_sequence<boost::tuples::length<std::decay_t<BoostTuple>>::value>());
  }


}  // muSpectre

#endif /* UTILITIES_H */
