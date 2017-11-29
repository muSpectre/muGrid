/**
 * @file   aka_iterators.hh
 *
 * @author Nicolas Richart
 *
 * @date creation  Wed Jul 19 2017
 *
 * @brief iterator interfaces
 *
 * @section LICENSE
 *
 * Copyright (©) 2010-2011 EPFL (Ecole Polytechnique Fédérale de Lausanne)
 * Laboratory (LSMS - Laboratoire de Simulation en Mécanique des Solides)
 *
 * Akantu is free  software: you can redistribute it and/or  modify it under the
 * terms  of the  GNU Lesser  General Public  License as  published by  the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * Akantu is  distributed in the  hope that it  will be useful, but  WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A  PARTICULAR PURPOSE. See  the GNU  Lesser General  Public License  for more
 * details.
 *
 * You should  have received  a copy  of the GNU  Lesser General  Public License
 * along with Akantu. If not, see <http://www.gnu.org/licenses/>.
 *
 * Above block was left intact as in akantu. µSpectre exercises the
 * right to redistribute and modify the code below
 *
 */
/* -------------------------------------------------------------------------- */
//#include <cassert>
//#include <iostream>
#include <tuple>
#include <utility>
/* -------------------------------------------------------------------------- */

#ifndef __AKANTU_AKA_ITERATORS_HH__
#define __AKANTU_AKA_ITERATORS_HH__

namespace akantu {

namespace tuple {
  /* ------------------------------------------------------------------------ */
  namespace details {
    template <size_t N> struct Foreach {
      template <class F, class Tuple>
      static inline decltype(auto) transform_forward(F && func,
                                                     Tuple && tuple) {
        return std::tuple_cat(
            Foreach<N - 1>::transform_forward(std::forward<F>(func),
                                              std::forward<Tuple>(tuple)),
            std::forward_as_tuple(std::forward<F>(func)(
                std::get<N - 1>(std::forward<Tuple>(tuple)))));
      }

      template <class F, class Tuple>
      static inline decltype(auto) transform(F && func, Tuple && tuple) {
        return std::tuple_cat(
            Foreach<N - 1>::transform(std::forward<F>(func),
                                      std::forward<Tuple>(tuple)),
            std::make_tuple(std::forward<F>(func)(
                std::get<N - 1>(std::forward<Tuple>(tuple)))));
      }

      template <class F, class Tuple>
      static inline void foreach (F && func, Tuple && tuple) {
        Foreach<N - 1>::foreach (std::forward<F>(func),
                                 std::forward<Tuple>(tuple));
        std::forward<F>(func)(std::get<N - 1>(std::forward<Tuple>(tuple)));
      }

      template <class Tuple> static inline bool equal(Tuple && a, Tuple && b) {
        if (not(std::get<N - 1>(std::forward<Tuple>(a)) ==
                std::get<N - 1>(std::forward<Tuple>(b))))
          return false;
        return Foreach<N - 1>::equal(std::forward<Tuple>(a),
                                     std::forward<Tuple>(b));
      }
    };

    /* ------------------------------------------------------------------------
     */
    template <> struct Foreach<1> {
      template <class F, class Tuple>
      static inline decltype(auto) transform_forward(F && func,
                                                     Tuple && tuple) {
        return std::forward_as_tuple(
            std::forward<F>(func)(std::get<0>(std::forward<Tuple>(tuple))));
      }

      template <class F, class Tuple>
      static inline decltype(auto) transform(F && func, Tuple && tuple) {
        return std::make_tuple(
            std::forward<F>(func)(std::get<0>(std::forward<Tuple>(tuple))));
      }

      template <class F, class Tuple>
      static inline void foreach (F && func, Tuple && tuple) {
        std::forward<F>(func)(std::get<0>(std::forward<Tuple>(tuple)));
      }

      template <class Tuple> static inline bool equal(Tuple && a, Tuple && b) {
        return std::get<0>(std::forward<Tuple>(a)) ==
               std::get<0>(std::forward<Tuple>(b));
      }
    };
  } // namespace details
  /* ------------------------------------------------------------------------ */
  template <class Tuple> bool are_equal(Tuple && a, Tuple && b) {
    return details::Foreach<std::tuple_size<std::decay_t<Tuple>>::value>::equal(
        std::forward<Tuple>(a), std::forward<Tuple>(b));
  }

  template <class F, class Tuple> void foreach (F && func, Tuple && tuple) {
    details::Foreach<std::tuple_size<std::decay_t<Tuple>>::value>::foreach (
        std::forward<F>(func), std::forward<Tuple>(tuple));
  }

  template <class F, class Tuple>
  decltype(auto) transform_forward(F && func, Tuple && tuple) {
    return details::Foreach<std::tuple_size<std::decay_t<Tuple>>::value>::
        transform_forward(std::forward<F>(func), std::forward<Tuple>(tuple));
  }

  template <class F, class Tuple>
  decltype(auto) transform(F && func, Tuple && tuple) {
    return details::Foreach<std::tuple_size<std::decay_t<Tuple>>::value>::
        transform(std::forward<F>(func), std::forward<Tuple>(tuple));
  }
} // namespace tuple

namespace iterators {
  namespace details {
    struct dereference_iterator {
      template <class Iter> decltype(auto) operator()(Iter & it) const {
        return std::forward<decltype(*it)>(*it);
      }
    };

    struct increment_iterator {
      template <class Iter> void operator()(Iter & it) const { ++it; }
    };

    struct begin_container {
      template <class Container>
      decltype(auto) operator()(Container && cont) const {
        return std::forward<Container>(cont).begin();
      }
    };

    struct end_container {
      template <class Container>
      decltype(auto) operator()(Container && cont) const {
        return std::forward<Container>(cont).end();
      }
    };
  } // namespace details

  /* ------------------------------------------------------------------------ */
  template <class... Iterators> class ZipIterator {
  private:
    using tuple_t = std::tuple<Iterators...>;

  public:
    explicit ZipIterator(tuple_t iterators) : iterators(std::move(iterators)) {}

    decltype(auto) operator*() {
      return tuple::transform_forward(details::dereference_iterator(),
                                      iterators);
    }

    ZipIterator & operator++() {
      tuple::foreach (details::increment_iterator(), iterators);
      return *this;
    }

    bool operator==(const ZipIterator & other) const {
      return tuple::are_equal(iterators, other.iterators);
    }

    bool operator!=(const ZipIterator & other) const {
      return not operator==(other);
    }

  private:
    tuple_t iterators;
  };
} // namespace iterators

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
template <class... Iterators>
decltype(auto) zip_iterator(std::tuple<Iterators...> && iterators_tuple) {
  auto zip = iterators::ZipIterator<Iterators...>(
      std::forward<decltype(iterators_tuple)>(iterators_tuple));
  return zip;
}

/* -------------------------------------------------------------------------- */
namespace containers {
  template <class... Containers> class ZipContainer {
    using containers_t = std::tuple<Containers...>;

  public:
    explicit ZipContainer(Containers &&... containers)
        : containers(std::forward<Containers>(containers)...) {}

    decltype(auto) begin() const {
      return zip_iterator(
          tuple::transform(iterators::details::begin_container(),
                           std::forward<containers_t>(containers)));
    }

    decltype(auto) end() const {
      return zip_iterator(
          tuple::transform(iterators::details::end_container(),
                           std::forward<containers_t>(containers)));
    }

    decltype(auto) begin() {
      return zip_iterator(
          tuple::transform(iterators::details::begin_container(),
                           std::forward<containers_t>(containers)));
    }

    decltype(auto) end() {
      return zip_iterator(
          tuple::transform(iterators::details::end_container(),
                           std::forward<containers_t>(containers)));
    }

  private:
    containers_t containers;
  };
} // namespace containers

/* -------------------------------------------------------------------------- */
template <class... Containers> decltype(auto) zip(Containers &&... conts) {
  return containers::ZipContainer<Containers...>(
      std::forward<Containers>(conts)...);
}

/* -------------------------------------------------------------------------- */
/* Arange                                                                     */
/* -------------------------------------------------------------------------- */
namespace iterators {
  template <class T> class ArangeIterator {
  public:
    using value_type = T;
    using pointer = T *;
    using reference = T &;
    using iterator_category = std::input_iterator_tag;

    constexpr ArangeIterator(T value, T step) : value(value), step(step) {}
    constexpr ArangeIterator(const ArangeIterator &) = default;

    constexpr ArangeIterator & operator++() {
      value += step;
      return *this;
    }

    constexpr const T & operator*() const { return value; }

    constexpr bool operator==(const ArangeIterator & other) const {
      return (value == other.value) and (step == other.step);
    }

    constexpr bool operator!=(const ArangeIterator & other) const {
      return not operator==(other);
    }

  private:
    T value{0};
    const T step{1};
  };
} // namespace iterators

namespace containers {
  template <class T> class ArangeContainer {
  public:
    using iterator = iterators::ArangeIterator<T>;

    constexpr ArangeContainer(T start, T stop, T step = 1)
        : start(start), stop((stop - start) % step == 0
                                 ? stop
                                 : start + (1 + (stop - start) / step) * step),
          step(step) {}
    explicit constexpr ArangeContainer(T stop) : ArangeContainer(0, stop, 1) {}

    constexpr T operator[](size_t i) {
      T val = start + i * step;
      assert(val < stop && "i is out of range");
      return val;
    }

    constexpr size_t size() { return (stop - start) / step; }

    constexpr iterator begin() { return iterator(start, step); }
    constexpr iterator end() { return iterator(stop, step); }

  private:
    const T start{0}, stop{0}, step{1};
  };
} // namespace containers

template <class T> inline decltype(auto) arange(T stop) {
  return containers::ArangeContainer<T>(stop);
}

template <class T>
inline constexpr decltype(auto) arange(T start, T stop, T step = 1) {
  return containers::ArangeContainer<T>(start, stop, step);
}

template <class Container>
inline constexpr decltype(auto) enumerate(Container && container,
                                          size_t start_ = 0) {
  auto stop = std::forward<Container>(container).size();
  decltype(stop) start = start_;
  return zip(arange(start, stop), std::forward<Container>(container));
}

} // namespace akantu

#endif /* __AKANTU_AKA_ITERATORS_HH__ */
