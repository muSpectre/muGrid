/**
 * @file   iterators.hh
 *
 * @author Nicolas Richart (original akantu version)
 * @author Lars Pastewka (C++20 modernization)
 *
 * @date creation  Wed Jul 19 2017
 *
 * @brief iterator interfaces - modernized with C++20 features
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
 * Above block was left intact as in akantu. µGrid exercises the
 * right to redistribute and modify the code below
 *
 */

#ifndef SRC_LIBMUGRID_ITERATORS_HH_
#define SRC_LIBMUGRID_ITERATORS_HH_

#include <ranges>
#include <tuple>
#include <utility>

namespace akantu {

    namespace tuple {
        namespace details {
            //! Create tuple without decay (preserves references)
            template <typename... Ts>
            decltype(auto) make_tuple_no_decay(Ts &&... args) {
                return std::tuple<Ts...>(std::forward<Ts>(args)...);
            }
        }  // namespace details

        //! Check if ALL corresponding elements in two tuples are not equal
        //! For zip iterators: returns false when ANY iterator reaches its end
        //! (stops iteration at shortest container)
        //! Uses C++17 fold expression
        template <class Tuple, size_t... Is>
        bool are_not_equal_impl(Tuple && a, Tuple && b,
                                std::index_sequence<Is...>) {
            // Return true only if ALL elements are different
            // Return false if ANY element is equal (i.e., any iterator at end)
            return ((std::get<Is>(std::forward<Tuple>(a)) !=
                     std::get<Is>(std::forward<Tuple>(b))) &&
                    ...);
        }

        template <class Tuple>
        bool are_not_equal(Tuple && a, Tuple && b) {
            return are_not_equal_impl(
                std::forward<Tuple>(a), std::forward<Tuple>(b),
                std::make_index_sequence<
                    std::tuple_size_v<std::decay_t<Tuple>>>{});
        }

        //! Apply function to each element of tuple (C++17 fold expression)
        template <class F, class Tuple, size_t... Is>
        void foreach_impl(F && func, Tuple && tuple,
                          std::index_sequence<Is...>) {
            (std::forward<F>(func)(std::get<Is>(std::forward<Tuple>(tuple))),
             ...);
        }

        template <class F, class Tuple>
        void foreach_(F && func, Tuple && tuple) {
            foreach_impl(std::forward<F>(func), std::forward<Tuple>(tuple),
                         std::make_index_sequence<
                             std::tuple_size_v<std::decay_t<Tuple>>>{});
        }

        //! Transform each element of tuple with function
        template <class F, class Tuple, size_t... Is>
        decltype(auto) transform_impl(F && func, Tuple && tuple,
                                      std::index_sequence<Is...>) {
            return details::make_tuple_no_decay(std::forward<F>(func)(
                std::get<Is>(std::forward<Tuple>(tuple)))...);
        }

        template <class F, class Tuple>
        decltype(auto) transform(F && func, Tuple && tuple) {
            return transform_impl(
                std::forward<F>(func), std::forward<Tuple>(tuple),
                std::make_index_sequence<
                    std::tuple_size_v<std::decay_t<Tuple>>>{});
        }
    }  // namespace tuple

    /* ------------------------------------------------------------------------
     */
    namespace iterators {
        //! Iterator for zip - holds tuple of iterators
        template <class... Iterators>
        class ZipIterator {
           private:
            using tuple_t = std::tuple<Iterators...>;

           public:
            explicit ZipIterator(tuple_t iterators)
                : iterators(std::move(iterators)) {}

            decltype(auto) operator*() {
                return tuple::transform(
                    [](auto && it) -> decltype(auto) { return *it; },
                    iterators);
            }

            ZipIterator & operator++() {
                tuple::foreach_([](auto && it) { ++it; }, iterators);
                return *this;
            }

            bool operator==(const ZipIterator & other) const {
                return !tuple::are_not_equal(iterators, other.iterators);
            }

            bool operator!=(const ZipIterator & other) const {
                return tuple::are_not_equal(iterators, other.iterators);
            }

           private:
            tuple_t iterators;
        };
    }  // namespace iterators

    //! Create zip iterator from tuple of iterators
    template <class... Iterators>
    decltype(auto) zip_iterator(std::tuple<Iterators...> && iterators_tuple) {
        return iterators::ZipIterator<Iterators...>(
            std::forward<decltype(iterators_tuple)>(iterators_tuple));
    }

    /* ------------------------------------------------------------------------
     */
    namespace containers {
        //! Container adapter for zip iteration
        template <class... Containers>
        class ZipContainer {
            using containers_t = std::tuple<Containers...>;

           public:
            explicit ZipContainer(Containers &&... containers)
                : containers(std::forward<Containers>(containers)...) {}

            decltype(auto) begin() const {
                return zip_iterator(
                    tuple::transform([](auto && c) { return c.begin(); },
                                     std::forward<containers_t>(containers)));
            }

            decltype(auto) end() const {
                return zip_iterator(
                    tuple::transform([](auto && c) { return c.end(); },
                                     std::forward<containers_t>(containers)));
            }

            decltype(auto) begin() {
                return zip_iterator(
                    tuple::transform([](auto && c) { return c.begin(); },
                                     std::forward<containers_t>(containers)));
            }

            decltype(auto) end() {
                return zip_iterator(
                    tuple::transform([](auto && c) { return c.end(); },
                                     std::forward<containers_t>(containers)));
            }

           private:
            containers_t containers;
        };
    }  // namespace containers

    /**
     * Emulates Python's zip() - iterate over multiple containers in parallel
     */
    template <class... Containers>
    decltype(auto) zip(Containers &&... conts) {
        return containers::ZipContainer<Containers...>(
            std::forward<Containers>(conts)...);
    }

    /* ------------------------------------------------------------------------
     */

    /**
     * Emulates Python's enumerate() - iterate with index
     * Uses C++20 std::views::iota for the index range
     */
    template <class Container>
    inline auto enumerate(Container && container, size_t start = 0) {
        // Get size before forwarding to avoid double-move
        auto stop = start + container.size();
        return zip(std::views::iota(start, stop),
                   std::forward<Container>(container));
    }

}  // namespace akantu

#endif  // SRC_LIBMUGRID_ITERATORS_HH_
