/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once

#include <mitsuba/core/bbox.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/util.h>
#include <mitsuba/render/fwd.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief Irradiance single-reference static octree
 *
 * This class is currently used to implement BSSRDF evaluation
 * with irradiance point clouds.
 *
 * The \c Item template parameter must implement a function
 * named <tt>getPosition()</tt> that returns a \ref Point.
 *
 */
template <typename Float, typename Spectrum>
class MI_EXPORT_LIB IrradianceOctree : public Object {
public:
    MI_IMPORT_TYPES()

    using BoundingBox = BoundingBox<Point<dr::scalar_t<Float>, 3>>;
    using Point       = typename BoundingBox::Point;
    using Vector      = typename BoundingBox::Vector;
    using Scalar      = dr::value_t<Vector>;
    using SizedInt    = dr::uint_array_t<Scalar>;
    struct OctreeNode {
        bool leaf : 1;
        IrradianceSample3f data;

        union {
            struct {
                OctreeNode *children[8];
            };

            struct {
                uint32_t offset;
                uint32_t count;
            };
        };

        ~OctreeNode() {
            if (!leaf) {
                for (int i = 0; i < 8; ++i) {
                    if (children[i])
                        delete children[i];
                }
            }
        }
    };

    /**
     * \brief Create a new octree
     * \param maxDepth
     *     Maximum tree depth (24 by default)
     * \param maxItems
     *     Maximum items per interior node (8 by default)
     *
     * By default, the maximum tree depth is set to 16
     */
    inline IrradianceOctree(const BoundingBox &aabb, Float solidAngleThreshold,
                            std::vector<IrradianceSample3f> &records,
                            uint32_t maxDepth = 24, uint32_t maxItems = 8)
        : m_aabb(aabb), m_maxDepth(maxDepth), m_maxItems(maxItems),
          m_root(nullptr), m_solidAngleThreshold(solidAngleThreshold) {
        m_items.swap(records);
        build();
        propagate(m_root);
    }

    /// Release all memory
    ~IrradianceOctree() {
        if (m_root)
            delete m_root;
    }

    void build() {
        Log(m_log_level, "Building an octree over %d data points",
            m_items.size());

        std::vector<uint32_t> perm(m_items.size()), temp(m_items.size());

        for (uint32_t i = 0; i < m_items.size(); ++i)
            perm[i] = i;

        /* Build the octree and compute a suitable permutation of the
         * elements
         */
        m_root = build(m_aabb, 0, &perm[0], &temp[0], &perm[0],
                       &perm[0] + m_items.size());

        /* Apply the permutation */
        permute_inplace(&m_items[0], perm);
    }

    /// Query the octree using a customizable functor, while representatives for
    /// distant nodes
    template <typename QueryType>
    inline void perform_query(QueryType &query) const {
        perform_query(m_aabb, m_root, query);
    }

    /// Return the log level of octree status messages
    LogLevel log_level() const { return m_log_level; }

    /// Return the log level of octree status messages
    void set_log_level(LogLevel level) { m_log_level = level; }

    MI_DECLARE_CLASS()
protected:
    /// Return the AABB for a child of the specified index
    inline BoundingBox childBounds(int child, const BoundingBox &nodeAABB,
                                   const Point &center) const {
        BoundingBox childAABB;
        childAABB.min.x() = (child & 4) ? center.x() : nodeAABB.min.x();
        childAABB.max.x() = (child & 4) ? nodeAABB.max.x() : center.x();
        childAABB.min.y() = (child & 2) ? center.y() : nodeAABB.min.y();
        childAABB.max.y() = (child & 2) ? nodeAABB.max.y() : center.y();
        childAABB.min.z() = (child & 1) ? center.z() : nodeAABB.min.z();
        childAABB.max.z() = (child & 1) ? nodeAABB.max.z() : center.z();
        return childAABB;
    }

    OctreeNode *build(const BoundingBox &aabb, uint32_t depth, uint32_t *base,
                      uint32_t *temp, uint32_t *start, uint32_t *end) {
        if (start == end) {
            return nullptr;
        } else if ((uint32_t) (end - start) < m_maxItems ||
                   depth > m_maxDepth) {
            OctreeNode *result = new OctreeNode();
            result->count      = (uint32_t) (end - start);
            result->offset     = (uint32_t) (start - base);
            result->leaf       = true;
            return result;
        }

        Point center = aabb.center();
        uint32_t nestedCounts[8];
        memset(nestedCounts, 0, sizeof(uint32_t) * 8);

        /* Label all items */
        for (uint32_t *it = start; it != end; ++it) {
            IrradianceSample3f &item = m_items[*it];
            const Point3f &p         = item.p;

            uint8_t label = 0;
            if (dr::any(p.x() > center.x()))
                label |= 4;
            if (dr::any(p.y() > center.y()))
                label |= 2;
            if (dr::any(p.z() > center.z()))
                label |= 1;

            BoundingBox bounds = childBounds(label, aabb, center);
            Assert(dr::any(bounds.contains(p)));

            item.label = label;
            nestedCounts[label]++;
        }

        uint32_t nestedOffsets[9];
        nestedOffsets[0] = 0;
        for (int i = 1; i <= 8; ++i)
            nestedOffsets[i] = nestedOffsets[i - 1] + nestedCounts[i - 1];

        /* Sort by label */
        for (uint32_t *it = start; it != end; ++it) {
            int offset   = nestedOffsets[m_items[*it].label]++;
            temp[offset] = *it;
        }
        memcpy(start, temp, (end - start) * sizeof(uint32_t));

        /* Recurse */
        OctreeNode *result = new OctreeNode();
        for (int i = 0; i < 8; i++) {
            BoundingBox bounds = childBounds(i, aabb, center);

            uint32_t *it = start + nestedCounts[i];
            result->children[i] =
                build(bounds, depth + 1, base, temp, start, it);
            start = it;
        }

        result->leaf = false;

        return result;
    }

    /// Propagate irradiance approximations througout the tree
    void propagate(OctreeNode *node) {
        IrradianceSample3f &repr = node->data;

        /* Initialize the cluster values */
        repr.E          = Spectrum(0.0f);
        repr.area       = 0.0f;
        repr.p          = Point(0.0f, 0.0f, 0.0f);
        Float weightSum = 0.0f;

        if (node->leaf) {
            /* Inner node */
            for (uint32_t i = 0; i < node->count; ++i) {
                const IrradianceSample3f &sample = m_items[i + node->offset];
                repr.E += sample.E * sample.area;
                repr.area += sample.area;
                Float weight = luminance(sample.E, Wavelength()) * sample.area;
                repr.p += sample.p * weight;
                weightSum += weight;
            }
        } else {
            /* Inner node */
            for (int i = 0; i < 8; i++) {
                OctreeNode *child = node->children[i];
                if (!child)
                    continue;
                propagate(child);
                repr.E += child->data.E * child->data.area;
                repr.area += child->data.area;
                Float weight =
                    luminance(child->data.E, Wavelength()) * child->data.area;
                repr.p += child->data.p * weight;
                weightSum += weight;
            }
        }
        if (repr.area != 0)
            repr.E /= repr.area;
        if (weightSum != 0)
            repr.p /= weightSum;
    }

    /// Query the octree using a customizable functor, while representatives for
    /// distant nodes
    template <typename QueryType>
    void perform_query(const BoundingBox &aabb, OctreeNode *node,
                       QueryType &query) const {
        /* Compute the approximate solid angle subtended by samples within this
         * node */
        Float approxSolidAngle =
            node->data.area / dr::squared_norm(query.p - node->data.p);

        /* Use the representative if this is a distant node */
        if (dr::any(!aabb.contains(query.p) &&
                    approxSolidAngle < m_solidAngleThreshold)) {
            query(node->data);
        } else {
            if (node->leaf) {
                for (uint32_t i = 0; i < node->count; ++i)
                    query(m_items[node->offset + i]);
            } else {
                Point center = aabb.center();
                for (int i = 0; i < 8; i++) {
                    if (!node->children[i])
                        continue;

                    BoundingBox childAABB = childBounds(i, aabb, center);
                    perform_query(childAABB, node->children[i], query);
                }
            }
        }
    }

    inline IrradianceOctree() : m_root(nullptr) {}

private:
    /**
     * \brief Apply an arbitrary permutation to an array in linear time
     *
     * This algorithm is based on Donald Knuth's book
     * "The Art of Computer Programming, Volume 3: Sorting and Searching"
     * (1st edition, section 5.2, page 595)
     *
     * Given a permutation and an array of values, it applies the permutation
     * in linear time without requiring additional memory. This is based on
     * the fact that each permutation can be decomposed into a disjoint set
     * of permutations, which can then be applied individually.
     *
     * \param data
     *     Pointer to the data that should be permuted
     * \param perm
     *     Input permutation vector having the same size as \c data. After
     *     the function terminates, this vector will be set to the
     *     identity permutation.
     */
    template <typename DataType, typename IndexType>
    void permute_inplace(DataType *data, std::vector<IndexType> &perm) {
        for (size_t i = 0; i < perm.size(); i++) {
            if (perm[i] != i) {
                /* The start of a new cycle has been found. Save
                   the value at this position, since it will be
                   overwritten */
                IndexType j     = (IndexType) i;
                DataType curval = data[i];

                do {
                    /* Shuffle backwards */
                    IndexType k = perm[j];
                    data[j]     = data[k];

                    /* Also fix the permutations on the way */
                    perm[j] = j;
                    j       = k;

                    /* Until the end of the cycle has been found */
                } while (perm[j] != i);

                /* Fix the final position with the saved value */
                data[j] = curval;
                perm[j] = j;
            }
        }
    }

    BoundingBox m_aabb;
    std::vector<IrradianceSample3f> m_items;
    uint32_t m_maxDepth;
    uint32_t m_maxItems;
    OctreeNode *m_root;
    Float m_solidAngleThreshold;
    LogLevel m_log_level = Debug;
};

MI_IMPLEMENT_CLASS_VARIANT(IrradianceOctree, Object)

NAMESPACE_END(mitsuba)
