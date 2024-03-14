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

#include "mitsuba/core/platform.h"
#include <unordered_map>

#include <drjit/morton.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/records.h>
#include <mitsuba/render/scene.h>
#include <nanothread/nanothread.h>

NAMESPACE_BEGIN(mitsuba)

/// Used to store a white noise position sample along with its cell ID
template <typename Float, typename Spectrum>
class MI_EXPORT_LIB BlueNoiseSampler {
public:
    MI_IMPORT_TYPES(Scene, Shape, Sampler)

    /**
     * \brief Generate a point set with blue noise properties
     *
     * Based on the paper "Parallel Poisson Disk Sampling with
     * Spectrum Analysis on Surfaces" by John Bowers, Rui Wang,
     * Li-Yi Wei and David Maletz.
     *
     * \param scene
     *    A pointer to the underlying scene
     * \param shapes
     *    A list of input shapes on which samples should be placed
     * \param radius
     *    The Poisson radius of the point set to be generated
     * \param target
     *    A position sample vector (which will be populated with the result)
     * \param sa
     *    Will be used to return the combined surface area of the shapes
     * \param aabb
     *    Will be used to store an an axis-aligned box containing all samples
     * \param data
     *    Custom pointer that will be sent along with progress messages
     *    (usually contains a pointer to the \ref RenderJob instance)
     */

    void blueNoisePointSet(const Scene *scene,
                           const std::vector<Shape *> &shapes, uint32_t seed,
                           Float radius, std::vector<PositionSample3f> *target,
                           Float &sa, BoundingBox3f &aabb) {
        int kmax = 8; /* Perform 8 trial runs */

        uint32_t n_threads = (uint32_t) Thread::thread_count();

        /* Create a random number generator for each thread */
        ref<Sampler> rootSampler = new Sampler();
        std::vector<BoundingBox3f> t_aabb(n_threads);
        for (int i = 0; i < n_threads; ++i) {
            t_aabb[i].reset();
        }

        DiscreteDistribution<Float> areaDistr;
        std::vector<int> shapeMap(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i) {
            shapeMap[i] = -1;
            for (size_t j = 0; j < scene->getShapes().size(); ++j) {
                if (scene->getShapes()[j].get() == shapes[i]) {
                    shapeMap[i] = (int) j;
                    break;
                }
            }
            Assert(shapeMap[i] != -1);
            areaDistr.append(shapes[i]->getSurfaceArea());
        }
        areaDistr.normalize();
        sa = areaDistr.getSum();

        /* Start with a fairly dense set of white noise points */
        uint32_t nsamples = dr::ceil2int(15 * sa / (M_PI * radius * radius));

        Log(m_log_level,
            "Creating a blue noise point set (radius=%f, "
            "surface area = %f)",
            radius, sa);
        Log(m_log_level, "  phase 1: creating dense white noise (%i samples)",
            nsamples);
        ref<Timer> timer = new Timer();
        timer->begin_stage("phase 1");
        std::vector<UniformSample> samples(nsamples);

        // TODO check if the grain size is correct
        uint32_t grain_size = std::max(nsamples / (4 * n_threads), 1u);

        ThreadEnvironment env;
        dr::parallel_for(
            dr::blocked_range<uint32_t>(0, nsamples, grain_size),
            [&](const dr::blocked_range<uint32_t> &range) {
                ScopedSetThreadEnvironment set_env(env);

                // Fork a non-overlapping sampler for the current
                // worker
                ref<Sampler> sampler = rootSampler->fork();

                for (uint32_t i = range.begin(); i != range.end(); ++i) {
                    sampler->seed(seed + i);
                    Point2f sample(sampler->next_2d(true));
                    int shapeIndex      = (int) areaDistr.sampleReuse(sample.x);
                    Shape *shape        = shapes[shapeIndex];
                    PositionSample3f ps = shape->sample_position(0, sample);

                    samples[i] =
                        UniformSample(ps.p, ps.n, 0, shapeMap[shapeIndex]);
                    t_aabb[i].expand(ps.p);
                }
            });
        timer->end_stage();
        timer->reset();

        Float cellWidth    = radius / std::sqrt(3.0f),
              invCellWidth = 1.0f / cellWidth;

        aabb.reset();
        for (int i = 0; i < n_threads; ++i)
            aabb.expand(t_aabb[i]);
        Vector extents = aabb.extents();

        Vector3i cellCount;
        for (int i = 0; i < 3; ++i)
            cellCount[i] = std::max(1, dr::ceil2int(extents[i] * invCellWidth));

        Log(m_log_level, "  phase 2: computing cell indices ..");
        timer->begin_stage("phase 2");

        dr::parallel_for(
            dr::blocked_range<uint32_t>(0, nsamples, grain_size),
            [&](const dr::blocked_range<uint32_t> &range) {
                ScopedSetThreadEnvironment set_env(env);
                for (uint32_t i = range.begin(); i != range.end(); ++i) {
                    Vector rel = samples[i].p - aabb.min;
                    Vector3i idx;
                    for (int j = 0; j < 3; ++j)
                        idx[j] = std::min((int) (rel[j] * invCellWidth),
                                          cellCount[j] - 1);
                    samples[i].cellID =
                        idx[0] + (int64_t) cellCount[0] *
                                     (idx[1] + idx[2] * (int64_t) cellCount[1]);
                }
            });
        timer->end_stage();
        timer->reset();

        Log(m_log_level, "  phase 3: sorting ..");
        timer->begin_stage("phase 3");

        std::sort(samples.begin(), samples.end(), CellIDOrdering());

        timer->end_stage();
        timer->reset();

        Log(m_log_level,
            "  phase 4: establishing valid cells and phase groups ..");
        timer->begin_stage("phase 4");

        std::unordered_map<int64_t, Cell> cells(samples.size());
        std::vector<std::vector<int64_t>> phaseGroups(27);
        for (int i = 0; i < 27; ++i)
            phaseGroups[i].reserve(cells.size() / 27);

        int64_t last = -1;
        for (uint32_t i = 0; i < nsamples; ++i) {
            int64_t id = samples[i].cellID;
            if (id != last) {
                cells[id] = Cell(i);
                last      = id;

                /* Schedule this cell wrt. the corresponding phase group */
                int64_t tmp = id;
                int64_t z   = tmp / (cellCount[0] * cellCount[1]);
                tmp -= z * (cellCount[0] * cellCount[1]);
                int64_t y   = tmp / cellCount[0];
                int64_t x   = tmp - y * cellCount[0];
                int phaseID = x % 3 + (y % 3) * 3 + (z % 3) * 9;
                phaseGroups[phaseID].push_back(id);
            }
        }
        timer->end_stage();
        timer->reset();
        Log(m_log_level, "    got %i cells, avg. samples per cell: %f",
            (int) cells.size(), samples.size() / (Float) cells.size());

        Log(m_log_level, "  phase 5: parallel sampling ..");
        for (int trial = 0; trial < kmax; ++trial) {
            for (int phase = 0; phase < 27; ++phase) {
                const std::vector<int64_t> &phaseGroup = phaseGroups[phase];

                // TODO check grain size
                int64_t grain_size =
                    std::max(phaseGroup.size() / (4 * n_threads), 1ul);

                dr::parallel_for(
                    dr::blocked_range<int64_t>(0, phaseGroup.size(),
                                               grain_size),
                    [&](const dr::blocked_range<int64_t> &range) {
                        ScopedSetThreadEnvironment set_env(env);
                        for (int64_t i = range.begin(); i != range.end(); ++i) {
                            int64_t cellID = phaseGroup[i];
                            Cell &cell     = cells[cellID];
                            int arrayIndex = cell.firstIndex + trial;

                            if (arrayIndex >= (int) samples.size() ||
                                samples[arrayIndex].cellID != cellID ||
                                cell.sample != -1)
                                continue;

                            const UniformSample &sample = samples[arrayIndex];

                            bool conflict = false;

                            for (int z = -2; z < 3; ++z) {
                                for (int y = -2; y < 3; ++y) {
                                    for (int x = -2; x < 3; ++x) {
                                        int64_t neighborCellID =
                                            cellID + x +
                                            (int64_t) cellCount[0] *
                                                (y +
                                                 z * (int64_t) cellCount[1]);

                                        auto &it = cells.find(neighborCellID);

                                        if (it != cells.end()) {
                                            const Cell &neighbor = it->second;
                                            if (neighbor.sample != -1) {
                                                const UniformSample &sample2 =
                                                    samples[neighbor.sample];

                                                if ((sample.p - sample2.p)
                                                        .lengthSquared() <
                                                    radius * radius) {
                                                    conflict = true;
                                                    goto bailout;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                        bailout:
                            if (!conflict)
                                cell.sample = arrayIndex;
                        }
                    });
            }
        }
        timer->end_stage();
        timer->reset();

        for (const auto &it = cells.begin(); it != cells.end(); ++it) {
            const Cell &cell = it->second;
            if (cell.sample == -1)
                continue;
            const UniformSample &sample = samples[cell.sample];
            target->put(PositionSample(sample.p, sample.n, sample.shapeIndex));
        }

        Log(m_log_level, "Sampling finished (obtained %i blue noise samples)",
            (int) target->size());
    }

private:
    struct UniformSample {
        Point3f p;
        Vector3f n;
        int64_t cellID;
        int shapeIndex;

        inline UniformSample() {}
        inline UniformSample(const Point3f &p, const Vector3f &n, int cellID,
                             int shapeIndex)
            : p(p), n(n), cellID(cellID), shapeIndex(shapeIndex) {}
    };

    /// Functor for sorting 'UniformSample' instances with respect to their cell
    /// ID
    struct CellIDOrdering {
    public:
        inline bool operator()(const UniformSample &s1,
                               const UniformSample &s2) const {
            return s1.cellID < s2.cellID;
        }
    };

    /// Stores the first UniformSample that falls in this cell and the chosen
    /// one (if any)
    struct Cell {
        int firstIndex;
        int sample;

        inline Cell() {}
        inline Cell(int firstIndex, int sample = -1)
            : firstIndex(firstIndex), sample(sample) {}
    };

    LogLevel m_log_level = Info;
};

MI_EXTERN_CLASS(BlueNoiseSampler)

NAMESPACE_END(mitsuba)
