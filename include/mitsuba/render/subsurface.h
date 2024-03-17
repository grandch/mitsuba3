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

#include <drjit/vcall.h>
#include <mitsuba/core/object.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/traits.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>


NAMESPACE_BEGIN(mitsuba)

/**
 * \brief subsurface scattering interface
 *
 */
template <typename Float, typename Spectrum>
class MI_EXPORT_LIB Subsurface : public Object {
public:
    MI_IMPORT_TYPES(Sampler, Scene, Shape);

    /// Get the exitant radiance for a point on the surface
    virtual Spectrum Lo(const Scene *scene, Sampler *sampler,
                        const SurfaceInteraction3f &si, const Vector3f &d,
                        const UInt32 depth = 0) const = 0;

    /// Return the list of shapes associated with this subsurface integrator
    inline const std::vector<Shape *> shapes() const { return m_shapes; }

    /**
     * \brief Possibly perform a pre-process task.
     *
     * The last three parameters are resource IDs of the associated scene,
     * camera and sample generator, which have been made available to all
     * local and remote workers.
     */
    virtual bool preprocess(const Scene *scene, Sampler *sampler) = 0;

    DRJIT_VCALL_REGISTER(Float, mitsuba::Subsurface)

    MI_DECLARE_CLASS()
protected:
    Subsurface();
    Subsurface(const Properties &props);
    virtual ~Subsurface();

protected:
    std::vector<ref<Shape>> m_shapes;
};


MI_EXTERN_CLASS(Subsurface)
NAMESPACE_END(mitsuba)

DRJIT_VCALL_TEMPLATE_BEGIN(mitsuba::Subsurface)
    DRJIT_VCALL_METHOD(Lo)
DRJIT_VCALL_TEMPLATE_END(mitsuba::Subsurface)
