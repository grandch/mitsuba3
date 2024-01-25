#pragma once

#include <mitsuba/core/object.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/traits.h>
#include <mitsuba/render/fwd.h>
#include <drjit/vcall.h>

NAMESPACE_BEGIN(mitsuba)

/**
 * \brief subsurface scattering interface
 *
 */
template <typename Float, typename Spectrum>
class MI_EXPORT_LIB Subsurface : public Object {
public:
    MI_IMPORT_TYPES(Sampler, Scene);

    /// Get the exitant radiance for a point on the surface
    virtual Spectrum Lo(const Scene *scene, Sampler *sampler,
                        const SurfaceInteraction3f &si, const Vector3f &d,
                        int depth = 0) const = 0;

    /// Return a human-readable representation of the Subsurface
    std::string to_string() const override = 0;

    DRJIT_VCALL_REGISTER(Float, mitsuba::Subsurface)

    MI_DECLARE_CLASS()
protected:
    Subsurface();
    Subsurface(const Properties &props);
    virtual ~Subsurface();

protected:
    // TODO: check what should go here
    // std::vector<Shape *> m_shapes;
};

MI_EXTERN_CLASS(Subsurface)
NAMESPACE_END(mitsuba)
