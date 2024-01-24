#include <mitsuba/core/properties.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/render/mesh.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class Laser final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags, m_medium, m_to_world)
    MI_IMPORT_TYPES(Scene, Texture)

    Laser(const Properties &props) : Base(props) {    
        m_position = props.get<ScalarPoint3f>("position");
        m_radiance = props.texture_d65<Texture>("radiance", 1.f);

        if (props.has_property("direction")) {
            if (props.has_property("to_world"))
                Throw("Only one of the parameters 'direction' and 'to_world' "
                      "can be specified at the same time!'");

            ScalarVector3f direction(dr::normalize(props.get<ScalarVector3f>("direction")));
            auto [up, unused] = coordinate_system(direction);

            m_to_world = ScalarTransform4f::look_at(0.0f, ScalarPoint3f(direction), up);
            dr::make_opaque(m_to_world);
        }

        m_flags = EmitterFlags::Surface | EmitterFlags::DeltaDirection;
        dr::set_attr(this, "flags", m_flags);
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_object("radiance", m_radiance.get(), +ParamFlags::Differentiable);
    }

    // std::pair<Ray3f, Spectrum> sample_ray(Float /*time*/, Float /*wavelength_sample*/,
    //                                       const Point2f &/*sample2*/,
    //                                       const Point2f & /*sample3*/,
    //                                       Mask /*active*/) const override {
    //     // 1. Sample spatial component
    //     // 2. Directional component is the normal vector at that position.
    //     // 3. Sample spectral component
    //     // SurfaceInteraction3f si(ps, dr::zeros<Wavelength>());
    //     // auto [wavelength, wav_weight] = sample_wavelengths(si, wavelength_sample, active);
    //     // si.time = time;
    //     // si.wavelengths = wavelength;

    //     // return { si.spawn_ray(d), m_area * wav_weight };
    //     return { 0, 0 };
    // }

    /**
     * Current strategy: don't try to connect this emitter
     * observed from the reference point `it`, since it's
     * unlikely to correspond to the surface normal (= the emission
     * direction).
     *
     * TODO: instead, we could try and find the orthogonal projection
     *       and make the connection then. But that would require a
     *       flat surface.
     */
    std::pair<DirectionSample3f, Spectrum>
    sample_direction(const Interaction3f & /*it*/, const Point2f & /*sample*/,
                     Mask /*active*/) const override {
        return { dr::zeros<DirectionSample3f>(), dr::zeros<Spectrum>() };
    }

    Float pdf_direction(const Interaction3f & /*it*/,
                        const DirectionSample3f & /*ds*/,
                        Mask /*active*/) const override {
        return 0.f;
    }

    // std::pair<PositionSample3f, Float>
    // sample_position(Float /*time*/, const Point2f &/*sample*/,
    //                 Mask /*active*/) const override {
    //     // PositionSample3f ps = m_shape->sample_position(time, sample, active);
    //     // Float weight        = dr::select(ps.pdf > 0.f, dr::rcp(ps.pdf), 0.f);
    //     // return { ps, weight };
    //     return { 0, 0 };
    // }

    /**
     * This will always 'fail': since `si.wi` is given,
     * there's zero probability that it is the exact direction of emission.
     */
    std::pair<Wavelength, Spectrum>
    sample_wavelengths(const SurfaceInteraction3f &si, Float sample,
                       Mask active) const override {
        return m_radiance->sample_spectrum(
            si, math::sample_shifted<Wavelength>(sample), active);
    }

    Spectrum eval(const SurfaceInteraction3f & /*si*/,
                  Mask /*active*/) const override {
        return 0.f;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "Laser[" << std::endl
            << "  radiance = " << string::indent(m_radiance) << "," << std::endl;
        oss << std::endl << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    ref<Texture> m_radiance;
    field<Point3f> m_position;
    Float m_area = 0.f;
};

MI_IMPLEMENT_CLASS_VARIANT(Laser, Emitter)
MI_EXPORT_PLUGIN(Laser, "Laser emitter");

NAMESPACE_END(mitsuba)