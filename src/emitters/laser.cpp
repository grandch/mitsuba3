#include <mitsuba/core/properties.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/texture.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class Laser final : public Emitter<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Emitter, m_flags)
    MI_IMPORT_TYPES(Scene, Texture)

    Laser(const Properties &props) : Base(props) {
        m_position = props.get<ScalarPoint3f>("position");
        m_radiance = props.texture_d65<Texture>("radiance", 1.f);
        m_radius = props.get<float>("radius");
        m_direction = dr::normalize(props.get<ScalarVector3f>("direction"));
        m_flags = EmitterFlags::DeltaPosition | EmitterFlags::DeltaDirection | EmitterFlags::Infinite;
        dr::set_attr(this, "flags", m_flags);
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_object("radiance", m_radiance.get(), +ParamFlags::NonDifferentiable);
        callback->put_parameter("radius", m_radius, +ParamFlags::NonDifferentiable);
        callback->put_parameter("position", (Point3f &) m_position.value(), +ParamFlags::NonDifferentiable);
        callback->put_parameter("direction", (ScalarVector3f &) m_direction.value(), +ParamFlags::NonDifferentiable);
    }

    /*
    * Don't use this method
    * The laser emitter is designed to be used with a VolPath Integrator and it doesn't make sense to implement the sample_ray method.
    */
    std::pair<Ray3f, Spectrum> sample_ray(Float /*time*/, Float /*wavelength_sample*/,
                                          const Point2f &/*sample2*/,
                                          const Point2f & /*sample3*/,
                                          Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleRay, active);                                
        return { Ray3f(0,0), Spectrum(0) };
    }

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
    sample_direction(const Interaction3f & it, const Point2f & /*sample*/,
                     Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointSampleDirection, active);
        Float a = dr::dot(-m_direction.value(), -m_direction.value());
        Float b = 2 * dr::dot(-m_direction.value(), it.p - m_position.value());
        Float c = dr::dot(it.p - m_position.value(), it.p - m_position.value()) - m_radius * m_radius;
        Float d = b*b - 4*a*c;

        DirectionSample3f ds;
        ds.p      = m_position.value();
        ds.n      = -m_direction.value();
        ds.uv     = 0.f;
        ds.time   = it.time;
        ds.delta  = true;
        ds.emitter = this;
        ds.d      = m_direction.value();
        ds.dist   = dr::norm(ds.p - it.p);
        
        active &= d >= 0;
        
        Float t1 = (-b + dr::sqrt(d)) / (2 * a);
        Float t2 = (-b - dr::sqrt(d)) / (2 * a);
        
        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.wavelengths = it.wavelengths;
        
        active &=0 <= t1 || 0 <= t2;
        ds.pdf = dr::select(active, 1.f, 0.f);
        
        return {ds, depolarizer<Spectrum>(m_radiance->eval(si, active))};
    }

    Float pdf_direction(const Interaction3f & it,
                        const DirectionSample3f & /*ds*/,
                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        Float a = dr::dot(-m_direction.value(), -m_direction.value());
        Float b = 2 * dr::dot(-m_direction.value(), it.p - m_position.value());
        Float c = dr::dot(it.p - m_position.value(), it.p - m_position.value()) - m_radius * m_radius;
        Float d = b*b - 4*a*c;
        
        active &= d >= 0;
        
        Float t1 = (-b + dr::sqrt(d)) / (2 * a);
        Float t2 = (-b - dr::sqrt(d)) / (2 * a);
        
        active &= (0 <= t1 && t1 <= 1) || (0 <= t2 && t2 <= 1);

        return dr::select(active, 1.f, 0.f);
    }

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

    Spectrum eval(const SurfaceInteraction3f & si,
                  Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        return depolarizer<Spectrum>(m_radiance->eval(si, active));
    }

    /// This emitter does not occupy any particular region of space, return an invalid bounding box
    ScalarBoundingBox3f bbox() const override { return ScalarBoundingBox3f(); }

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
    field<ScalarVector3f> m_direction;
    ScalarFloat m_radius;
};

MI_IMPLEMENT_CLASS_VARIANT(Laser, Emitter)
MI_EXPORT_PLUGIN(Laser, "Laser emitter");

NAMESPACE_END(mitsuba)