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
        m_flags = EmitterFlags::DeltaPosition | EmitterFlags::DeltaDirection;
        dr::set_attr(this, "flags", m_flags);
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_object("radiance", m_radiance.get(), +ParamFlags::Differentiable);
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
        Float a = dr::dot(-m_direction, -m_direction);
        Float b = 2 * dr::dot(-m_direction, it.p - m_position);
        Float c = dr::dot(it.p - m_position, it.p - m_position) - m_radius * m_radius;
        Float d = b*b - 4*a*c;

        DirectionSample3f ds;
        ds.p      = m_position;
        ds.n      = -m_direction;
        ds.uv     = 0.f;
        ds.time   = it.time;
        ds.pdf    = 0.f;
        ds.delta  = true;
        ds.emitter = this;
        ds.d      = -m_direction;
        ds.dist   = dr::norm(ds.p - it.p);
        
        if(dr::all(d < 0))
        {
            return {ds, Spectrum(0)};
        }
            
        
        Float t1 = (-b + dr::sqrt(d)) / (2 * a);
        Float t2 = (-b - dr::sqrt(d)) / (2 * a);
        
        if(dr::all(0 <= t1 || 0 <= t2))
        {
            SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
            si.wavelengths = it.wavelengths;
            ds.pdf = 1.f;
            return {ds, depolarizer<Spectrum>(m_radiance->eval(si, active))};
        }
        
        return {ds, dr::zeros<Spectrum>() };
    }

    Float pdf_direction(const Interaction3f & it,
                        const DirectionSample3f & /*ds*/,
                        Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::EndpointEvaluate, active);
        Float a = dr::dot(-m_direction, -m_direction);
        Float b = 2 * dr::dot(-m_direction, it.p - m_position);
        Float c = dr::dot(it.p - m_position, it.p - m_position) - m_radius * m_radius;
        Float d = b*b - 4*a*c;
        
        if(dr::any(d < 0))
            return 0.f;
        
        Float t1 = (-b + dr::sqrt(d)) / (2 * a);
        Float t2 = (-b - dr::sqrt(d)) / (2 * a);
        
        if(dr::any(dr::all(0 <= t1 && t1 <= 1) || dr::all(0 <= t2 && t2 <= 1)))
            return 1.f;

        return 0.f;
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
    Point3f m_position;
    ScalarVector3f m_direction;
    ScalarFloat m_radius;
};

MI_IMPLEMENT_CLASS_VARIANT(Laser, Emitter)
MI_EXPORT_PLUGIN(Laser, "Laser emitter");

NAMESPACE_END(mitsuba)