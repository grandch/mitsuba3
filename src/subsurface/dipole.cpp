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


#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/subsurface.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum> class IsotropicDipoleQuery {
public:
    MI_IMPORT_TYPES(Point3f, IrradianceSample3f)
    inline IsotropicDipoleQuery(const Spectrum &zr, const Spectrum &zv,
                                const Spectrum &sigmaTr, const Point3f &p)
        : zr(zr), zv(zv), sigmaTr(sigmaTr), result(0.0f), p(p) {}

    inline void operator()(const IrradianceSample3f &sample) {
        Spectrum rSqr = Spectrum((p - sample.p).lengthSquared());

        /* Distance to the real source */
        Spectrum dr = (rSqr + zr * zr).sqrt();

        /* Distance to the image point source */
        Spectrum dv = (rSqr + zv * zv).sqrt();

        Spectrum C1 = zr * (sigmaTr + Spectrum(1.0f) / dr);
        Spectrum C2 = zv * (sigmaTr + Spectrum(1.0f) / dv);

        /* Do not include the reduced albedo - will be canceled out later */
        Spectrum dMo = Spectrum(dr::InvFourPi<Float>) *
                       (C1 * ((-sigmaTr * dr).exp()) / (dr * dr) +
                        C2 * ((-sigmaTr * dv).exp()) / (dv * dv));

        result += dMo * sample.E * sample.area;
    }

    inline const Spectrum &getResult() const { return result; }

    const Spectrum &zr, &zv, &sigmaTr;
    Spectrum result;
    Point3f p;
};

template <typename Float, typename Spectrum>
class IsotropicDipole : public Subsurface<Float, Spectrum> {

public:
    MI_IMPORT_BASE(Subsurface, m_shapes)
    MI_IMPORT_TYPES(Texture, Scene, Sensor, Sampler, IrradianceOctree)

    IsotropicDipole(const Properties &props) : Base(props) {
        /* How many samples should be taken when estimating
           the irradiance at a given point in the scene? */
        m_irr_samples = props.get<int>("irr_samples", 16);

        /* When estimating the irradiance at a given point,
           should indirect illumination be included in the final estimate? */
        m_irr_indirect = props.get<bool>("irr_indirect", true);

        /* Multiplicative factor, which can be used to adjust the number of
           irradiance samples */
        m_sample_multiplier = props.get<Float>("sample_multiplier", 1.0f);

        /* Error threshold - lower means better quality */
        m_quality = props.get<Float>("quality", 0.2f);

        // Specifies the internal index of refraction at the interface
        ScalarFloat int_ior = lookup_ior(props, "int_ior", "bk7");

        // Specifies the external index of refraction at the interface
        ScalarFloat ext_ior = lookup_ior(props, "ext_ior", "air");

        if (int_ior < 0 || ext_ior < 0)
            Throw("The interior and exterior indices of refraction must"
                  " be positive!");

        m_eta = int_ior / ext_ior;

        m_sigma_t = props.texture<Texture>("sigma_t", 1.0);
        m_albedo  = props.texture<Texture>("albedo", 1.0);
        m_g       = props.texture<Texture>("g", 1.0);

        // m_sigma_s = m_albedo * m_sigma_t;
        // m_sigma_a = m_sigma_t - m_sigma_s;

        // m_sigma_s_prime = m_sigma_s * (new <Texture>(1.0f) - m_g);
        // m_sigma_t_prime = m_sigma_s_prime + m_sigma_a;

        // /* Find the smallest mean-free path over all wavelengths */
        // Spectrum mfp = Spectrum(1.0f) / m_sigma_t_prime;
        // m_radius     = std::numeric_limits<Float>::max();
        // for (auto lambda : mfp)
        //     m_radius = std::min(m_radius, mfp[lambda]);

        // /* Average diffuse reflectance due to mismatched indices of refraction
        //  */
        // m_Fdr = fresnel_diffuse_reflectance(1 / m_eta);

        // /* Dipole boundary condition distance term */
        // Float A = (1 + m_Fdr) / (1 - m_Fdr);

        // /* Effective transport extinction coefficient */
        // m_sigma_tr = (m_sigma_a * m_sigma_t_prime * 3.0f).sqrt();

        // /* Distance of the two dipole point sources to the surface */
        // m_zr = mfp;
        // m_zv = mfp * (1.0f + 4.0f / 3.0f * A);
    }

    Spectrum Lo(const Scene *scene, Sampler *sampler,
                const SurfaceInteraction3f &si, const Vector3f &d,
                int depth = 0) {
        if (dr::dot(si.sh_frame.n, d) < 0)
            return 0.f;

        // IsotropicDipoleQuery<Float, Spectrum> query(m_zr, m_zv, m_sigma_tr, si.p);
        // m_octree->perform_query(query);
        // Spectrum result(query.getResult() * dr::InvPi<Float>);

        // if (m_eta != 1.f)
        //     result *= 1.f - fresnel_conductor(dr::dot(si.sh_frame.n, d), m_eta);

        // return result;
    };

    bool preprocess(const Scene *scene, int sceneResID, int cameraResID,
                    int _samplerResID) {
        // if (m_octree)
        //     return true;

        // ref<Timer> timer = new Timer();

        // BoundingBox3f aabb;
        // Float sa;

        // std::vector<PositionSample3f> points;
        // /* It is necessary to increase the sampling resolution to
        //    prevent low-frequency noise in the output */
        // Float actualRadius = m_radius / std::sqrt(m_sample_multiplier * 20);

        // BlueNoiseSampler<Float, Spectrum> bns{};

        // bns.blueNoisePointSet(scene, m_shapes, 0, actualRadius, &points, sa, aabb);

        /* 2. Gather irradiance in parallel */
        // const ref<Sensor> sensor                = scene->sensors()[0];
        // ref<IrradianceSamplingProcess> proc = new IrradianceSamplingProcess(
        //     points, 1024, m_irrSamples, m_irrIndirect,
        //     sensor->getShutterOpen() + 0.5f * sensor->getShutterOpenTime(),
        //     job);

        // /* Create a sampler instance for every core */
        // ref<Sampler> sampler =
        //     static_cast<Sampler *>(PluginManager::getInstance()->createObject(
        //         MTS_CLASS(Sampler), Properties("independent")));
        // std::vector<SerializableObject *> samplers(sched->getCoreCount());
        // for (size_t i = 0; i < sched->getCoreCount(); ++i) {
        //     ref<Sampler> clonedSampler = sampler->clone();
        //     clonedSampler->incRef();
        //     samplers[i] = clonedSampler.get();
        // }

        // int samplerResID    = sched->registerMultiResource(samplers);
        // int integratorResID = sched->registerResource(
        //     const_cast<Integrator *>(scene->getIntegrator()));

        // proc->bindResource("scene", sceneResID);
        // proc->bindResource("integrator", integratorResID);
        // proc->bindResource("sampler", samplerResID);
        // scene->bindUsedResources(proc);
        // m_proc = proc;
        // sched->schedule(proc);
        // sched->wait(proc);
        // m_proc = NULL;
        // for (size_t i = 0; i < samplers.size(); ++i)
        //     samplers[i]->decRef();

        // sched->unregisterResource(samplerResID);
        // sched->unregisterResource(integratorResID);
        // if (proc->getReturnStatus() != ParallelProcess::ESuccess)
        //     return false;

        // Log(EDebug, "Done gathering (took %i ms), clustering ..",
        //     timer->getMilliseconds());
        // timer->reset();

        // std::vector<IrradianceSample> &samples =
        //     proc->getIrradianceSampleVector()->get();
        // sa /= samples.size();

        // for (size_t i = 0; i < samples.size(); ++i)
        //     samples[i].area = sa;

        // m_octree = new IrradianceOctree(aabb, m_quality, samples);

        // Log(EDebug, "Done clustering (took %i ms).", timer->getMilliseconds());
        // m_octreeResID = Scheduler::getInstance()->registerResource(m_octree);

        // return true;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "IsotropicDipole[" << std::endl;
        oss << "  albedo = " << string::indent(m_albedo) << "," << std::endl;
        oss << "  sigma_t = " << m_sigma_t << "," << std::endl;
        oss << "  eta = " << m_eta << "," << std::endl;
        oss << "  irr_samples = " << m_irr_samples << "," << std::endl;
        oss << "  irr_indirect = " << m_irr_indirect << "," << std::endl;
        oss << "  sample_multiplier = " << m_sample_multiplier << ","
            << std::endl;
        oss << "  quality = " << m_quality << "]" << std::endl;
        return oss.str();
    }

    MI_DECLARE_CLASS()

private:
    ref<Texture> m_sigma_t, m_albedo;
    Float m_eta;

    ref<Texture> m_sigma_a, m_sigma_s, m_g;
    ref<Texture> m_sigma_tr, m_zr, m_zv;
    ref<Texture> m_sigma_s_prime, m_sigma_t_prime;
    Float m_Fdr;

    // ref<IrradianceOctree<Float, Spectrum>> m_octree;
    int m_octreeResID, m_octreeIndex;
    int m_irr_samples;
    bool m_irr_indirect;
    Float m_sample_multiplier, m_quality;
    Float m_radius;
};

MI_IMPLEMENT_CLASS_VARIANT(IsotropicDipole, Subsurface)
MI_EXPORT_PLUGIN(IsotropicDipole, "Isotropic dipole")
NAMESPACE_END(mitsuba)
