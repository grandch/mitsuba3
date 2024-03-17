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

#include "mitsuba/render/integrator.h"
#include <mitsuba/render/bluenoise.h>
#include <mitsuba/render/fresnel.h>
#include <mitsuba/render/ior.h>
#include <mitsuba/render/irrtree.h>
#include <mitsuba/render/subsurface.h>
#include <mitsuba/render/texture.h>
#include <vector>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum> class IsotropicDipoleQuery {
public:
    MI_IMPORT_TYPES(Texture)
    inline IsotropicDipoleQuery(const Spectrum &zr, const Spectrum &zv,
                                const Spectrum &sigmaTr, const Point3f &p)
        : zr(zr), zv(zv), sigmaTr(sigmaTr), result(0.0f), p(p) {}

    inline void operator()(const IrradianceSample3f &sample) {
        Spectrum rSqr = Spectrum(dr::squared_norm(p - sample.p));

        /* Distance to the real source */
        Spectrum dr = dr::sqrt(rSqr + zr * zr);

        /* Distance to the image point source */
        Spectrum dv = dr::sqrt(rSqr + zv * zv);

        Spectrum C1 = zr * (sigmaTr + Spectrum(1.0f) / dr);
        Spectrum C2 = zv * (sigmaTr + Spectrum(1.0f) / dv);

        /* Do not include the reduced albedo - will be canceled out later */
        Spectrum dMo = Spectrum(dr::InvFourPi<Float>) *
                       (C1 * (dr::exp(-sigmaTr * dr)) / (dr * dr) +
                        C2 * (dr::exp(-sigmaTr * dv)) / (dv * dv));

        result += dMo * sample.E * sample.area;
    }

    inline const Spectrum &getResult() const { return result; }

    const Spectrum &zr, &zv, &sigmaTr;
    Spectrum result;
    Point3f p;
};

template <typename Float, typename Spectrum>
class IsotropicDipole final : public Subsurface<Float, Spectrum> {

public:
    MI_IMPORT_BASE(Subsurface, m_shapes)
    MI_IMPORT_TYPES(Texture, Scene, Sensor, Sampler, Integrator,
                    IrradianceOctree)

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
    }

    bool preprocess(const Scene *scene, Sampler *sampler) override {
        if (m_octree)
            return true;

        BoundingBox3f aabb;
        Float sa;

        std::vector<MiniPositionSample3f> points;
        /* It is necessary to increase the sampling resolution to
           prevent low-frequency noise in the output */
        Float actualRadius = m_radius / dr::sqrt(m_sample_multiplier * 20);

        BlueNoiseSampler<Float, Spectrum> bns{};

        bns.blueNoisePointSet(scene, m_shapes, 0, sampler, actualRadius,
                              &points, sa, aabb);

        /* 2. Gather irradiance in parallel */
        ref<Sensor> sensor = scene->sensors()[0];

        Float time =
            sensor->shutter_open() + 0.5f * sensor->shutter_open_time();

        std::vector<IrradianceSample3f> samples;
        samples.reserve(points.size());

        const SamplingIntegrator<Float, Spectrum> *integrator =
            dynamic_cast<const SamplingIntegrator<Float, Spectrum> *>(
                scene->integrator());

        ref<Sampler> samplr = sampler->clone();

        for (size_t i = 0; i < points.size(); ++i) {
            /* Create a fake intersection record */
            const MiniPositionSample mps = points[i];

            PositionSample3f ps{ mps.p, mps.n, {}, time, 0.0, false };

            SurfaceInteraction3f si(ps, Wavelength());
            si.p        = mps.p;
            si.sh_frame = Frame3f(mps.n);
            // si.shape = scene->shapes()[sample.shapeIndex].get();
            si.time   = time;
            si.duv_dx = 0;
            si.duv_dy = 0;

            Spectrum E = 0.0;
            for (int i = 0; i < m_irr_samples; ++i) {
                auto [direct, weight, emitter] = scene->sample_emitter_ray(
                    time, samplr->next_1d(true), samplr->next_2d(true),
                    samplr->next_2d(true));

                E += integrator->sample(scene, samplr, direct).first;
                if (m_irr_indirect) {
                    // unimplemented for now
                    //     RayDifferential3f indirect{ si.p, d, si.time };
                    //     integrator->sample(scene, sampler, indirect);
                }
            }
            E /= m_irr_samples;
            samples.push_back(IrradianceSample3f(si.p, E, {}));
        }

        sa /= samples.size();

        for (size_t i = 0; i < samples.size(); ++i)
            samples[i].area = sa;

        m_octree = new IrradianceOctree(aabb, m_quality, samples);

        return true;
    }

    Spectrum Lo(const Scene * /*scene*/, Sampler * /*sampler*/,
                const SurfaceInteraction3f &si, const Vector3f &d,
                UInt32 /*depth*/) const override {

        if (dr::any(dr::dot(si.sh_frame.n, d) < 0))
            return 0.f;

        Spectrum sigma_s       = m_albedo->eval(si) * m_sigma_t->eval(si);
        Spectrum sigma_a       = m_sigma_t->eval(si) - sigma_s;
        Spectrum sigma_s_prime = sigma_s * (1.0f - m_albedo->eval(si));
        Spectrum sigma_t_prime = sigma_s_prime + sigma_a;

        /* Find the smallest mean-free path over all wavelengths */
        Spectrum mfp = Spectrum(1.0f) / sigma_t_prime;
        Float radius = std::numeric_limits<Float>::max();
        for (Float lambda : mfp)
            radius = dr::minimum(radius, lambda);

        /* Average diffuse reflectance due to mismatched indices of refraction
         */
        Float fdr = fresnel_diffuse_reflectance(1 / m_eta);

        /* Dipole boundary condition distance term */
        Float A = (1 + fdr) / (1 - fdr);

        /* Effective transport extinction coefficient */
        Spectrum sigma_tr = dr::sqrt(sigma_a * sigma_t_prime * 3.0f);

        /* Distance of the two dipole point sources to the surface */
        Spectrum zr = mfp;
        Spectrum zv = mfp * (1.0f + 4.0f / 3.0f * A);

        IsotropicDipoleQuery<Float, Spectrum> query(zr, zv, sigma_tr, si.p);
        m_octree->perform_query(query);
        Spectrum result(query.getResult() * dr::InvPi<Float>);

        if (m_eta != 1.f) {
            dr::Complex<Float> eta_c(m_eta, 0);
            result *=
                (1.f - fresnel_conductor(dr::dot(si.sh_frame.n, d), eta_c));
        }

        return result;
    };

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
    ref<Texture> m_sigma_t, m_albedo, m_g;
    Float m_eta;

    ref<IrradianceOctree> m_octree;
    int m_octreeResID, m_octreeIndex;
    int m_irr_samples;
    bool m_irr_indirect;
    Float m_sample_multiplier, m_quality;
    Float m_radius;
};

MI_IMPLEMENT_CLASS_VARIANT(IsotropicDipole, Subsurface)
MI_EXPORT_PLUGIN(IsotropicDipole, "Isotropic dipole")
NAMESPACE_END(mitsuba)
