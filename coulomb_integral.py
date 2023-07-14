#!/usr/bin/env python3
import numpy as np
from numpy.fft import fftn, ifftn
from vasp_constant import TPI, AUTOA, RYTOEV
from vaspwfc import vaspwfc
from paw import pawpotcar


class PAWCoulombIntegral(pawpotcar):
    '''
    Evaluate the PAW contribution to Coulomb Integral

                             ⌠⌠  drdr′  *       *
    K        = (n   |n   ) = ⎮⎮ ────── n   (r) n   (r′)         (97)
     nn′,mm′     nn′  mm′    ⌡⌡ |r-r′|  nn′     mm′

    Reference:
    - "The Projector Augmented-wave Method"
        https://arxiv.org/abs/0910.1921.pdf
    '''

    def __init__(self, potfile="POTCAR"):
        # Init pawpotcar class
        super().__init__(potfile=potfile)

        # set the compensation charge function
        self.set_comp_function()

    def set_comp_function(self):
        '''
        Set the compensation charge function gₗ(r)

        According to Kresse's paper

        gₗ(r) = a₁ jₗ(q₁ r) + a₂ jₗ(q₂ r)       (61)
        =================================
        where
            jₗ is `spherical_jn`
        *********************************
            gₗ(rcomp) = 0
        *********************************
            ⌠ʳᶜᵒᵐᵖ
            ⎮      dr gₗ(r) rˡ⁺² = 1            (62)
            ⌡₀
        *********************************
            d jₗ(qᵢ r)⎮
            ──────────⎮        = 0              (63)
               d r    ⎮r=rcomp
        =================================

        Reference:
        - "From ultrasoft pseudopotentials to the projector augmented-wave method"
            https://journals.aps.org/prb/abstract/10.1103/PhysRevB.59.1758
        '''

        from scipy.special import spherical_jn
        from scipy.integrate import quadrature
        from scipy.linalg import solve

        def find_q(L: int) -> [float, float]:
            """
            @in:
                - L: angular momentum number
            @out:
                - two roots of j_L(x) = 0
            """
            THRESHOLD = 1E-10

            nfound = 0
            ret = [0.0, 0.0]

            xinit = 1.0
            for nfound in range(2):
                # find the coarse interval of root
                x1 = xinit
                x2 = x1 + 1.0
                fx1 = spherical_jn(L, x1)
                fx2 = spherical_jn(L, x2)
                while fx1 * fx2 > 0:
                    x2 += 1.0
                    fx2 = spherical_jn(L, x2)

                # binary search
                x1  = x2 - 1.0; fx1 = spherical_jn(L, x1)
                while x2 - x1 > THRESHOLD:
                    mid = (x1 + x2) / 2
                    fmid = spherical_jn(L, mid)
                    if fx1 * fmid < 0:
                        x2 = mid
                        fx2 = fmid
                    else:
                        x1 = mid
                        fx1 = fmid

                ret[nfound] = x1
                xinit = x2
            return ret

        def find_alpha(R: float, q: [float, float], l: int) -> [float, float]:
            """
            Returns a₁ and a₂ that satisfie

            gₗ(r) = a₁ jₗ(q₁ r) + a₂ jₗ(q₂ r)       (61)
            """
            def glr_int(r: float, q: float, l: int) -> float:
                """

                @in:
                    r: compensation charge radius
                    q: q produced in `find_q`
                    l: angular momentum number
                @out:
                    ⌠ʳ
                    ⎮  dr jₗ(q * r) rˡ⁺²
                    ⌡₀
                """
                GAUSSIAN_QUADRATURE_ORDER = 32
                val, err = quadrature(
                        lambda x: spherical_jn(l, q * x) * x**(l+2),
                        0.0, r, maxiter=GAUSSIAN_QUADRATURE_ORDER)
                return val

            # Solve 2x2 linear system using gaussian elimination
            Amat = np.zeros((2, 2))
            Amat[0, 0] = q[0] * spherical_jn(l, q[0] * R, derivative=True)
            Amat[0, 1] = q[1] * spherical_jn(l, q[1] * R, derivative=True)
            Amat[1, 0] = glr_int(R, q[0], l)
            Amat[1, 1] = glr_int(R, q[1], l)

            Bmat = np.array([0.0, 1.0])
            return solve(Amat, Bmat)
        
        lmax = self.proj_l.max() * 2 + 1
        gl = []
        for l in range(lmax):
            q = np.array(find_q(l)) / self.rcomp
            a = find_alpha(self.rcomp, q, l)
            gl.append(lambda r: (
                a[0] * spherical_jn(l, q[0] * r) +
                a[1] * spherical_jn(l, q[1] * r)
                ))

        self.gl = gl

    @property
    def lmidx(self):
        '''
        Enumerating (l,m)
        '''
        if not hasattr(self, '_lmidx'):
            lmax = self.proj_l.max() * 2 + 1
            _lmidx = np.array([[l, m]
                               for l in range(lmax)
                               for m in range(-l, l+1)])
            self._lmidx = _lmidx

        return self._lmidx

    def get_Delta_Li1i2(self):
        """
                 ⌠
        Δ      = ⎮ dr rˡ Y (r) [ϕ  (r) ϕ  (r) - ϕ̃  (r) ϕ̃  (r) ]
         Li₁i₂   ⌡        L      i₁     i₂       i₁     i₂

                 ⌠
               = ⎮ dr rˡ⁺² ϕ  (r) ϕ  (r) ⋅G(l₁,l₂,l,m₁,m₂,m)
                 ⌡          i₁     i₂

        where L  = (l, m)
              i1 = (n1, l1, m1)
              i2 = (n1, l2, m2)

        and G(l1,l2,l,m1,m2,m) is the ``Gaunt Coefficient''

        Returns: Delta_Li1i2
        """
        from pysbt import GauntTable

        L    = self.lmidx
        i1i2 = np.array(self.ilm)
        lmax = self.proj_l.max() * 2 + 1
        npro = self.proj_l.size

        ## Raidal part \int dr r^(l+2) (phi_n1^ae(r) * phi_n2^ae(r) - phi_n1^ps(r) * phi_n2^ps(r))
        ## Note that pwav_ae = phi_n(r) / r as defined in POTCAR
        radial_integral = np.zeros((lmax, npro, npro))
        for l in range(lmax):
            rpower = self.rgrid ** l    ## r^l
            for n1 in range(npro):
                for n2 in range(npro):
                    radial_integral[l, n1, n2] = self.radial_simp_int(
                            rpower * (self.paw_ae_wfc[n1,:] * self.paw_ae_wfc[n2,:] -
                                      self.paw_ps_wfc[n1,:] * self.paw_ps_wfc[n2,:])
                            )

        Delta_Li1i2 = np.zeros((L.shape[0], i1i2.shape[0], i1i2.shape[0]),
                               dtype=float)

        ## Asselmble Delta_Li1i2
        for iL, [l, m] in enumerate(L):
            for i1, [n1, l1, m1] in enumerate(i1i2):
                for i2, [n2, l2, m2] in enumerate(i1i2):
                    Delta_Li1i2[iL, i1, i2] = radial_integral[l, n1, n2] * GauntTable(l1, l2, l, m1, m2, m)

        return Delta_Li1i2

    def get_integral_1234(self, use_ps_wav=False):
        """
                              ⌠         *      *       1
        (ϕ   ϕ   | ϕ   ϕ  ) = ⎮ dr dr′ ϕ  (r) ϕ  (r) ────── ϕ  (r′) ϕ  (r′)
          i₁  i₂    i₃  i₄    ⌡         i₁     i₂    |r-r′|  i₃      i₄

                                                                     rˡ
                              ⎲   4π  ⌠                *      *       <
                            = ⎳  ──── ⎮ dr dr′ r² r′² ϕ  (r) ϕ  (r) ──── ϕ  (r′) ϕ  (r′) ⋅G(l₁,l₂,l,m₁,m₂,m) ⋅G(l₃,l₄,l,m₃,m₄,m)
                              ˡᵐ 2l+1 ⌡                i₁     i₂    rˡ⁺¹  i₃      i₄
                                                                     >
        Returns an array that enumerates all (12|34), [i1,i2,i3,i4]
        """
        from pysbt import GauntTable

        L    = self.lmidx
        i1i2 = np.array(self.ilm)
        lmax = self.proj_l.max() * 2 + 1
        npro = self.proj_l.size

        if use_ps_wav:
            wav = self.paw_ae_wfc
        else:
            wav = self.paw_ps_wfc

        ## r_< and r_>
        r = self.rgrid
        r_l = np.minimum(r[:, None], r[None, :])
        r_g = np.maximum(r[:, None], r[None, :])

        ## radial part:
        ## \int drdr' r^2 r'^2 * phi_n1(r)* phi_n2(r)*   ( r_<^l/_>^(l+1) )   phi_n3(r') phi_n4(r')
        radial_integral = np.zeros((lmax, npro, npro, npro, npro), dtype=float)
        for l in range(lmax):
            for n1 in range(npro):
                for n2 in range(npro):
                    for n3 in range(npro):
                        for n4 in range(npro):
                            for irp, rprime in enumerate(r):
                                radial_integral[l, n1, n2, n3, n4] += (
                                        # integrate dr first
                                        self.radial_simp_int(wav[n1,:] * wav[n2,:]
                                                             * (r_l[irp] ** l / r_g[irp] ** (l+1))
                                                             )  # r_<^l / r_>^(l+1)
                                        * wav[n3,irp] * wav[n4,irp] # phi(r')
                                        * self.rad_simp_w[irp]  # simpson integrate rule
                                )

        ## Assemble 4 term integral with radial part and angular part
        integral_1234 = np.zeros((i1i2.shape[0], i1i2.shape[0], i1i2.shape[0], i1i2.shape[0]), dtype=float)
        for iL, [l, m] in enumerate(L):
            for i1, [n1, l1, m1] in enumerate(i1i2):
                for i2, [n2, l2, m2] in enumerate(i1i2):
                    G12 = GauntTable(l1, l2, l, m1, m2, m)
                    if np.abs(G12) < 1E-6:
                        continue

                    for i3, [n3, l3, m3] in enumerate(i1i2):
                        for i4, [n4, l4, m4] in enumerate(i1i2):
                            G34 = GauntTable(l3, l4, l, m3, m4, m)
                            integral_1234[i1, i2, i3, i4] += (
                                    4 * np.pi / (2 * l + 1)
                                    * radial_integral[l, n1, n2, n3, n4]
                                    * G12 * G34
                                    )

        return integral_1234

    def get_integral_phi12_gl(self):
        '''
                         ⌠                        1
        (ϕ̃   ϕ̃   | g ) = ⎮ dr dr′ ϕ̃  (r) ϕ̃  (r) ────── g (r′)
          i₁  i₂    L    ⌡         i₁     i₂    |r-r′|  L
                                       rˡ

                         ⎲   4π  ⌠                               <                                ┌─┐
                       = ⎳  ──── ⎮ dr dr′ r² r′² ϕ̃  (r) ϕ̃  (r) ──── g (r′) ⋅G(l₁,l₂,l,m₁,m₂,m) ⋅2╲│π  G(l₃,l,0,m₃,m,0)
                         ˡᵐ 2l+1 ⌡                i₁     i₂    rˡ⁺¹  L
                                                                >

        where L = (l3,m3)
              i1 = (n1, l1, m1)
              i2 = (n2, l2, m2)
              i3 = (n3, l3, m3)
              i4 = (n4, l4, m4)
        '''
        from pysbt import GauntTable

        L    = self.lmidx
        i1i2 = np.array(self.ilm)
        lmax = self.proj_l.max() * 2 + 1
        npro = self.proj_l.size

        ## r_< and r_>
        r = self.rgrid
        r_l = np.minimum(r[:, None], r[None, :])
        r_g = np.maximum(r[:, None], r[None, :])

        ## gl(r) and gl(r')
        glr = np.array([[self.gl[l](rp)
                         for rp in r]
                        for l in range(lmax)
                        ])

        ## radial part
        ## \int drdr' 4pi / (2l+1) r^2 r'^2 * ~phi_n1(r)* ~phi_n2(r)*  (r_<^l/r_>^(l+1))  g_l(r')
        radial_integral = np.zeros((npro, npro, lmax, lmax), dtype=float)   # [n1, n2, l3, l]
        for n1 in range(npro):
            for n2 in range(npro):
                for l3 in range(lmax):
                    for l in range(lmax):
                        for irp, rprime in enumerate(glr[l,0:self.rcomp_idx]):
                            radial_integral[n1, n2, l3, l] += 4 * np.pi / (2 * l + 1) * (
                                    self.radial_simp_int(self.paw_ps_wfc[n1,:] * self.paw_ps_wfc[n2,:]  # r^2 ~phi_n1(r) ~phi_n2(r)
                                                         * r_l[irp,:] ** l / r_g[irp,:] ** (l+1), # r_<^l / r_>^(l+1)
                                                         inside_rcomp=True)
                                    * rprime ** 2               # r'^2
                                    * glr[l3,irp]               # g_l(r')
                                    * self.rad_simp_w[irp])     # simpson rule

        ## assemble integral_phi12_gl
        integral_phi12_gl = np.zeros((i1i2.shape[0], i1i2.shape[0], L.shape[0]))
        for [l, m] in L:
            for i1, [n1, l1, m1] in enumerate(i1i2):
                for i2, [n2, l2, m2] in enumerate(i1i2):
                    G12 = GauntTable(l1,l2,l,m1,m2,m)
                    if np.abs(G12) < 1E-6:
                        continue
                    for iL, [l3, m3] in enumerate(L):
                        G3l = GauntTable(l3,l,0,m3,m,0)
                        integral_phi12_gl[i1,i2,iL] += radial_integral[n1,n2,l3,l] * 2 * np.sqrt(np.pi) * G12 * G3l

        return integral_phi12_gl

    def get_integral_gl(self):
        '''
                           ⌠                1
        ((gₗ)) = (gₗ|gₗ) = ⎮ dr dr′ gₗ(r) ────── gₗ(r′)
                           ⌡              |r-r′|

                                                   l′
                                                  r
                  ⎲     4π  ⌠                      <           ⌠              *        *          ⌠               *            *
               =  ⎳   ───── ⎮ drdr′ r² r′² gₗ(r) ───── gₗ(r′) ⋅⎮ dθd𝜑 sin(θ) Yₗₘ(θ,𝜑) Y    (θ,𝜑) ⋅⎮ dθ'd𝜑' sin(θ') Yₗₘ(θ',𝜑') Y    (θ',𝜑')
                 l′m′ 2l′+1 ⌡                     l′+1         ⌡                       l′m′       ⌡                            l′m′
                                                 r
                                                  >

                                                   l′
                                                  r
                  ⎲   (4π)² ⌠                      <
               =  ⎳   ───── ⎮ drdr′ r² r′² gₗ(r) ───── gₗ(r′) ⋅G(l,l′,0,m,m′,0)²
                 l′m′ 2l′+1 ⌡                     l′+1
                                                 r
                                                  >
        '''
        from pysbt import GauntTable

        L    = self.lmidx
        lmax = self.proj_l.max() * 2 + 1

        ## r_< and r_>
        r = self.rgrid
        r_l = np.minimum(r[:, None], r[None, :])
        r_g = np.maximum(r[:, None], r[None, :])

        ## gl(r) and gl(r')
        glr = np.array([[self.gl[l](rp)
                         for rp in r]
                        for l in range(lmax)
                        ])

        integral_gl = np.zeros(L.size, dtype=float)
        for iL1, (l1, m1) in enumerate(L):
            for iL2, (l2, m2) in enumerate(L):
                Gl1l2 = GauntTable(l1, l2, 0, m1, m2, 0)
                if np.abs(Gl1l2) < 1E-6:
                    continue
                for irp, rprime in enumerate(self.rgrid[0:self.rcomp_idx]):
                    # integrate dr first
                    integral_gl[l1] += 4 * np.pi / (2 * l2 + 1) * (
                            self.radial_simp_int(r ** 2 * rprime ** 2
                                                 * r_l[irp,:] ** (l2) / r_g[irp,:] ** (l2 + 1)
                                                 * glr[l1,:] * glr[l1,irp])
                            ) * Gl1l2 ** 2 * self.rad_simp_w[irp]   # simpson's rule

        return integral_gl

    def get_DeltaC_1234(self):
        '''
                     1                                             ⎲  1                         1
        ΔC         = ─ [(ϕ  ϕ   | ϕ  ϕ  ) - (ϕ̃   ϕ̃   | ϕ̃   ϕ̃  )] - ⎳ [─ Δ      (ϕ̃   ϕ̃   | g ) + ─ Δ      (ϕ̃   ϕ̃   | g ) + Δ      ((g )) Δ     ]
          i₁i₂i₃i₄   2    i₁ i₂    i₃ i₄      i₁  i₂    i₃  i₄     ᴸ  2  Li₁i₂   i₁  i₂    L    2  Li₃i₄   i₃  i₄    L     Li₁i₂    L    Li₃i₄
        '''
        L    = self.lmidx
        i1i2 = np.array(self.ilm)

        Delta_Li1i2       = self.get_Delta_Li1i2()
        integral_phi12_gl = self.get_integral_phi12_gl()
        integral_gl       = self.get_integral_gl()

        first_term  = 0.5 * (self.get_integral_1234(use_ps_wav=False) - self.get_integral_1234(use_ps_wav=True))
        second_term = np.zeros((i1i2.shape[0], i1i2.shape[0], i1i2.shape[0], i1i2.shape[0]), dtype=float)
        for i1, [n1, l1, m1] in enumerate(i1i2):
            for i2, [n2, l2, m2] in enumerate(i1i2):
                for i3, [n3, l3, m3] in enumerate(i1i2):
                    for i4, [n4, l4, m4] in enumerate(i1i2):
                        for iL, [l, m] in enumerate(L):
                            second_term[i1, i2, i3, i4] += 0.5 * (
                                Delta_Li1i2[iL, i1, i2] * integral_phi12_gl[i1, i2, iL] +
                                Delta_Li1i2[iL, i3, i4] + integral_phi12_gl[i3, i4, iL]
                            ) + Delta_Li1i2[iL, i1, i2] * integral_gl[iL] * Delta_Li1i2[iL, i3, i4]

        ## V = 1/(4pi*e0) * e^2/r
        ## 4pi*e0 = e^2/(a0*Eh) where a0 is the Bohr radius
        ## Thus 1/(4pi*e0) = a0*Eh/e^2
        ##      a0 / Angstrom = 0.529177...
        ##      Eh = 27.2114... eV
        ##      e^2 vanished with (rho_12|rho_34)

        return (first_term - second_term) * BOHR2ANGSTROM * EHARTREE


class PWCoulombIntegral(vaspwfc):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def density_matrix(self, m: int, n: int):
        '''
              ⌠     *           iGr
        uₘₙ = ⎮ dr ϕₘ(r) ϕₙ(r) e
              ⌡
        '''
        um = self.wfc_r(ispin=1, ikpt=1, iband=m, ngrid=self._ngrid, norm=False)
        un = self.wfc_r(ispin=1, ikpt=1, iband=n, ngrid=self._ngrid, norm=False)
        umun = um.conj() * un

        print("|umun| = {}".format(np.linalg.norm(umun)))
        print("fftn(umun) = {}".format(np.linalg.norm(fftn(umun, norm='ortho'))))

        return fftn(umun, norm='ortho')

    @property
    def gvectors_cart(self):
        if not hasattr(self, '_gvectors_cart'):
            fx, fy, fz = [np.arange(n, dtype=int) for n in self._ngrid]
            fx[self._ngrid[0] // 2 + 1:] -= self._ngrid[0]
            fy[self._ngrid[1] // 2 + 1:] -= self._ngrid[1]
            fz[self._ngrid[2] // 2 + 1:] -= self._ngrid[2]
            gz, gy, gx = np.array(
                    np.meshgrid(fz, fy, fx, indexing='ij')
                    ).reshape((3, -1))
            kgrid = np.array([gx, gy, gz], dtype=float).T
            self._gvectors_cart = kgrid @ self._Bcell * TPI
        return self._gvectors_cart

    def coulomb_integral(self, m: int, n: int, p: int, q: int):
        '''
                  ⌠                 *      1   *
        (mn|pq) = ⎮ dr₁ dr₂ ψₘ(r₁) ψₙ(r₁) ─── ψₚ(r₂) ψ (r₂)
                  ⌡                       r₁₂         q
        '''
        denmat_mn = self.density_matrix(m, n).flatten().conj()
        denmat_pq = self.density_matrix(p, q).flatten()
        gvec      = np.linalg.norm(self.gvectors_cart, axis=-1)

        integral = 0.0 + 0.0j
        for i in range(denmat_mn.size):
            if gvec[i] < 1E-6:
                continue
            integral += denmat_mn[i] * denmat_pq[i] / gvec[i] ** 2

        integral *= AUTOA * RYTOEV / np.pi**2
        return integral


if '__main__' == __name__:
    # pawci = PAWCoulombIntegral(potfile='examples/projectors/lreal_false/POTCAR')
    # pwci  = PWCoulombIntegral(fnm='examples/projectors/lreal_false/WAVECAR')
    pass
