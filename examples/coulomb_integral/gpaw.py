#!/usr/bin/env python3

import gzip
from xml.etree import ElementTree as ET
from glob import glob

import numpy as np
import numpy.typing as npt


"""
All the functions ended with
    - RETURN -> finished function
    - PASS   -> work in progress function

All the referred equations are in the paper
    "The Projector Augmented-wave Method"
    https://arxiv.org/pdf/0910.1921.pdf
"""


HARTREE = 27.211386024367243


class Gaunt:
    def __init__(self):
        self.set_g()
        self.set_YL()
        return


    def __call__(self, lmax: int = 2) -> npt.NDArray:
        r'''Gaunt coefficients

        :::

             ^      ^     -- L      ^
          Y (r)  Y (r) =  > G    Y (r)
           L      L       -- L L  L
            1      2      L   1 2

        Copied from gpaw/gaunt.py L13-45
        '''
        if not hasattr(self, 'gaunt_dict'):
            self.gaunt_dict = {}
        
        if lmax in self.gaunt_dict:
            return self.gaunt_dict[lmax]

        Lmax  = (  lmax + 1) ** 2
        L2max = (2*lmax + 1) ** 2
        G_LLL = np.zeros((Lmax, L2max, L2max))
        for L1 in range(Lmax):
            for L2 in range(L2max):
                for L in range(L2max):
                    r = 0.0
                    for c1, n1 in self.YL[L1]:
                        for c2, n2 in self.YL[L2]:
                            for c, n in self.YL[L]:
                                nx = n1[0] + n2[0] + n[0]
                                ny = n1[1] + n2[1] + n[1]
                                nz = n1[2] + n2[2] + n[2]
                                r += c * c1 * c2 * self.gam(nx, ny, nz)
                    G_LLL[L1, L2, L] = r

        self.gaunt_dict[lmax] = G_LLL
        return G_LLL


    def set_g(self) -> None:
        '''
        Copied from gpaw/spherical_harmonics.py L69-L71
        '''
        LMAX = 10
        g = [1.0] * LMAX
        for l in range(1, LMAX):
            g[l] = g[l-1] * (l - 0.5)
        self.g = g
        return


    def gam(self, n0: int, n1: int, n2: int) -> float:
        if (n0 % 2 != 0
            or n1 % 2 != 0
            or n2 % 2 != 0):
            return 0.0

        h0 = n0 // 2
        h1 = n1 // 2
        h2 = n2 // 2
        return (2.0 * np.pi
                * self.g[h0] * self.g[h1] * self.g[h2]
                / self.g[1 + h0 + h1 + h2])


    def set_YL(self) -> None:
        '''
        # Computer generated table - do not touch!
        # The numbers match those in c/bmgs/spherical_harmonics.c and were
        # originally generated with c/bmgs/sharmonic.py (old Python 2 code).

        Copied from gpaw/spherical_harmonics.py
        '''
        YL = [
            # s, l=0:
            [(0.28209479177387814, (0, 0, 0))],
            # p, l=1:
            [(0.4886025119029199, (0, 1, 0))],
            [(0.4886025119029199, (0, 0, 1))],
            [(0.4886025119029199, (1, 0, 0))],
            # d, l=2:
            [(1.0925484305920792, (1, 1, 0))],
            [(1.0925484305920792, (0, 1, 1))],
            [(0.6307831305050401, (0, 0, 2)),
             (-0.31539156525252005, (0, 2, 0)),
             (-0.31539156525252005, (2, 0, 0))],
            [(1.0925484305920792, (1, 0, 1))],
            [(0.5462742152960396, (2, 0, 0)),
             (-0.5462742152960396, (0, 2, 0))],
            # f, l=3:
            [(-0.5900435899266435, (0, 3, 0)),
             (1.7701307697799304, (2, 1, 0))],
            [(2.890611442640554, (1, 1, 1))],
            [(-0.4570457994644658, (0, 3, 0)),
             (1.828183197857863, (0, 1, 2)),
             (-0.4570457994644658, (2, 1, 0))],
            [(0.7463526651802308, (0, 0, 3)),
             (-1.1195289977703462, (2, 0, 1)),
             (-1.1195289977703462, (0, 2, 1))],
            [(1.828183197857863, (1, 0, 2)),
             (-0.4570457994644658, (3, 0, 0)),
             (-0.4570457994644658, (1, 2, 0))],
            [(1.445305721320277, (2, 0, 1)),
             (-1.445305721320277, (0, 2, 1))],
            [(0.5900435899266435, (3, 0, 0)),
             (-1.7701307697799304, (1, 2, 0))],
            # g, l=4:
            [(2.5033429417967046, (3, 1, 0)),
             (-2.5033429417967046, (1, 3, 0))],
            [(-1.7701307697799307, (0, 3, 1)),
             (5.310392309339792, (2, 1, 1))],
            [(-0.9461746957575601, (3, 1, 0)),
             (-0.9461746957575601, (1, 3, 0)),
             (5.6770481745453605, (1, 1, 2))],
            [(-2.0071396306718676, (2, 1, 1)),
             (2.676186174229157, (0, 1, 3)),
             (-2.0071396306718676, (0, 3, 1))],
            [(0.6347132814912259, (2, 2, 0)),
             (-2.5388531259649034, (2, 0, 2)),
             (0.31735664074561293, (0, 4, 0)),
             (-2.5388531259649034, (0, 2, 2)),
             (0.31735664074561293, (4, 0, 0)),
             (0.8462843753216345, (0, 0, 4))],
            [(2.676186174229157, (1, 0, 3)),
             (-2.0071396306718676, (3, 0, 1)),
             (-2.0071396306718676, (1, 2, 1))],
            [(2.8385240872726802, (2, 0, 2)),
             (0.47308734787878004, (0, 4, 0)),
             (-0.47308734787878004, (4, 0, 0)),
             (-2.8385240872726802, (0, 2, 2))],
            [(1.7701307697799307, (3, 0, 1)),
             (-5.310392309339792, (1, 2, 1))],
            [(-3.755014412695057, (2, 2, 0)),
             (0.6258357354491761, (0, 4, 0)),
             (0.6258357354491761, (4, 0, 0))],
            # h, l=5:
            [(-6.5638205684017015, (2, 3, 0)),
             (3.2819102842008507, (4, 1, 0)),
             (0.6563820568401701, (0, 5, 0))],
            [(8.302649259524165, (3, 1, 1)),
             (-8.302649259524165, (1, 3, 1))],
            [(-3.913906395482003, (0, 3, 2)),
             (0.4892382994352504, (0, 5, 0)),
             (-1.467714898305751, (4, 1, 0)),
             (-0.9784765988705008, (2, 3, 0)),
             (11.741719186446009, (2, 1, 2))],
            [(-4.793536784973324, (3, 1, 1)),
             (-4.793536784973324, (1, 3, 1)),
             (9.587073569946648, (1, 1, 3))],
            [(-5.435359814348363, (0, 3, 2)),
             (0.9058933023913939, (2, 3, 0)),
             (-5.435359814348363, (2, 1, 2)),
             (3.6235732095655755, (0, 1, 4)),
             (0.45294665119569694, (4, 1, 0)),
             (0.45294665119569694, (0, 5, 0))],
            [(3.508509673602708, (2, 2, 1)),
             (-4.678012898136944, (0, 2, 3)),
             (1.754254836801354, (0, 4, 1)),
             (-4.678012898136944, (2, 0, 3)),
             (1.754254836801354, (4, 0, 1)),
             (0.9356025796273888, (0, 0, 5))],
            [(-5.435359814348363, (3, 0, 2)),
             (3.6235732095655755, (1, 0, 4)),
             (0.45294665119569694, (5, 0, 0)),
             (0.9058933023913939, (3, 2, 0)),
             (-5.435359814348363, (1, 2, 2)),
             (0.45294665119569694, (1, 4, 0))],
            [(-2.396768392486662, (4, 0, 1)),
             (2.396768392486662, (0, 4, 1)),
             (4.793536784973324, (2, 0, 3)),
             (-4.793536784973324, (0, 2, 3))],
            [(3.913906395482003, (3, 0, 2)),
             (-0.4892382994352504, (5, 0, 0)),
             (0.9784765988705008, (3, 2, 0)),
             (-11.741719186446009, (1, 2, 2)),
             (1.467714898305751, (1, 4, 0))],
            [(2.075662314881041, (4, 0, 1)),
             (-12.453973889286246, (2, 2, 1)),
             (2.075662314881041, (0, 4, 1))],
            [(-6.5638205684017015, (3, 2, 0)),
             (0.6563820568401701, (5, 0, 0)),
             (3.2819102842008507, (1, 4, 0))],
            # i, l=6:
            [(4.099104631151485, (5, 1, 0)),
             (-13.663682103838287, (3, 3, 0)),
             (4.099104631151485, (1, 5, 0))],
            [(11.83309581115876, (4, 1, 1)),
             (-23.66619162231752, (2, 3, 1)),
             (2.366619162231752, (0, 5, 1))],
            [(20.182596029148968, (3, 1, 2)),
             (-2.0182596029148967, (5, 1, 0)),
             (2.0182596029148967, (1, 5, 0)),
             (-20.182596029148968, (1, 3, 2))],
            [(-7.369642076119388, (0, 3, 3)),
             (-5.527231557089541, (2, 3, 1)),
             (2.7636157785447706, (0, 5, 1)),
             (22.108926228358165, (2, 1, 3)),
             (-8.29084733563431, (4, 1, 1))],
            [(-14.739284152238776, (3, 1, 2)),
             (14.739284152238776, (1, 1, 4)),
             (1.842410519029847, (3, 3, 0)),
             (0.9212052595149235, (5, 1, 0)),
             (-14.739284152238776, (1, 3, 2)),
             (0.9212052595149235, (1, 5, 0))],
            [(2.9131068125936572, (0, 5, 1)),
             (-11.652427250374629, (0, 3, 3)),
             (5.8262136251873144, (2, 3, 1)),
             (-11.652427250374629, (2, 1, 3)),
             (2.9131068125936572, (4, 1, 1)),
             (4.660970900149851, (0, 1, 5))],
            [(5.721228204086558, (4, 0, 2)),
             (-7.628304272115411, (0, 2, 4)),
             (-0.9535380340144264, (2, 4, 0)),
             (1.0171072362820548, (0, 0, 6)),
             (-0.9535380340144264, (4, 2, 0)),
             (5.721228204086558, (0, 4, 2)),
             (-0.3178460113381421, (0, 6, 0)),
             (-7.628304272115411, (2, 0, 4)),
             (-0.3178460113381421, (6, 0, 0)),
             (11.442456408173117, (2, 2, 2))],
            [(-11.652427250374629, (3, 0, 3)),
             (4.660970900149851, (1, 0, 5)),
             (2.9131068125936572, (5, 0, 1)),
             (5.8262136251873144, (3, 2, 1)),
             (-11.652427250374629, (1, 2, 3)),
             (2.9131068125936572, (1, 4, 1))],
            [(7.369642076119388, (2, 0, 4)),
             (-7.369642076119388, (0, 2, 4)),
             (-0.46060262975746175, (2, 4, 0)),
             (-7.369642076119388, (4, 0, 2)),
             (0.46060262975746175, (4, 2, 0)),
             (-0.46060262975746175, (0, 6, 0)),
             (0.46060262975746175, (6, 0, 0)),
             (7.369642076119388, (0, 4, 2))],
            [(7.369642076119388, (3, 0, 3)),
             (-2.7636157785447706, (5, 0, 1)),
             (5.527231557089541, (3, 2, 1)),
             (-22.108926228358165, (1, 2, 3)),
             (8.29084733563431, (1, 4, 1))],
            [(2.522824503643621, (4, 2, 0)),
             (5.045649007287242, (0, 4, 2)),
             (-30.273894043723452, (2, 2, 2)),
             (-0.5045649007287242, (0, 6, 0)),
             (-0.5045649007287242, (6, 0, 0)),
             (5.045649007287242, (4, 0, 2)),
             (2.522824503643621, (2, 4, 0))],
            [(2.366619162231752, (5, 0, 1)),
             (-23.66619162231752, (3, 2, 1)),
             (11.83309581115876, (1, 4, 1))],
            [(-10.247761577878714, (4, 2, 0)),
             (-0.6831841051919143, (0, 6, 0)),
             (0.6831841051919143, (6, 0, 0)),
             (10.247761577878714, (2, 4, 0))],
            # j, l=7:
            [(14.850417383016522, (2, 5, 0)),
             (4.950139127672174, (6, 1, 0)),
             (-24.75069563836087, (4, 3, 0)),
             (-0.7071627325245963, (0, 7, 0))],
            [(-52.91921323603801, (3, 3, 1)),
             (15.875763970811402, (1, 5, 1)),
             (15.875763970811402, (5, 1, 1))],
            [(-2.5945778936013015, (6, 1, 0)),
             (2.5945778936013015, (4, 3, 0)),
             (-62.26986944643124, (2, 3, 2)),
             (4.670240208482342, (2, 5, 0)),
             (6.226986944643123, (0, 5, 2)),
             (31.13493472321562, (4, 1, 2)),
             (-0.5189155787202603, (0, 7, 0))],
            [(41.513246297620825, (3, 1, 3)),
             (12.453973889286246, (1, 5, 1)),
             (-41.513246297620825, (1, 3, 3)),
             (-12.453973889286246, (5, 1, 1))],
            [(-18.775072063475285, (2, 3, 2)),
             (-0.4693768015868821, (0, 7, 0)),
             (0.4693768015868821, (2, 5, 0)),
             (2.3468840079344107, (4, 3, 0)),
             (-12.516714708983523, (0, 3, 4)),
             (37.55014412695057, (2, 1, 4)),
             (1.4081304047606462, (6, 1, 0)),
             (9.387536031737643, (0, 5, 2)),
             (-28.162608095212928, (4, 1, 2))],
            [(13.27598077334948, (3, 3, 1)),
             (6.63799038667474, (1, 5, 1)),
             (-35.402615395598616, (3, 1, 3)),
             (21.24156923735917, (1, 1, 5)),
             (-35.402615395598616, (1, 3, 3)),
             (6.63799038667474, (5, 1, 1))],
            [(-0.4516580379125865, (0, 7, 0)),
             (10.839792909902076, (0, 5, 2)),
             (-1.3549741137377596, (2, 5, 0)),
             (-1.3549741137377596, (4, 3, 0)),
             (-21.679585819804153, (0, 3, 4)),
             (-21.679585819804153, (2, 1, 4)),
             (5.781222885281108, (0, 1, 6)),
             (-0.4516580379125865, (6, 1, 0)),
             (21.679585819804153, (2, 3, 2)),
             (10.839792909902076, (4, 1, 2))],
            [(-11.471758521216831, (2, 0, 5)),
             (1.0925484305920792, (0, 0, 7)),
             (-11.471758521216831, (0, 2, 5)),
             (28.67939630304208, (2, 2, 3)),
             (-2.3899496919201733, (6, 0, 1)),
             (-7.16984907576052, (4, 2, 1)),
             (14.33969815152104, (4, 0, 3)),
             (-2.3899496919201733, (0, 6, 1)),
             (-7.16984907576052, (2, 4, 1)),
             (14.33969815152104, (0, 4, 3))],
            [(10.839792909902076, (1, 4, 2)),
             (-0.4516580379125865, (7, 0, 0)),
             (21.679585819804153, (3, 2, 2)),
             (-1.3549741137377596, (5, 2, 0)),
             (-0.4516580379125865, (1, 6, 0)),
             (-21.679585819804153, (3, 0, 4)),
             (-1.3549741137377596, (3, 4, 0)),
             (5.781222885281108, (1, 0, 6)),
             (-21.679585819804153, (1, 2, 4)),
             (10.839792909902076, (5, 0, 2))],
            [(10.620784618679584, (2, 0, 5)),
             (-10.620784618679584, (0, 2, 5)),
             (3.31899519333737, (6, 0, 1)),
             (3.31899519333737, (4, 2, 1)),
             (-17.701307697799308, (4, 0, 3)),
             (-3.31899519333737, (0, 6, 1)),
             (-3.31899519333737, (2, 4, 1)),
             (17.701307697799308, (0, 4, 3))],
            [(-1.4081304047606462, (1, 6, 0)),
             (0.4693768015868821, (7, 0, 0)),
             (18.775072063475285, (3, 2, 2)),
             (-0.4693768015868821, (5, 2, 0)),
             (12.516714708983523, (3, 0, 4)),
             (-2.3468840079344107, (3, 4, 0)),
             (28.162608095212928, (1, 4, 2)),
             (-37.55014412695057, (1, 2, 4)),
             (-9.387536031737643, (5, 0, 2))],
            [(10.378311574405206, (4, 0, 3)),
             (-3.1134934723215615, (0, 6, 1)),
             (15.56746736160781, (2, 4, 1)),
             (-62.26986944643124, (2, 2, 3)),
             (10.378311574405206, (0, 4, 3)),
             (-3.1134934723215615, (6, 0, 1)),
             (15.56746736160781, (4, 2, 1))],
            [(-2.5945778936013015, (1, 6, 0)),
             (-62.26986944643124, (3, 2, 2)),
             (-0.5189155787202603, (7, 0, 0)),
             (31.13493472321562, (1, 4, 2)),
             (2.5945778936013015, (3, 4, 0)),
             (6.226986944643123, (5, 0, 2)),
             (4.670240208482342, (5, 2, 0))],
            [(2.6459606618019005, (6, 0, 1)),
             (-2.6459606618019005, (0, 6, 1)),
             (-39.68940992702851, (4, 2, 1)),
             (39.68940992702851, (2, 4, 1))],
            [(0.7071627325245963, (7, 0, 0)),
             (-14.850417383016522, (5, 2, 0)),
             (24.75069563836087, (3, 4, 0)),
             (-4.950139127672174, (1, 6, 0))]]
        self.YL = YL
        return


class GPaw:
    def __init__(self, fname: str='N.PBE.gz'):
        self.parse_psfile(fname)
        self.calculate_compensation_charges()
        self.calculate_integral_potentials()
        self.calculate_Delta_lq()

        _np = self.ni * (self.ni + 1) // 2
        self.calculate_T_Lqp(self.lcut, _np, self.nj, self.jlL_i)
        self.calculate_coulomb_corrections(
                self.wn_lqg, self.wnt_lqg, self.wg_lg, self.wnc_g, self.wmct_g)
        return


    def parse_psfile(self, fname:str):
        txt  = gzip.open(fname, 'rb').read()
        tree = ET.fromstring(txt)

        self.build_atom_spec(tree.find('atom').attrib)
        self.build_radial_grid(tree.find('radial_grid').attrib)
        self.build_l(tree)
        self.build_shape_function(tree.find('shape_function').attrib, self.lmax)
        self.build_paw_functions(tree)
        return


    def build_l(self, tree):
        self.l_j  = [int(x.attrib['l']) for x in tree.find('valence_states')]
        self.lmax = 2
        self.lcut = max(self.l_j)

        rcut_j    = [float(x.attrib['rc']) for x in tree.find('valence_states')]
        rcut2     = max(rcut_j) * 2

        self.rcut_j = rcut_j
        self.rcut2  = rcut2
        self.gcut2  = np.searchsorted(self.r_g, rcut2)

        self.n_j    = [int(x.attrib.get('n',    -1)) for x in tree.find('valence_states')]
        self.f_j    = [float(x.attrib.get('f',   0)) for x in tree.find('valence_states')]
        self.eps_j  = [float(x.attrib['e'])          for x in tree.find('valence_states')]
        self.rcut_j = [float(x.attrib.get('rc', -1)) for x in tree.find('valence_states')]
        self.id_j   = [x.attrib['id']                for x in tree.find('valence_states')]

        jlL_i = [(j, l, l**2+m)
                 for (j, l) in enumerate(self.l_j)
                 for m in range(2*l + 1)]

        self.jlL_i = jlL_i
        self.ni = len(jlL_i)
        self.nj = len(self.l_j)

        self.nq = self.nj * (self.nj + 1) // 2
        return


    def build_atom_spec(self, dic: dict):
        self.symbol  = dic['symbol']
        self.Z       = int(float(dic['Z']))
        self.core    = int(float(dic['core']))
        self.valence = int(float(dic['valence']))
        return


    def build_radial_grid(self, dic: dict):
        eq = dic['eq']
        if 'r=a*i/(n-i)' == eq:
            n = int(dic['n'])
            a = float(dic['a']) / n
            b = 1.0 / n
        elif 'r=a*i/(1-b*i)' == eq:
            a = float(dic['a'])
            b = float(dic['b'])
            n = int(dic['n'])
        else:
            raise ValueError(f"Invalid radial grid eq: {eq}")

        g    = np.arange(n)
        r_g  = a * g / (1 - b * g)
        dr_g = (b * r_g + a)**2 / a

        self.r_g   = r_g
        self.dr_g  = dr_g
        self.rdr_g = r_g * dr_g
        return


    def build_shape_function(self, dic: dict, lmax: int):
        r_g  = self.r_g
        ng   = r_g.size
        g_lg = np.zeros((lmax+1, ng))

        typ = dic['type']
        rc  = float(dic['rc'])
        if 'gauss' == typ:
            g_lg[0,:] = 4 / rc**3 / np.sqrt(np.pi) * np.exp(-(r_g / rc)**2)
            for l in range(1, lmax+1):
                g_lg[l,:] = 2.0 / (2*l + 1) / rc**2 * r_g * g_lg[l-1,:]
        else:
            raise ValueError(f"Invalid type of shape function: {typ}")

        for l in range(lmax+1):
            g_lg[l] /= self.rgd_integrate(g_lg[l], l) / (4 * np.pi)

        self.g_lg = g_lg
        return


    def build_paw_functions(self, tree):
        self.nc_g = np.array(
                tree.find('ae_core_density').text.split(),
                dtype=float)
        self.nct_g = np.array(
                tree.find('pseudo_core_density').text.split(),
                dtype=float)
        self.phi_g = np.array(
                [ae.text.split() for ae in tree.findall('ae_partial_wave')],
                dtype=float)
        self.phit_g = np.array(
                [ae.text.split() for ae in tree.findall('pseudo_partial_wave')],
                dtype=float)
        return


    def rgd_integrate(self, a_xg: npt.NDArray, n: int=0):
        assert n >= -2
        return np.dot(a_xg[..., 1:],
                      (self.r_g**(2+n) * self.dr_g)[1:]) * (4 * np.pi)


    # 69us, corrected now
    @staticmethod
    def hartree(l: int, nrdr: npt.NDArray, r: npt.NDArray):
        M = nrdr.size
        vr = np.zeros(M, dtype=float)

        rl   = r[1:]**l
        rlp1 = rl * r[1:]
        dp   = nrdr[1:] / rl
        dq   = nrdr[1:] * rlp1
        dpfl = np.flip(dp)
        dqfl = np.flip(dq)

        p = np.flip(np.r_[0, np.cumsum(dpfl)])    # prepend 0 to cumsum
        q = np.flip(np.r_[0, np.cumsum(dqfl)])

        vr[1:] = (p[1:] + 0.5 * dp) * rlp1 - (q[1:] + 0.5 * dq) / rl
        vr[0] = 0.0

        f = 4.0 * np.pi / (2 * l + 1)
        vr[1:] = f * (vr[1:] + q[0] / rl)
        return vr


    ## 424us, reference implementation from gpaw/c/utilities.c
    # @staticmethod
    # def hartree2(l, nrdr, r):
    #     M = nrdr.size
    #
    #     vr = np.zeros(M)
    #     
    #     p = 0.0
    #     q = 0.0
    #
    #     for g in range(M-1, 0, -1):
    #         R = r[g]
    #         rl = R**l
    #         rlp1 = rl * R
    #         dp = nrdr[g] / rl
    #         dq = nrdr[g] * rlp1
    #         vr[g] = (p + 0.5 * dp) * rlp1 - (q + 0.5 * dq) / rl
    #         p += dp
    #         q += dq
    #
    #     vr[0] = 0.0
    #     f = 4.0 * np.pi / (2 * l + 1)
    #     
    #     for g in range(1, M):
    #         R = r[g]
    #         vr[g] = f * (vr[g] + q / R**l)
    #     return vr


    def calculate_compensation_charges(self):
        # lmax  = self.lmax
        gcut2 = self.gcut2
        # g_lg  = self.g_lg
        nq    = self.nq

        phi_g  = self.phi_g[:,:gcut2]
        phit_g = self.phit_g[:,:gcut2]

        n_qg  = np.zeros((nq, gcut2))
        nt_qg = np.zeros((nq, gcut2))
        for (q, (j1, j2)) in enumerate([(j1, j2) for j1 in range(self.nj)
                                                 for j2 in range(j1,self.nj)]):
            n_qg[q,:]  = phi_g[j1,:]  * phi_g[j2,:]                     #  phi_i1^a *  phi_i2^a
            nt_qg[q,:] = phit_g[j1,:] * phit_g[j2,:]                    # ~phi_i1^a * ~phi_i2^a

        self.n_qg  = n_qg
        self.nt_qg = nt_qg

        ## Delta0 is an constant for each atom
        self.Delta0 = np.dot(self.nc_g[:gcut2] - self.nct_g[:gcut2],    # 
                             self.rdr_g[:gcut2] * self.r_g[:gcut2]) - self.Z / np.sqrt(4 * np.pi)
        return


    def poisson(self, n_g, l=0, *, s:slice=slice(None)):
        n_g    = n_g[s].copy()
        nrdr_g = n_g[s] * self.rdr_g[s]
        return GPaw.hartree(l, nrdr_g, self.r_g[s])


    def calculate_integral_potentials(self):
        def H(n_g, l, *, s:slice=slice(None)):
            return self.poisson(n_g[s], l, s=s) * self.r_g[s] * self.dr_g[s]

        gcut2 = self.gcut2

        wg_lg   = [H(self.g_lg[l,:], l, s=slice(None,gcut2))
                   for l in range(self.lmax + 1)]                       # ((~g_l^a)) in Eq (47)
        wn_lqg  = [np.array([H(self.n_qg[q,:], l, s=slice(None,gcut2))
                             for q in range(self.nq)])
                   for l in range(2*self.lcut + 1)]                     # ( phi_i1^a *  phi_i2^a  |   phi_i3^a *  phi_i4^a * r^l) in Eq (47)
        wnt_lqg = [np.array([H(self.nt_qg[q,:], l, s=slice(None,gcut2))
                             for q in range(self.nq)])
                   for l in range(2*self.lcut + 1)]                     # (~phi_i1^a * ~phi_i2^a  |  ~phi_i3^a * ~phi_i4^a * r^l) in Eq (47)

        wnc_g  = H(self.nc_g[:],  l=0, s=slice(None,gcut2))             # (( n_c^a))
        wnct_g = H(self.nct_g[:], l=0, s=slice(None,gcut2))             # ((~n_c^a))
        wmct_g = wnct_g + self.Delta0 * wg_lg[0]

        self.wg_lg   = wg_lg
        self.wn_lqg  = wn_lqg
        self.wnt_lqg = wnt_lqg
        self.wnc_g   = wnc_g
        self.wnct_g  = wnct_g
        self.wmct_g  = wmct_g
        return


    def calculate_T_Lqp(self, lcut, _np, nj, jlL_i):
        '''
        T_Lqp is the Gaunt-Coefficients.

        T_Lqp[L, l1 l2, p1 p2] = G_LLL[L1, L2, L]
        '''

        Lcut  = (2*lcut + 1)**2
        gaunt = Gaunt()
        G_LLL = gaunt(max(self.l_j))[:, :, :Lcut]
        LGcut = G_LLL.shape[2]
        T_Lqp = np.zeros((Lcut, self.nq, _np))

        p  = 0
        i1 = 0

        for j1, l1, L1 in jlL_i:
            for j2, l2, L2 in jlL_i[i1:]:
                if j1 < j2:
                    q = j2 + j1 * nj - j1 * (j1 + 1) // 2
                else:
                    q = j1 + j2 * nj - j2 * (j2 + 1) // 2
                T_Lqp[:LGcut, q, p] = G_LLL[L1, L2, :]
                p += 1
            i1 += 1
        self.T_Lqp = T_Lqp
        return


    def calculate_Delta_lq(self):
        gcut2 = self.gcut2
        r_g   = self.r_g[:gcut2]
        dr_g  = self.dr_g[:gcut2]
        n_qg  = self.n_qg
        nt_qg = self.nt_qg

        Delta_lq = np.zeros((self.lmax + 1, self.nq))
        for l in range(self.lmax + 1):
            Delta_lq[l,:] = (n_qg - nt_qg) @ (r_g**(l+2) * dr_g)        # \sum dr * r^l * ( phi_i1 *  phi_i2 - ~phi_i1 * ~phi_i2)
        self.Delta_lq = Delta_lq                                        # Delta_Li1i2^a in Eq (41b) (without Y_L(r) part)
        return


    def calculate_coulomb_corrections(self, wn_lqg, wnt_lqg, wg_lg, wnc_g, wmct_g):
        gcut2 = self.gcut2
        _np   = self.ni * (self.ni + 1) // 2
        mct_g = self.nct_g[:gcut2] + self.Delta0 * self.g_lg[0, :gcut2] # ~n_c^a + Delta^a * ~g_00^a
        rdr_g = self.r_g[:gcut2] * self.dr_g[:gcut2]

        A_q  = 0.5 * (wn_lqg[0] @ self.nc_g[:gcut2] + self.n_qg @ wnc_g)        # (phi_i1 * phi_i2 | n_c^a) + 
        A_q -= np.sqrt(4 * np.pi) * self.Z * (self.n_qg @ rdr_g)
        A_q -= 0.5 * (wn_lqg[0] @ mct_g + self.nt_qg @ wmct_g)
        A_q -= 0.5 * (mct_g @ wg_lg[0] + self.g_lg[0,:gcut2] @ wmct_g) * self.Delta_lq[0,:]
        M_p  = A_q @ self.T_Lqp[0]                                      # DeltaC_i1i2^a in Eq (46)

        A_lqq = []
        for l in range(2 * self.lcut + 1):
            A_qq  = 0.5 * self.n_qg @ wn_lqg[l].T
            A_qq -= 0.5 * self.nt_qg @ wnt_lqg[l].T                     # 1/2 * [ ( | ) - ( ~ | ~ ) ] in Eq (47)

            if l <= self.lmax:
                A_qq -= 0.5 * np.outer(self.Delta_lq[l,:],              # 1/2 * Delta_Li1i2^a (~phi_i1^a * ~phi_i2^a | ~g_l^a)
                                       wnt_lqg[l] @ self.g_lg[l,:gcut2])
                A_qq -= 0.5 * np.outer(self.nt_qg @ wg_lg[l],           # 1/2 * Delta_Li3i4^a (~phi_i3^a * ~phi_i4^a | ~g_l^a)
                                       self.Delta_lq[l,:])
                A_qq -= 0.5 * ((self.g_lg[l,:gcut2] @ wg_lg[l])         # Delta_Li1i2^a * ((~g_l^a)) * Delta_Li3i4
                               * np.outer(self.Delta_lq[l], self.Delta_lq[l]))
            A_lqq.append(A_qq)
            pass

        M_pp = np.zeros((_np, _np))                                     # DeltaC_i1i2i3i4^a in Eq (47)
        L = 0
        for l in range(2 * self.lcut + 1):
            for m in range(2 * l + 1):
                M_pp += self.T_Lqp[L].T @ A_lqq[l] @ self.T_Lqp[L]      # Multiple all the quantities with the angular term
                L += 1

        self.M_p  = M_p
        self.M_pp = M_pp
        return


    def get_coulomb_corrections(self):
        # if not hasattr(self, 'M_pp') or not hasattr(self, 'M_p'):
        #     self.calculate_coulomb_corrections(
        #         self.wn_lqg, self.wnt_lqg, self.wg_lg, self.wnc_g, self.wmct_g
        #     )
        return self.M_p, self.M_pp


def pack(A) -> npt.NDArray:
    ni = A.shape[0]
    N  = ni * (ni + 1) // 2
    B  = np.zeros(N, dtype=A.dtype)

    k = 0
    for i in range(ni):
        B[k] = A[i,i]
        k += 1
        for j in range(i+1, ni):
            B[k] = A[i,j] + A[j,i]
            k += 1
    return B


if '__main__' == __name__:
    gaunt = Gaunt()
    gp = GPaw("./H.PBE.gz")
    M_p, M_pp = gp.get_coulomb_corrections()

    # for fname in glob('/public/home/chenlj/.gpaw/gpaw-setups-0.9.20000/*.gz'):
        # if 'basis.gz' in fname:
            # continue
        # print(fname)
        # xx = GPaw(fname)
        # M_p, M_pp = xx.get_coulomb_corrections()
        # print(M_pp)
