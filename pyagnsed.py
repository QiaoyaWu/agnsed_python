#!/usr/bin/python3

import os,sys,time
import numpy as np
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
from astropy import units as u
from astropy import constants as const
from astropy.modeling.models import BlackBody
from scipy import integrate
from scipy import interpolate
import scipy.optimize as opt
from scipy.interpolate import UnivariateSpline
import pickle
import numba

class agnsed:
    def __init__(self, 
        energy_bin = None,
        MBH=1e8,
        logmdot=-1,
        a_spin=0.,
        cosi = 0.5,
        colortemp_correct = -1,
        if_qsosed = True,
        kT_hot=100,
        kT_warm=0.2,
        Gamma_hot=2.7,
        Gamma_warm=2.5,
        r_hot=-1,
        r_warm=-1,
        logr_out=-1,
        hmax_hot=100,
        method = 'KD18',
        solv_slim_args = None,
        if_reprocessing = True,
        alpha = 0.1,
        reflection_albedo = 0.3,
        verbose = False):

        self.verbose = verbose
        self.reflection_albedo = reflection_albedo 
        self.alpha = alpha
        
        ### generate the energy grid
        if energy_bin is None:
            self.Ebins = np.geomspace(1e-4, 5e4, num=50000)
        else:
            self.Ebins = energy_bin

        self.dEs = np.diff(self.Ebins)
        self.Egrid = self.Ebins[:-1] + 0.5 * self.dEs
        self.nu_arr = (self.Ebins*u.keV / const.h).to(u.Hz)

        ### Setting up the parameters
        self.MBH = MBH * const.M_sun # solar mass
        self.logmdot = logmdot # dimensionless
        self.mdot = 10**logmdot # in units of Mdot_edd

        if a_spin >= -0.998 and a_spin <= 0.998:
            self.a_spin = a_spin
        else:
            raise ValueError(f"Spin parameter {a_spin} is unphysical. Spin must be between -0.998 and 0.998")
        
        if cosi >=0.087 and cosi <= 0.985:
            self.cosi = cosi
        else:
            raise ValueError(f"Inclination out of bounds - will not work with kyconv! \n'+\
                             'Translates to: 10 <= inc <= 85 deg")

        ### Calculating params 
        r_isco = self._compute_r_isco(a_spin) # in unit of Rg
        r_sg = self._compute_r_selfGravity(MBH, self.mdot) # in unit of Rg
        L_edd = self._compute_Ledd() # erg/s
        eta = 0.1
        self.eta = eta
        #eta = self._compute_efficiency(r_isco) # dimensionless

        self.fcol_corr = colortemp_correct
        self.Mdot_edd = (L_edd/(eta * const.c**2)).cgs
        self.Mdot_crit = (L_edd/const.c**2).cgs
        self.Mdot = self.mdot * self.Mdot_edd # in g/s
        self.mdot_crit = (self.Mdot/self.Mdot_crit).cgs 
        self.Rg = (const.G * self.MBH/const.c**2).cgs

        if logr_out == -1:
            self.logr_out = np.log10(r_sg)
        else:   
            self.logr_out = logr_out

        
        self.if_reprocessing = if_reprocessing
        if method is None:
            self.method = 'KD18'
        elif method in ['KD18', 'KD19', 'Slim']:
            self.method = method
        else:
            raise ValueError(f"Method {method} not recognized. Choose from 'KD18', 'KD19', or 'Slim'.")
        #self.if_slimdisk = if_slimdisk

        if self.method=='KD19': #(self.if_slimdisk == True):
            # Eq1 in Kubota & Done2019
            r_horizon = 1+np.sqrt(1-a_spin*a_spin)
            if self.mdot <= 6:
                self.r_in = self.r_isco
            elif 6 < self.mdot <= 100:
                mod_rin = (self.mdot/6)**(np.log10(r_horizon/self.r_isco)/np.log10(100/6))
                self.r_in = self.r_isco * mod_rin
            elif self.mdot > 100:
                self.r_in = r_horizon

            # Find the r_bc and r_crit in the disk
            temp_r = np.geomspace(self.r_in, self.r_sg, num=1000)
            temp_fnt_fedd = interpolate.UnivariateSpline(temp_r, (self._compute_NT_temperature4(temp_r)*u.K**4 * const.sigma_sb).cgs.value - (self.L_edd/(4 * np.pi * (temp_r * self.Rg)**2)).cgs.value, s=0)
            r_super_edd = temp_fnt_fedd.roots()
            if self.mdot <= 6:
                if len(r_super_edd) == 2:
                    self.r_super_edd = r_super_edd
                elif len(r_super_edd) == 0:
                    self.method = 'KD18'
                    #self.if_slimdisk = False
                    if self.verbose:  print("Accretion rate is too low to be super-Eddington. Slim disk model not applied.")
                elif len(r_super_edd) == 1:
                    if (temp_fnt_fedd(self.r_in) > 0) and (temp_fnt_fedd(self.r_sg) < 0):
                        self.r_super_edd = [self.r_in, r_super_edd[0]]
                    elif (temp_fnt_fedd(self.r_in) < 0) and (temp_fnt_fedd(self.r_sg) > 0):
                        self.r_super_edd = [r_super_edd[0], self.r_sg]
                    else:
                        self.method = 'KD18'
                        #self.if_slimdisk = False
                        if self.verbose:  print("Problems occur when computing the super-Edding emissivity radii. Slim disk model not applied.")
            else:
                ind_rcrit = np.where((r_super_edd>self.r_isco) & (r_super_edd<self.r_sg))[0]
                if len(ind_rcrit) >= 1:
                    self.r_crit = r_super_edd[ind_rcrit][-1]
                else:
                    self.r_crit = None
        elif self.method=='Slim': 
            if solv_slim_args is not None:
                self.solv_slim_args = solv_slim_args
            else:
                self.solv_slim_args = {}
            self.r_in = 3
            if 'load_file' in self.solv_slim_args:
                file_path = self.solv_slim_args['load_file']
                tmp_model = pickle.load(open(file_path, 'rb'))
                self.slim_quantities = tmp_model.slim_quantities
                self.r_transonic = tmp_model.r_transonic
                self.slim_lin = tmp_model.slim_lin
                self.r_in_slim = tmp_model.r_in_slim
            else:
                self.slim_quantities = self._compute_slimdisk_equation_WSig()
        else:
            self.r_in = self.r_isco

        if self.verbose:
            print(f"MBH = {(self.MBH/const.M_sun).cgs}, mdot = {self.mdot}, a_spin = {self.a_spin}, cosi = {self.cosi}")
            print(f"Mdot = {self.Mdot.cgs}, Mdot_edd = {self.Mdot_edd.cgs}, L_edd = {L_edd}")
            print(f"mdot_crit = {self.mdot_crit}, eta = {eta}")
            print(f"Rg = {self.Rg.cgs}, logr_out = {self.logr_out}, r_isco = {r_isco}, r_sg = {r_sg}")
            print('Method = ', self.method, ', r_in = ', self.r_in)

        if if_qsosed:
            self.f_hard_Xray_dissipated = 0.02 # fraction of Eddington luminosity dissipated in the hot corona
            self.L_corona_dissipated = self.f_hard_Xray_dissipated * self.L_edd
            self.r_hot = self._compute_r_hot_corona()
            self.r_warm = 2 * self.r_hot
            if (self.r_hot == self.r_isco) or (self.logmdot<=-1.5):
                if_trans_ADAF = True
            else:
                if_trans_ADAF = False
        else:
            if (self.r_in < r_hot < r_sg) and (self.r_in < r_warm < r_sg) and (r_hot < r_warm):
                self.r_hot = r_hot
                self.L_corona_dissipated = self._compute_L_corona_dissipated()
                self.f_hard_Xray_dissipated = self.L_corona_dissipated / self.L_edd
                self.r_warm = r_warm
            else:
                raise ValueError(f"Warm and hot corona radii are unphysical. Must satisfy: \n'+\
                                 'r_isco < r_hot < r_warm < r_sg")
                
        if if_qsosed:
            self.hmax_hot = min(100.0, self.r_hot)
        else:
            if hmax_hot > self.r_hot:
                if verbose: print(f"Hot corona height must be smaller than the hot corona radius. set: hmax_hot = {hmax_hot}, r_hot = {self.r_hot}")
                self.hmax_hot = self.r_hot
            else:
                self.hmax_hot = hmax_hot
        
        self.L_corona_seed = self._compute_L_corona_seed(self.r_hot, self.r_sg)  

        if if_qsosed:
            if if_trans_ADAF:
                self.kT_hot = 100 
                self.kT_warm = 0.2
                self.Gamma_hot = 1.7
                self.Gamma_warm = 2.5
                self.r_warm = 200
                self.r_hot = 200
                mdot_ref = 10**(-1.5)
                q_riaf = 1.
                self.f_hard_Xray_dissipated = min(0.02, 0.02 * (self.mdot / mdot_ref)**q_riaf)
                self.L_corona_dissipated = self.f_hard_Xray_dissipated * self.L_edd
            else:
                self.kT_hot = 100 
                self.kT_warm = 0.2
                self.Gamma_warm = 2.5
                self.Gamma_hot = self._compute_Gamma_hot()
        else:
            self.kT_hot = kT_hot
            self.kT_warm = kT_warm
            self.Gamma_hot = Gamma_hot
            self.Gamma_warm = Gamma_warm
        
        if self.verbose:
            print(f"r_hot = {self.r_hot}, r_warm = {self.r_warm}")
            print(f"kT_hot = {self.kT_hot}, kT_warm = {self.kT_warm}")
            print(f"Gamma_hot = {self.Gamma_hot}, Gamma_warm = {self.Gamma_warm}")
            print(f"hmax_hot = {self.hmax_hot}")
    """
    Funtions for accretion paramaters, comptonization from xspec
    """

    def _compute_r_isco(self, a_spin):
        """
        Calculating innermost stable circular orbit for a spinning
        black hole. Follows Page and Thorne (1974). Note, in the litterature
        this is also reffered to as r_ms, for marginally stable orbit.
        We will stick to r_isco throughout!

        """

        Z1 = 1 + (1 - a_spin**2)**(1/3) * (
            (1 + a_spin)**(1/3) + (1 - a_spin)**(1/3))
        Z2 = np.sqrt(3 * a_spin**2 + Z1**2)

        r_isco = 3 + Z2 - np.sign(a_spin) * np.sqrt(
            (3 - Z1) * (3 + Z1 + 2*Z2))
        self.r_isco = r_isco

        return r_isco

    def _compute_r_selfGravity(self, MBH, mdot):
        """
        Calcultes the self gravity radius according to Laor & Netzer 1989
        
        NOTE: Assuming that alpha=0.1 

        """
        #See Laor & Netzer 1989 for more details on constraining this parameter
        m9 = MBH/1e9
        r_sg = 2150 * m9**(-2/9.) * mdot**(4/9.) * self.alpha**(2/9.)
        self.r_sg = r_sg
        return r_sg

    def _compute_Ledd(self):
        """
        Calculates the Eddington luminosity for a black hole of mass M.
        Ledd = 1.26e38 * M/Msol erg/s
        Note: xspec-qsosed used 1.39e38 for Ledd
        """
        #L_edd = 1.39e38 * MBH * u.erg/u.s
        L_edd = (4 * np.pi * const.G * const.m_p * const.c * self.MBH / const.sigma_T ).cgs
        self.L_edd = L_edd
        return L_edd

    def _compute_efficiency(self, r_isco):
        """
        Calculates the accretion efficiency eta, s.t L_bol = eta Mdot c^2
        Using the GR case, where eta = 1 - sqrt(1 - 2/(3 r_isco)) 
            Taken from: The Physcis and Evolution of Active Galactic Nuceli,
            H. Netzer, 2013, p.38
        """
        eta = 1 - np.sqrt(1 - 2/(3*r_isco))
        self.eta = eta
        return eta

    def _compute_r_hot_corona(self):
        try:
            if self.method == 'Slim':
                r_hot = opt.brentq(lambda r: self._disk_blackbody_luminosity(self.r_isco, r).value - self.L_corona_dissipated.value, 
                                self.r_isco, self.r_sg)
            else:
                r_hot = opt.brentq(lambda r: self._disk_NT_blackbody_luminosity(self.r_isco, r).value - self.L_corona_dissipated.value, 
                                self.r_isco, self.r_sg)
        except:
            if self.verbose: print("Accretion rate is too low to power a corona. Radius is smaller than last circular stable orbit.")
            r_hot = self.r_isco.copy()
        return r_hot
    
    def _compute_donthcomp(self, ear, param):
        """
        adopted from pyqsosed github repository

        This function was adapted by ADT from the subroutine donthcomp in
        donthcomp.f, distributed with XSpec.
        Nthcomp documentation:
        https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodelNthcomp.html
        Refs:
        Zdziarski, Johnson & Magdziarz 1996, MNRAS, 283, 193,
        as extended by Zycki, Done & Smith 1999, MNRAS 309, 561
        Note that the subroutine has been modified so that parameter 4
        is ignored, and the seed spectrum is always a blackbody.
        ear: Energy vector, listing "Energy At Right" of bins (keV)
        param: list of parameters; see the 5 parameters listed below.
        The original fortran documentation for this subroutine is included below:
        Driver for the Comptonization code solving Kompaneets equation
        seed photons  -  (disk) blackbody
        reflection + Fe line with smearing
        
        Model parameters:
        1: photon spectral index
        2: plasma temperature in keV
        3: (disk)blackbody temperature in keV
        4: type of seed spectrum (0 - blackbody, 1 - diskbb)
        5: redshift
        """
        param = np.array(param)
        param = np.insert(param,0,0)
        ne = ear.size  # Length of energy bin vector
        # Note that this model does not calculate errors.
        #c     xth is the energy array (units m_e c^2)
        #c     spnth is the nonthermal spectrum alone (E F_E)
        #c     sptot is the total spectrum array (E F_E), = spref if no reflection
        zfactor = 1.0 + param[5]
        #c  calculate internal source spectrum
        #                           blackbody temp,   plasma temp,      Gamma
        xth, nth, spt = self._compute_thcompton(param[3] / 511.0, param[2] / 511.0, param[1])
        # The temperatures are normalized by 511 keV, the electron rest energy
        # Calculate normfac:
        xninv = 511.0 / zfactor
        ih = 1
        xx = 1.0 / xninv
        while (ih < nth and xx > xth[ih]):
            ih = ih + 1
        il = ih - 1
        spp = spt[il] + (spt[ih] - spt[il]) * (xx - xth[il]) / (xth[ih] - xth[il])
        normfac = 1.0 / spp

        #c     zero arrays
        photar = np.zeros(ne)
        prim   = np.zeros(ne)
        #c     put primary into final array only if scale >= 0.
        j = 0
        for i in range(0, ne):
            while (j <= nth and 511.0 * xth[j] < ear[i] * zfactor):
                j = j + 1
            if (j <= nth):
                if (j > 0):
                    jl = j - 1
                    prim[i] = spt[jl] + ((ear[i] / 511.0 * zfactor - xth[jl]) * 
                                        (spt[jl + 1] - spt[jl]) / 
                                        (xth[jl + 1] - xth[jl])                 )
                else:
                    prim[i] = spt[0]
        for i in range(1, ne):
            photar[i] = (0.5 * (prim[i] / ear[i]**2 + prim[i - 1] / ear[i - 1]**2) 
                            * (ear[i] - ear[i - 1]) * normfac                    )

        return photar
    
    def _compute_thcompton(self, tempbb, theta, gamma):
        """
        Adopted from pyqsosed github repository

        This function was adapted by ADT from the subroutine thcompton in
        donthcomp.f, distributed with XSpec.
        Nthcomp documentation:
        https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodelNthcomp.html
        Refs:
        Zdziarski, Johnson & Magdziarz 1996, MNRAS, 283, 193,
        as extended by Zycki, Done & Smith 1999, MNRAS 309, 561
        The original fortran documentation for this subroutine is included below:
        Thermal Comptonization; solves Kompaneets eq. with some
        relativistic corrections. See Lightman \ Zdziarski (1987), ApJ
        The seed spectrum is a blackbody.
        version: January 96
        #c  input parameters:
        #real * 8 tempbb,theta,gamma
        """
        #c use internally Thomson optical depth
        tautom = np.sqrt(2.250 + 3.0 / (theta * ((gamma + .50)**2 - 2.250))) - 1.50

        # Initialise arrays
        dphdot = np.zeros(900); rel = np.zeros(900); c2 = np.zeros(900)
        sptot  = np.zeros(900); bet = np.zeros(900); x  = np.zeros(900)

        #c JMAX  -  # OF PHOTON ENERGIES
        #c delta is the 10 - log interval of the photon array.
        delta = 0.02
        deltal = delta * np.log(10.0)
        xmin = 1e-4 * tempbb
        xmax = 40.0 * theta
        jmax = min(899, int(np.log10(xmax / xmin) / delta) + 1)

        #c X  -  ARRAY FOR PHOTON ENERGIES
        # Energy array is normalized by 511 keV, the rest energy of an electron
        x[:(jmax + 1)] = xmin * 10.0**(np.arange(jmax + 1) * delta)

        #c compute c2(x), and rel(x) arrays
        #c c2(x) is the relativistic correction to Kompaneets equation
        #c rel(x) is the Klein - Nishina cross section divided by the
        #c Thomson crossection
        for j in range(0, jmax):
            w = x[j]
        #c c2 is the Cooper's coefficient calculated at w1
        #c w1 is x(j + 1 / 2) (x(i) defined up to jmax + 1)
            w1 = np.sqrt(x[j] * x[j + 1])
            c2[j] = (w1**4 / (1.0 + 4.60 * w1 + 1.1 * w1 * w1))
            if (w <= 0.05):
                #c use asymptotic limit for rel(x) for x less than 0.05
                rel[j] = (1.0 - 2.0 * w + 26.0 * w * w * 0.2)
            else:
                z1 = (1.0 + w) / w**3
                z2 = 1.0 + 2.0 * w
                z3 = np.log(z2)
                z4 = 2.0 * w * (1.0 + w) / z2
                z5 = z3 / 2.0 / w
                z6 = (1.0 + 3.0 * w) / z2 / z2
                rel[j] = (0.75 * (z1 * (z4 - z3) + z5 - z6))

        #c the thermal emission spectrum
        jmaxth = min(900, int(np.log10(50 * tempbb / xmin) / delta))
        if (jmaxth > jmax):
            jmaxth = jmax
        planck = 15.0 / (np.pi * tempbb)**4
        dphdot[:jmaxth] = planck * x[:jmaxth]**2 / (np.exp(x[:jmaxth] / tempbb)-1)

        #c compute beta array, the probability of escape per Thomson time.
        #c bet evaluated for spherical geometry and nearly uniform sources.
        #c Between x = 0.1 and 1.0, a function flz modifies beta to allow
        #c the increasingly large energy change per scattering to gradually
        #c eliminate spatial diffusion
        jnr  = int(np.log10(0.10 / xmin) / delta + 1)
        jnr  = min(jnr, jmax - 1)
        jrel = int(np.log10(1 / xmin) / delta + 1)
        jrel = min(jrel, jmax)
        xnr  = x[jnr - 1]
        xr   = x[jrel - 1]
        for j in range(0, jnr - 1):
            taukn = tautom * rel[j]
            bet[j] = 1.0 / tautom / (1.0 + taukn / 3.0)
        for j in range(jnr - 1, jrel):
            taukn = tautom * rel[j]
            arg = (x[j] - xnr) / (xr - xnr)
            flz = 1 - arg
            bet[j] = 1.0 / tautom / (1.0 + taukn / 3.0 * flz)
        for j in range(jrel, jmax):
            bet[j] = 1.0 / tautom

        dphesc = self._compute_thermlc(tautom, theta, deltal, x, jmax, dphdot, bet, c2)

        #c     the spectrum in E F_E
        for j in range(0, jmax - 1):
            sptot[j] = dphesc[j] * x[j]**2

        return x, jmax, sptot

    def _compute_thermlc(self, tautom, theta, deltal, x, jmax, dphdot, bet, c2):
        """
        Adopted from pyqsosed github repository

        This function was adapted by ADT from the subroutine thermlc in 
        donthcomp.f, distributed with XSpec.
        Nthcomp documentation:
        https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodelNthcomp.html
        Refs:
        Zdziarski, Johnson & Magdziarz 1996, MNRAS, 283, 193,
        as extended by Zycki, Done & Smith 1999, MNRAS 309, 561
        The original fortran documentation for this subroutine is included below:
        This program computes the effects of Comptonization by
        nonrelativistic thermal electrons in a sphere including escape, and
        relativistic corrections up to photon energies of 1 MeV.
        the dimensionless photon energy is x = hv / (m * c * c)
        The input parameters and functions are:
        dphdot(x), the photon production rate
        tautom, the Thomson scattering depth
        theta, the temperature in units of m*c*c
        c2(x), and bet(x), the coefficients in the K - equation and the
        probability of photon escape per Thomson time, respectively,
        including Klein - Nishina corrections
        The output parameters and functions are:
        dphesc(x), the escaping photon density
        """
        dphesc = np.zeros(900)  # Initialise the output
        a = np.zeros(900); b   = np.zeros(900); c = np.zeros(900)
        d = np.zeros(900); alp = np.zeros(900); u = np.zeros(900)
        g = np.zeros(900); gam = np.zeros(900)

        #c u(x) is the dimensionless photon occupation number
        c20 = tautom / deltal

        #c determine u
        #c define coefficients going into equation
        #c a(j) * u(j + 1) + b(j) * u(j) + c(j) * u(j - 1) = d(j)
        for j in range(1, jmax - 1):
            w1 = np.sqrt( x[j] * x[j + 1] )
            w2 = np.sqrt( x[j - 1] * x[j] )
            #c  w1 is x(j + 1 / 2)
            #c  w2 is x(j - 1 / 2)
            a[j] =  -c20 * c2[j] * (theta / deltal / w1 + 0.5)
            t1 =  -c20 * c2[j] * (0.5 - theta / deltal / w1)
            t2 = c20 * c2[j - 1] * (theta / deltal / w2 + 0.5)
            t3 = x[j]**3 * (tautom * bet[j])
            b[j] = t1 + t2 + t3
            c[j] = c20 * c2[j - 1] * (0.5 - theta / deltal / w2)
            d[j] = x[j] * dphdot[j]

        #c define constants going into boundary terms
        #c u(1) = aa * u(2) (zero flux at lowest energy)
        #c u(jx2) given from region 2 above
        x32 = np.sqrt(x[0] * x[1])
        aa = (theta / deltal / x32 + 0.5) / (theta / deltal / x32 - 0.5)

        #c zero flux at the highest energy
        u[jmax - 1] = 0.0

        #c invert tridiagonal matrix
        alp[1] = b[1] + c[1] * aa
        gam[1] = a[1] / alp[1]
        for j in range(2, jmax - 1):
            alp[j] = b[j] - c[j] * gam[j - 1]
            gam[j] = a[j] / alp[j]
        g[1] = d[1] / alp[1]
        for j in range(2, jmax - 2):
            g[j] = (d[j] - c[j] * g[j - 1]) / alp[j]
        g[jmax - 2] = (d[jmax - 2] - a[jmax - 2] * u[jmax - 1] 
                                - c[jmax - 2] * g[jmax - 3]) / alp[jmax - 2]
        u[jmax - 2] = g[jmax - 2]
        for j in range(2, jmax + 1):
            jj = jmax - j
            u[jj] = g[jj] - gam[jj] * u[jj + 1]
        u[0] = aa * u[1]
        #c compute new value of dph(x) and new value of dphesc(x)
        dphesc[:jmax] = x[:jmax] * x[:jmax] * u[:jmax] * bet[:jmax] * tautom

        return dphesc

    def _compute_compton_photon_flux(self, energy_arr, params):
        nu_arr = (energy_arr*u.keV / const.h).to(u.Hz)
        photon_number_per_bin = self._compute_donthcomp(energy_arr, params) # units of Photons / cm^2 / s
        
        photon_energy_per_bin = (photon_number_per_bin /u.s/u.cm**2) * (energy_arr * u.keV) # units: KeV / cm^2 / s
        photon_flux_per_bin = (photon_energy_per_bin / nu_arr).to(u.erg / (u.s * u.cm**2 * u.Hz)) # convert to units erg/s/cm^2/Hz
        return photon_flux_per_bin

    def _compute_L_corona_dissipated(self):
        if self.r_in < self.r_hot:
            constant = 4 * np.pi * self.Rg ** 2 * const.sigma_sb 
            L_corona_dissipated = (constant * integrate.quad(lambda r: r * self._compute_disk_temperature4(r, True), self.r_in, self.r_hot)[0] * u.K**4).to(u.erg / u.s)
        else:
            L_corona_dissipated = 0. * u.erg / u.s
        return L_corona_dissipated
    
    def _compute_L_corona_seed(self, r_in, r_out):
        L_hotcorona_seed = (4 * np.pi * self.Rg ** 2 * const.sigma_sb * \
            integrate.quad(lambda r: r * self._compute_disk_temperature4(r, True)*self._compute_corona_covering_factor(r), \
                           r_in, r_out)[0]* u.K**4).to(u.erg / u.s)
        return L_hotcorona_seed

    def _compute_Gamma_hot(self):
        """
        Adopted from pyqsosed github repository

        Photon index (Gamma) for the corona SED. The functional form is assumed to be
        L_nu = k nu ^(-alpha) = k nu^( 1 - gamma ), where alpha = gamma - 1
        Computed using equation 14 of Beloborodov (1999).
        """
        # reproc = self.reprocessing
        # self.reprocessing = False
        gamma_hot = (7.0 / 3.0 * (self.L_corona_dissipated / self.L_corona_seed).value ** (-0.1))
        # self.reprocessing = reproc
        return gamma_hot
    
    def _compute_corona_covering_factor(self, r):
        """
        Adopted from pyqsosed github repository

        Corona covering factor as seen from the disk at radius r > r_cor.

        Parameters
        ----------
        r : float
            Observer disk radius.
        """

        if r < self.r_hot:
            #print("Radius smaller than corona radius!")
            return 0.
        else:
            theta_0 = np.arcsin(self.hmax_hot / r)
            covering_factor = theta_0 - 0.5 * np.sin(2 * theta_0)
            return covering_factor / np.pi

    def _compute_reprocessed_flux_r(self, radius):
        """
        Reprocessed flux as given by eq. 5 of Kubota & Done (2018).
        """
        flux_reprocessed = (const.G * self.MBH * self._hot_compton_luminosity() * self.r_hot)  \
                           / (8 * np.pi * (radius * self.Rg)**3.0 * const.c**2) \
                           * (1 - self.reflection_albedo) \
                           * (1 + (self.r_hot/radius)**2)**(-3./2.)
        return flux_reprocessed

    """
    Compute the disk component using the Novikov-Thorne model
    """

    def _compute_disk_NT_relparams(self, radius):
        """
         -- taken from relagn --
        Calculates the Novikov-Thorne relativistic factors.
        see Active Galactic Nuclei, J. H. Krolik, p.151-154
        and Page & Thorne (1974)
        
        Parameters
        ----------
        r : float OR array
            Disc radius (as measured from black hole)
            Units : Dimensionless (Rg)

        """
        y = np.sqrt(radius)
        y_isc = np.sqrt(self.r_isco.copy())
        y1 = 2 * np.cos((1/3) * np.arccos(self.a_spin) - (np.pi/3))
        y2 = 2 * np.cos((1/3) * np.arccos(self.a_spin) + (np.pi/3))
        y3 = -2 * np.cos((1/3) * np.arccos(self.a_spin))

        
        B = 1 - (3/radius) + ((2 * self.a_spin)/(radius**(3/2)))
        
        C1 = 1 - (y_isc/y) - ((3 * self.a_spin)/(2 * y)) * np.log(y/y_isc)
        
        C2 = ((3 * (y1 - self.a_spin)**2)/(y*y1 * (y1 - y2) * (y1 - y3))) * np.log(
            (y - y1)/(y_isc - y1))
        C2 += ((3 * (y2 - self.a_spin)**2)/(y*y2 * (y2 - y1) * (y2 - y3))) * np.log(
            (y - y2)/(y_isc - y2))
        C2 += ((3 * (y3 - self.a_spin)**2)/(y*y3 * (y3 - y1) * (y3 - y2))) * np.log(
            (y - y3)/(y_isc - y3))
        
        C = C1 - C2
        
        return C/B
    
    def _compute_NT_temperature4(self, radius):
        """
        -- taken from relagn --
        Computes Novikov-Thorne temperature in Kelvin (to the power of 4) of accretion disk annulus at radius r.
        Parameters
        ----------
        r : float
            disk radius in Rg.
        """
        NT_constant = (3*const.G * self.MBH * self.mdot * self.Mdot_edd) / (
            8 * np.pi * const.sigma_sb * (radius * self.Rg)**3)
        
        NT_rel_factor = self._compute_disk_NT_relparams(radius)

        Temp4 = NT_constant * NT_rel_factor
        return Temp4.cgs.value

    def _compute_SSD_rho_Tc(self, R, Rs):
        if False:
            # Frank et al. 2002
            f = (1 - np.sqrt((3*Rs/R).cgs))**(1/4.)
            rho = (3.1e-8 * self.alpha**(-7/10.) * (Mdot/1e16/u.g*u.s)**(11/20.) * (self.MBH/const.M_sun)**(5/8.) * \
                    (R/1e10/u.cm)**(-15/8.) * f**(11/5.)).cgs * u.g / u.cm**3
            Tc = (1.4e4 * self.alpha**(-1/5.) * (Mdot/1e16/u.g*u.s)**(3/10.) * (self.MBH/const.M_sun)**(1/4.) * \
                    (R/1e10/u.cm)**(-3/4.) * f**(6/5.)).cgs * u.K
        else:
            # Kato et al. 2008
            f = 1 - np.sqrt(3*Rs/R)
            rho = 4.7e1 * self.alpha**(-7/10.) * (self.MBH/const.M_sun)**(-7/10.) * self.mdot_crit**(11/20.) * (R/Rs)**(-15/8.) * f**(11/20.) * u.g/u.cm**3
            Tc = 6.9e7 * self.alpha**(-1/5.) * (self.MBH/const.M_sun)**(-1/5.) * self.mdot_crit**(3/10.) * (R/Rs)**(-3/4.) * f**(3/10.) * u.K
        return rho.cgs, Tc.cgs
    
    def _compute_SSD_H(self, R, Rs):
        if self.method == 'KD18':
            f = 1 - np.sqrt(3*Rs/R)
            H = (1.5e3 * self.alpha**(-1/10.) * ((self.MBH/const.M_sun).cgs.value)**(9/10.) * self.mdot_crit**(3/20.) * (R/Rs)**(9/8.) * f **(3/20.)).cgs * u.cm
            return H
        else:
            return None
    
    def _compute_slimdisk_equation_WSig(self):
        # define the constants and convert physical qunatities to unitless to speed up the computations
        mu_p = 0.617
        gamma = 5/3.
        G = const.G.cgs.value
        c = const.c.cgs.value
        sigma_sb = const.sigma_sb.cgs.value
        alpha = 0.1
        const1 = (const.k_B / (mu_p * const.m_p)).cgs.value
        const2 = (4 * const.sigma_sb / (3 * const.c)).cgs.value

        MBH = (self.MBH/const.M_sun).cgs
        mdot = self.mdot
        Rg = self.Rg.cgs.value
        L_Edd = self.L_edd.cgs.value
        effeciency = self.eta
        Mdot_edd = self.Mdot_edd.cgs.value
        Mdot = self.Mdot.cgs.value
        mdot_crit = Mdot / (L_Edd/c**2)
        MBH = MBH.value

        N = 3
        I_N = 16./35.
        I_Np1 = 128./315.

        a2, a3, a4, a5, a6 = 0.2, 0.3, 0.6, 1.0, 0.875
        b21 = 0.2
        b31, b32 = 0.075, 0.225
        b41, b42, b43 = 0.3, -0.9, 1.2
        b51, b52, b53, b54 = -11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0
        b61, b62, b63, b64, b65 = 1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0

        c1, c2, c3, c4, c5, c6 = 37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0
        dc1 = c1 - 2825.0/27648.0
        dc2 = 0.0
        dc3 = c3 - 18575.0/48384.0
        dc4 = c4 - 13525.0/55296.0
        dc5 = -277.0/14336.0
        dc6 = c6 - 0.25

        @njit
        def _compute_SSD_quant(r, Rg):
            OmgK = np.sqrt(G*MBH/r)/(r-2.)
            rho = 4.7e1 * alpha**(-7/10.) * (MBH)**(-7/10.) * (10*mdot)**(11/20.) * (r/2)**(-15/8.) 
            T = 6.9e7 * alpha**(-1/5.) * (MBH)**(-1/5.) * (10*mdot)**(3/10.) * (r/2)**(-3/4.)
            p_gas = const1 * rho * T
            p_rad = const2 * T**4
            pressure = p_rad + p_gas
            beta = p_gas / pressure
            height = np.sqrt(pressure / rho)/OmgK / (c/np.sqrt(G*MBH))
            Sigma = 2 * height * rho / (Mdot_edd/c/Rg**2)
            Wtp = 2 * height * pressure / (Mdot_edd*c/Rg**2)
            return Sigma, T, Wtp

        def _compute_quantities_N3(r, Wtp, Sigma, ellin, Rg):
            OmgK = 1/np.sqrt(r)/(r-2.)
            ellK = r * r * OmgK 
            height = np.sqrt((2*N+3) * Wtp/Sigma)/OmgK
            rho = Sigma / (2 * height * I_N) * (Mdot_edd/c/Rg**2)
            pressure = Wtp / (2 * height * I_Np1) * (Mdot_edd*c/Rg**2)
            Tc = float(opt.fsolve(lambda T: const1*rho*T + const2*T**4 - pressure, 3e5))
            Wgp = Sigma * const1 * Tc / c**2
            #2 * I_Np1 * height * pressure / (Mdot_edd*c/Rg**2)
            Wrp = max(Wtp - Wgp, 0.)
            beta = min(Wgp / Wtp, 1.0)
            # aleff = alpha * (beta**mu) 
            # this mu is not the average partical mass, mu is for the magnetic pressure mu == 0
            aleff = alpha
            gam1  = beta + (gamma-1.0)*(4.0-3.0*beta)**2.0 / (beta + 12.0*(gamma-1.0)*(1.0-beta))
            gam3  = 1.0 + (gamma-1.0)*(4.0-3.0*beta)       / (beta + 12.0*(gamma-1.0)*(1.0-beta))
            kappa = 0.40 + 0.64e23*(16/35.*rho) / ((2/3.*Tc)**3.5)
            tau = kappa * rho * (height * Rg)
            Qrad  = 8 * const2 * c * Tc**4 / tau * (Rg**2/Mdot_edd/c**2)
            # 8*c*Wrp/(kappa*Sigma) / height * (Rg/Mdot_edd)
            Teff = (Qrad/(Rg/Mdot_edd)*(c**2/Rg) /2/sigma_sb)**(1/4.)
            ell = ellin + 2.0*np.pi*aleff*Wtp*r*r/mdot
            vr = mdot / (2.0*np.pi*r*Sigma)

            # vr correction factor
            b1 = (3.0*gam1 - 1.0) / (2.0*(gam3 - 1.0))
            b2 = aleff*aleff * (9.0*0.*(1.0 - beta)) / (2.0*(4.0 - 3.0*beta))
            b3 = (gam1 + 1.0) / (2.0*(gam3 - 1.0))
            b4 = aleff*aleff * ( 0.*(1.0 + beta)/(2.0*(4.0 - 3.0*beta)) + (1.0 - 0.) )
            if b2 < 1.0e-10:
                X = b3 / (b1 + b4)
            else:
                X = (-(b1 + b4) + np.sqrt((b1 + b4)**2 + 4.0*b2*b3)) / (2.0*b2)

            factor = 1.0 / np.sqrt(X)
            
            return OmgK, ellK, height, Wgp, Wrp, Tc, rho, beta, \
                    gam1, gam3, kappa, Qrad, Teff, tau, ell, vr, \
                    aleff, factor

        def _compute_derivatives_N3(r, Wtp, Sigma, ellin):
            OmgK, ellK, height, Wgp, Wrp, Tc, rho, beta, \
                gam1, gam3, kappa, Qrad, Teff, tau, ell, vr, aleff, factor = _compute_quantities_N3(r, Wtp, Sigma, ellin, Rg)
            dOmgK = -(3*r-2)/(2*r*(r-2))

            a11 = Wtp/Sigma
            a12 = -(mdot/(2.0*np.pi*r*Sigma))**2
            c1 = (ell**2 - ellK**2)/r**3.0 - (Wtp/Sigma)*dOmgK \
                + mdot*mdot/(4.0*np.pi**2*r**3.0 * Sigma**2)

            a21 = ((gam1+1.0)/2.0/(gam3-1.0)) * mdot*Wtp/Sigma \
                - (2.0*np.pi*aleff*r*Wtp)**2.0/mdot
            a22 = - ((3.0*gam1-1.0)/2.0/(gam3-1.0)) * mdot*Wtp/Sigma

            temp1 = -(2.0*np.pi*aleff*Wtp*r*r) * 2.0*ellin/r**3.0 + 2.0*np.pi*r*Qrad
            temp2 = ((gam1-1.0)/(gam3-1.0))*mdot*Wtp/Sigma 
            
            c2 = temp1 + temp2*dOmgK

            den = a11*a22 - a12*a21

            df1 = (a22*c1 - a12*c2)/den * r   # d(ln Wt)/d(ln r)
            df2 = (a11*c2 - a21*c1)/den * r   # d(ln Sig)/d(ln r)
            #print("--- DERIVA ---")
            #print(Wgp, Wrp, beta, gam1, gam3, aleff)
            #print(kappa, rho*Mdot_edd/c/Rg/Rg, Qrad)
            #print(OmgK, dOmgK, ell, ellK)
            #print(a11, a12, a21, a22, c1, c2)
            return [df1, df2]

        def _RK4_N3(x, y, dydx, hstep, ellin):
            # RK-step1
            k1 = hstep * np.array(dydx)

            # RK-step2
            x2 = x + a2 * hstep
            y2 = y + b21 * k1
            dydx2 = _compute_derivatives_N3(np.exp(x2), np.exp(y2[0]), np.exp(y2[1]), ellin)
            k2 = hstep * np.array(dydx2)

            # RK-step3
            x3 = x + a3 * hstep
            y3 = y + b31 * k1 + b32 * k2
            dydx3 = _compute_derivatives_N3(np.exp(x3), np.exp(y3[0]), np.exp(y3[1]), ellin)
            k3 = hstep * np.array(dydx3)

            # RK-step4
            x4 = x + a4 * hstep
            y4 = y + b41 * k1 + b42 * k2 + b43 * k3
            dydx4 = _compute_derivatives_N3(np.exp(x4), np.exp(y4[0]), np.exp(y4[1]), ellin)
            k4 = hstep * np.array(dydx4)

            # RK-step5
            x5 = x + a5 * hstep
            y5 = y + b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4
            dydx5 = _compute_derivatives_N3(np.exp(x5), np.exp(y5[0]), np.exp(y5[1]), ellin)
            k5 = hstep * np.array(dydx5)

            # RK-step6
            x6 = x + a6 * hstep
            y6 = y + b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5
            dydx6 = _compute_derivatives_N3(np.exp(x6), np.exp(y6[0]), np.exp(y6[1]), ellin)
            k6 = hstep * np.array(dydx6)

            y_new = c1 * k1 + c3 * k3 + c4 * k4 + c6 * k6
            y_err = dc1 * k1 + dc3 * k3 + dc4 * k4 + dc5*k5 + dc6 * k6
            return y_new, y_err

        def crossonic(ii, step, r_int_arr, Wtp_int_arr, Sigma_int_arr):
            #ii_right = ii - 1
            #ii_left = ii + 1
            
            ii_right = ii - 2
            ii_left = ii 

            x_right = np.log(r_int_arr[ii_right:ii_left])
            y0_right = np.log(np.array(Wtp_int_arr[ii_right:ii_left]))
            y1_right = np.log(np.array(Sigma_int_arr[ii_right:ii_left]))
            
            r_new = r_int_arr[ii] * np.exp(step)
            x_new = np.log(r_new)

            sonic_interp = self.solv_slim_args.get('sonic_interp', 'linear')
            if sonic_interp == 'polint' and len(x_right) >= 3:
                y0_new = polint(x_right, y0_right, x_new)
                y1_new = polint(x_right, y1_right, x_new)
            else:
                y0_coeff = np.polyfit(x_right, y0_right, 1)
                y0_new = np.polyval(y0_coeff, x_new)
                y1_coeff = np.polyfit(x_right, y1_right, 1)
                y1_new = np.polyval(y1_coeff, x_new)
            
            max_log_jump = self.solv_slim_args.get('sonic_max_log_jump', 2.0)
            y0_new = y0_right[-1] + np.clip(y0_new - y0_right[-1], -max_log_jump, max_log_jump)
            y1_new = y1_right[-1] + np.clip(y1_new - y1_right[-1], -max_log_jump, max_log_jump)
            
            Wtp = np.exp(y0_new)
            Sigma = np.exp(y1_new)
            return r_new, Wtp, Sigma
        
        def polint(xarr, yarr, x):
            n = len(xarr)
            if (len(yarr) != n) or n == 0:
                raise ValueError("length error!")
            
            carr, darr = yarr.copy(), yarr.copy()
            ind_closest_x = np.argmin(abs(xarr-x))
            y = yarr[ind_closest_x]
            ns = ind_closest_x

            for m in range(1, n):
                for i in range(n-m):
                    ho = xarr[i] - x
                    hp = carr[i+m] - x
                    w = carr[i+1] - darr[i]
                    den = ho - hp
                    if den == 0:
                        raise ValueError('failure in polint')
                    den = w/den
                    darr[i] = hp*den
                    carr[i] = ho*den
                if 2*ns < (n-m):
                    dy = carr[ns]
                else:
                    dy = darr[ns-1]
                    ns = ns-1
                y = y + dy
            return y

        r_range = [1e4, 3]
        ellin_range = self.solv_slim_args.get('ellin_range', [3.5, 3.9])
        sonic_step = self.solv_slim_args.get('sonic_step', -5e-2)
        drstep_range = self.solv_slim_args.get('drstep_range', [1e-12, 1e-4])
        shoot_max_iter = self.solv_slim_args.get('shoot_max_iter', 10)
        sonic_ddf_threshold = self.solv_slim_args.get('sonic_ddf_threshold', 1.0e-1)

        Sigma0, T0, Wtp0 = _compute_SSD_quant(r_range[0], Rg)
        quant0 = _compute_quantities_N3(r_range[0], Wtp0, Sigma0, np.mean(ellin_range), Rg)

        if self.verbose:
            print('Initial Sigma0, T0, Wtp0 = ', Sigma0, T0, Wtp0)
            print('ell/ellK=', quant0[14]/quant0[1], 'height/r=', quant0[2])
            print('Tc=', quant0[5], 'rho=', quant0[6], 'beta=', quant0[7], 'kappa=', quant0[10], 'tau=', quant0[13])
        #raise ValueError('Break point')
    
        flag_transonic = False
        n_transonic = 0
        while not flag_transonic:
            # boundary condition
            ellin = np.mean(ellin_range)
            
            # initialization
            r, Wtp, Sigma = r_range[0], Wtp0, Sigma0
            
            if self.verbose:
                print('ellin = ', ellin, f'[{ellin_range[0]}, {ellin_range[1]}]')

            r_int_arr = []
            Sigma_int_arr = []
            Wtp_int_arr = []
            dydx_int_arr = []

            height_int_arr = []
            vr_int_arr = []
            Tc_int_arr = []
            rho_int_arr = []
            Teff_int_arr = []
            OmgK_int_arr = []
            Omg_int_arr = []
            vr2cs_int_arr = []

            hstep = - drstep_range[0]
            ii = 0
            transonic_failed = False
            prev_vr2cs = None
            while r > r_range[1]:
                # r, Wtp, Sigma are from the last step
                if (not np.isfinite(Wtp)) or (not np.isfinite(Sigma)) or Wtp <= 0.0 or Sigma <= 0.0:
                    if flag_transonic:
                        transonic_failed = True
                        if self.verbose:
                            print("   invalid Wtp/Sigma after sonic point", r)
                        break
                    raise ValueError("Wtp or Sigma has nan value. Last step: \n" + \
                        f" r={r_int_arr[-2:]}, Wtp={Wtp_int_arr[-2:]}, Sigma={Sigma_int_arr[-2:]}, "+ \
                        f"dydx = {dydx_int_arr[-2:]}, Tc={Tc_int_arr[-2:]}, rho={rho_int_arr[-2:]}")
                
                x = np.log(r)
                y = np.log(np.array([Wtp, Sigma]))

                dydx = _compute_derivatives_N3(r, Wtp, Sigma, ellin)

                y_new, y_err = _RK4_N3(x, y, dydx, hstep, ellin)

                errmax = np.max(abs(y_err)) / 1e-6
                while errmax > 1.0:
                    htemp = 0.9 * hstep * errmax**(-0.25) / 10.
                    htemp = np.sign(hstep) * max(abs(htemp), 0.1 * abs(hstep))
                    if abs(hstep) < drstep_range[0]:
                        if flag_transonic:
                            transonic_failed = True
                            if self.verbose:
                                print("   hstep underflow after sonic point", r)
                            break
                        print("   hstep underflow", r)
                        hstep = - drstep_range[0]
                        raise RuntimeError("hstep underflow")
                    hstep = htemp
                    y_new, y_err = _RK4_N3(x, y, dydx, hstep, ellin)
                    errmax = np.max(abs(y_err)) / 1e-6
                
                if errmax > 1.89e-4:
                    htemp = 0.9 * hstep * errmax**(-0.20)
                else:
                    htemp = 5.0 * hstep
                if abs(htemp) > drstep_range[1]:
                    htemp = - drstep_range[1]

                #print(ii, r, Sigma, Wtp, hstep, y_new/hstep)
                
                r_int_arr.append(r)
                Wtp_int_arr.append(Wtp)
                Sigma_int_arr.append(Sigma)
                dydx_int_arr.append(y_new/hstep)

                OmgK, ellK, height, Wgp, Wrp, Tc, rho, beta, \
                gam1, gam3, kappa, Qrad, Teff, tau, ell, vr, aleff, factor = _compute_quantities_N3(r, Wtp, Sigma, ellin, Rg)

                vr = mdot / (2.0 * np.pi * r * Sigma)
                cs = np.sqrt(Wtp / Sigma)
                vr2cs = vr / cs / factor
                vr2cs_int_arr.append(vr2cs)

                height_int_arr.append(height)
                vr_int_arr.append(vr)
                Tc_int_arr.append(Tc)
                rho_int_arr.append(rho)
                Teff_int_arr.append(Teff)
                OmgK_int_arr.append(OmgK)
                Omg_int_arr.append(ell/r/r)

                if (0.97 < vr2cs < 1.03) or (prev_vr2cs is not None and prev_vr2cs < 1.0 <= vr2cs):
                    ind_transonic = ii
                    flag_transonic = True
                    r, Wtp, Sigma = crossonic(ii, sonic_step, r_int_arr, Wtp_int_arr, Sigma_int_arr)
                    #r = r * np.exp(-5e-2)
                    if self.verbose: print('Transonic!', r, vr2cs)
                    #htemp = - drstep_range[0] # add
                else:
                    r = r * np.exp(hstep)
                    Wtp, Sigma = Wtp*np.exp(y_new[0]), Sigma*np.exp(y_new[1])
                
                prev_vr2cs = vr2cs
                hstep = htemp
                ii += 1
                if r * np.exp(hstep) < r_range[1]:
                    hstep = np.log(r_range[1]/r)

            ###flag_transonic = True
            if flag_transonic:
                n_transonic += 1
                self.r_transonic = r_int_arr[ind_transonic]
                ddf0 = abs((dydx_int_arr[ind_transonic+1][0] - dydx_int_arr[ind_transonic][0]) / (r_int_arr[ind_transonic+1] - r_int_arr[ind_transonic]))
                ddf1 = abs((dydx_int_arr[ind_transonic+1][1] - dydx_int_arr[ind_transonic][1]) / (r_int_arr[ind_transonic+1] - r_int_arr[ind_transonic]))
                ddf = np.nanmin([ddf0, ddf1])
                if self.verbose: print(ddf)
                if True: 
                    os.makedirs('./tmp', exist_ok=True)
                    np.savetxt(f'./tmp/{n_transonic}.npy', np.stack([r_int_arr, Teff_int_arr]))
                if transonic_failed:
                    if (ellin_range[1] - ellin_range[0]) > 1.0e-8:
                        ellin_range[1] = ellin
                        flag_transonic = False
                        if self.verbose:
                            print(f"Rejecting ellin={ellin}; post-sonic integration failed. New range={ellin_range}")
                    else:
                        raise RuntimeError(f"hstep underflow after sonic point and ellin_range is exhausted: {ellin_range}")
                elif n_transonic < shoot_max_iter and ind_transonic + 1 < len(dydx_int_arr):
                    if (ddf > sonic_ddf_threshold) and (ellin_range[1] - ellin_range[0] > 1.0e-6):
                        ellin_range[1] = ellin
                        flag_transonic = False
            else:
                if self.verbose: print("No transonic")
                if (ellin_range[1] - ellin_range[0]) > 1.0e-6:
                    ellin_range[0] = ellin
                else:
                    if self.verbose: print("Bad Initial ellin, no solution found.")
                    raise RuntimeError(f"No sonic crossing found before r={r_range[1]} after exhausting ellin_range={ellin_range}")
        
        
        self.slim_lin = ellin
        quantities = np.zeros((len(r_int_arr), 12))
        # r, Sigma, W, dydx[0], dydx[1], height, vr, Tc, rho, Teff, OmgK, Omg
        self.r_in_slim =  min(r_int_arr)

        for i in range(len(r_int_arr)):
            ii = - i - 1
            r, Sigma, W, dydx0, dydx1, height, vr, Tc, rho, Teff, OmgK, Omg = r_int_arr[ii], Sigma_int_arr[ii],  Wtp_int_arr[ii], \
                                                                                dydx_int_arr[ii][0], dydx_int_arr[ii][1], height_int_arr[ii],\
                                                                                vr_int_arr[ii], Tc_int_arr[ii], rho_int_arr[ii],\
                                                                                Teff_int_arr[ii], OmgK_int_arr[ii], Omg_int_arr[ii]
            quantities[i, :] = r, Sigma, W, dydx0, dydx1, height, vr, Tc, rho, Teff, OmgK, Omg
        
        mask_sonic = np.isclose(quantities[:, 0], self.r_transonic, rtol=0, atol=1e-8)

        quantities = pd.DataFrame(quantities[~mask_sonic, :], columns=['r', 'Sigma', 'W', 'dydx0', 'dydx1', 'height', 
                                                       'vr', 'Tc', 'rho', 'Teff', 'OmgK', 'Omg'])
        

        if self.verbose:
            print(f'Slim disk solution computed from r={r_range[0]} to r={r_range[1]} (Rg) successfully.')
            print(f'Slim disk quantities: r, Sigma, W, dydx0, dydx1, height, vr, Tc, rho, Teff, OmgK, Omg')
        return quantities
    
    def _compute_slimdisk_temperature4(self, radius):
        if self.method != 'Slim':
            raise ValueError("The method is not set to 'Slim'.")
        #if radius <= self.r_in_slim:
        #    return self._compute_NT_temperature4(radius)
        else:
            r_grid = self.slim_quantities['r'].values
            Teff_grid = (self.slim_quantities['Teff'].values)**4
            #Q_rad = self.slim_quantities['Q_rad'].values
            Teff = interpolate.UnivariateSpline(r_grid, Teff_grid, k=1, s=0)
            return Teff(radius)

    def _compute_disk_temperature4(self, radius, if_for_corona=False):
        if radius < self.r_in:
            Temp4 = 0
        elif self.method == 'Slim':
            Temp4 = self._compute_slimdisk_temperature4(radius)
        else:
            Temp4 = self._compute_NT_temperature4(radius)
            if self.method == 'KD19': #self.if_slimdisk:
                Temp4_edd = (self.L_edd/(4 * np.pi * (radius * self.Rg)**2)/const.sigma_sb).cgs.value
                if 2.39 <= self.mdot <= 6:
                    if (radius < self.r_super_edd[0]):
                        Temp4 = min(Temp4, Temp4_edd)
                        mod = (self.mdot/2.39)**(np.log10(Temp4_edd/Temp4)/np.log10(6/2.39))
                        Temp4 = Temp4 * mod
                    elif self.r_super_edd[0] <= radius < self.r_super_edd[1]:
                        Temp4 = Temp4_edd
                elif self.mdot >= 6:
                    if self.r_crit != None:
                        if radius < self.r_crit:
                            Temp4 = Temp4_edd
                    else:
                        Temp4 = Temp4_edd

        if if_for_corona:
            return Temp4
        
        if self.if_reprocessing:
            Temp4 = Temp4 + (self._compute_reprocessed_flux_r(radius)/const.sigma_sb).cgs.value
                
        return Temp4
    
    def _compute_disk_fcol(self, Temp):
        """
        -- taken from relagn --
        ## need to check this function
        Calculates colour temperature correction following Eqn. (1) and 
        Eqn. (2) in Done et al. (2012)

        Parameters
        ----------
        Tm : float
            Max temperature at annulus (ie T(r)) - units : K.

        Returns
        -------
        fcol_d : float
            colour temperature correction at temperature T

        """
        if Temp > 1e5:
            #For this region follows Eqn. (1)
            Tm_keV = (const.k_B * Temp * u.K).to(u.keV).value #convert to units consitant with equation
            
            fcol_d = (72/Tm_keV)**(1/9)
        
        elif 3e4 < Temp < 1e5:
            #For this region follows Eqn. (2)
            fcol_d = (Temp / (3e4))**(0.82)
        
        else:
            fcol_d = 1
        
        return fcol_d

    def _disk_blackbody_emission_r(self, radius):
        Temp = self._compute_disk_temperature4(radius) ** (1/4.) # no unit, unit: K
        
        if self.fcol_corr <= 0:
            fcol_r = self._compute_disk_fcol(Temp)
        else:
            fcol_r = self.fcol_corr
        Temp = Temp * fcol_r

        bb_func = BlackBody(temperature= Temp * u.K)

        Blackbody_emis = bb_func(self.nu_arr).to(u.erg / (u.cm**2 * u.s * u.Hz * u.sr)) * u.sr

        return Blackbody_emis
    
    def _disk_blackbody_emission_total(self, logr_in, logr_out):
        """
        Integrating the blackbody emission from the disk over the radius range r_in to r_out.
        logr_in and logr_out are in units of Rg
        2 pi * int (2 pi r blackbody(r) dr) 
        output unit: erg/(s Hz)
        """

        total_disk_flux = np.zeros(len(self.nu_arr))
        r_grid = np.geomspace(logr_in, logr_out, num=1000)
        self.disk_logr_grid = r_grid.copy()

        for i in range(len(r_grid) - 1):
            rbin_in = 10**r_grid[i]
            rbin_out = 10**r_grid[i+1]
            drbin = rbin_out - rbin_in
            rbin_cent = (rbin_out + rbin_in) / 2.
            
            flux_at_r = self._disk_blackbody_emission_r(rbin_cent) # unit: erg/(cm^2 s Hz)
            flux_annular_r = 2 * np.pi * (2 * np.pi * self.Rg**2 * rbin_cent * drbin * flux_at_r).to(u.erg / u.s / u.Hz).value # unit: erg/s/Hz
            flux_annular_r[~np.isfinite(flux_annular_r)] = 0 # set inf or nan to zero
            total_disk_flux += flux_annular_r
        
        total_disk_flux = total_disk_flux * u.erg / u.s / u.Hz
        
        return total_disk_flux

    def _disk_blackbody_luminosity(self, r_in, r_out):
        """
        Integrates the disk luminosity over the radius range r_in to r_out.
        At each r: (4 pi r^2) x (sigma_sb T^4)
        r_in and r_out are in units of Rg
        """
        constant = 4 * np.pi * self.Rg ** 2 * const.sigma_sb 
        disk_luminosity = (constant * integrate.quad(lambda r: r * self._compute_disk_temperature4(r), r_in, r_out)[0] * u.K**4).to(u.erg / u.s)
        return disk_luminosity

    def _disk_NT_blackbody_luminosity(self, r_in, r_out):
        """
        this one DOES NOT include reprocessing
        """
        constant = 4 * np.pi * self.Rg ** 2 * const.sigma_sb 
        disk_luminosity = (constant * integrate.quad(lambda r: r * self._compute_NT_temperature4(r), r_in, r_out)[0] * u.K**4).to(u.erg / u.s)
        return disk_luminosity
    
    def get_disk_component_intrinsic(self, logr_in=None, logr_out=None):
        """
        Returns the disk component of the spectrum
        """
        if logr_in is None:
            logr_in = np.log10(self.r_warm)
        if logr_out is None:
            logr_out = self.logr_out
        disk_spectrum = self._disk_blackbody_emission_total(logr_in, logr_out)
        disk_luminosity = self._disk_blackbody_luminosity(10**logr_in, 10**logr_out)
        op_disk_component = {'L_disk': disk_luminosity, 'Fnu_disk': disk_spectrum}
        
        self.SED_disk_component = op_disk_component
        return op_disk_component

    """
    Compute the warm compton component
    """

    def _warm_compton_emission_r(self, radius):
        wc_flux_at_r = np.zeros(len(self.nu_arr))
        if radius > self.r_warm:
            # return zero if radius is larger than r_warm
            return wc_flux_at_r
        else:
            Temp = self._compute_disk_temperature4(radius) ** (1/4.) # no unit, unit: K
            Temp_keV = (const.k_B * Temp * u.K).to(u.keV).value # convert to units KeV

            photon_flux_r = self._compute_compton_photon_flux(self.Ebins, [self.Gamma_warm, self.kT_warm, Temp_keV, 0, 0]) 

            return photon_flux_r # unit: erg/s/cm^2/Hz
    
    def _warm_compton_emission_total(self, logr_in, logr_out):
        total_warm_flux = np.zeros(len(self.nu_arr))
        r_grid = np.linspace(logr_in, logr_out, num=100)
        self.warm_logr_grid = r_grid.copy()

        for i in range(len(r_grid) - 1):
            rbin_in = 10**r_grid[i]
            rbin_out = 10**r_grid[i+1]
            drbin = rbin_out - rbin_in
            rbin_cent = (rbin_out + rbin_in) / 2.
            
            flux_at_r = self._warm_compton_emission_r(rbin_cent) # unit: erg/s/cm^2/Hz
            flux_at_r[~np.isfinite(flux_at_r)] = 0 # set inf or nan to zero

            # scale the compton flux to the disk flux at r
            flux_at_r_int = integrate.trapezoid(flux_at_r, self.nu_arr) # integrate
            disk_flux_at_r = const.sigma_sb * self._compute_disk_temperature4(rbin_cent) * u.K**4 # erg/s/cm^2
            ratio = (disk_flux_at_r / flux_at_r_int).cgs # ratio of disk flux to compton flux at rbin_cent
            wc_flux_at_r = flux_at_r * ratio # scale the compton flux to the disk flux

            flux_annular_r = 2 * (2 * np.pi * self.Rg**2 * rbin_cent * drbin * wc_flux_at_r).to(u.erg / u.s / u.Hz).value
            total_warm_flux += flux_annular_r
        total_warm_flux = total_warm_flux * u.erg / u.s / u.Hz

        return total_warm_flux

    def _warm_compton_luminosity(self, r_in, r_out):
        """
        Integrates the warm compton luminosity over the radius range r_in to r_out.
        At each r: (4 pi r^2) x (sigma_sb T^4)
        r_in and r_out are in units of Rg
        """
        constant = 4 * np.pi * self.Rg ** 2 * const.sigma_sb 
        warm_compton_luminosity = (constant * integrate.quad(lambda r: r * self._compute_disk_temperature4(r), r_in, r_out)[0] * u.K**4).to(u.erg / u.s)
        return warm_compton_luminosity
    
    def get_warm_component_intrinsic(self, logr_in=None, logr_out=None):
        """
        Returns the warm compton component of the spectrum
        """
        if logr_in is None:
            logr_in = np.log10(self.r_hot)
        if logr_out is None:
            logr_out = np.log10(self.r_warm)
        if logr_out <= logr_in:
            op_warm_compton_component = {'L_warm': 0*u.erg/u.s, 'Fnu_warm': np.zeros(len(self.nu_arr)) * u.erg / u.s / u.Hz}
            self.SED_warm_component = op_warm_compton_component
            if self.verbose: print('No warm corona region, return zero spectrum and luminosity')
            return op_warm_compton_component
        
        warm_compton_spectrum = self._warm_compton_emission_total(logr_in, logr_out)
        warm_compton_luminosity = self._warm_compton_luminosity(10**logr_in, 10**logr_out)
        op_warm_compton_component = {'L_warm': warm_compton_luminosity, 'Fnu_warm': warm_compton_spectrum}
        self.SED_warm_component = op_warm_compton_component
        return op_warm_compton_component
    
    """
    Compute the hot compton component
    """

    def _hot_compton_luminosity(self):
        return self.L_corona_dissipated + self.L_corona_seed

    def _hot_compton_emission_r(self, radius):
        Temp = self._compute_disk_temperature4(radius, True) ** (1/4.) # no unit, unit: K
        Temp_keV = (const.k_B * Temp * u.K).to(u.keV).value # convert to units KeV

        y_warm =  (4.0 / 9.0 * self.Gamma_warm) ** (-4.5)
        Temp_comp = Temp_keV * np.exp(y_warm) 
        #print(radius, Temp_keV, Temp_comp)
        photon_flux_r = self._compute_compton_photon_flux(self.Ebins, [self.Gamma_hot, self.kT_hot, Temp_comp, 0, 0]) 

        return photon_flux_r # unit: erg/s/cm^2/Hz

    def _hot_compton_emission_total(self, logr_in, logr_out):
        total_hot_flux = np.zeros(len(self.nu_arr))
        r_grid = np.linspace(logr_in, logr_out, num=100)
        self.hot_logr_grid = r_grid.copy()

        for i in range(len(r_grid) - 1):
            rbin_in = 10**r_grid[i]
            rbin_out = 10**r_grid[i+1]
            drbin = rbin_out - rbin_in
            rbin_cent = (rbin_out + rbin_in) / 2.
            
            flux_at_r = self._hot_compton_emission_r(rbin_cent)
            flux_at_r[~np.isfinite(flux_at_r)] = 0
            
            hc_flux_at_r = flux_at_r

            flux_annular_r = 2 * (2 * np.pi * self.Rg**2 * rbin_cent * drbin * hc_flux_at_r).to(u.erg / u.s / u.Hz).value
            total_hot_flux += flux_annular_r
        total_hot_flux = total_hot_flux * u.erg / u.s / u.Hz

        return total_hot_flux

    def get_hot_component_intrinsic(self, logr_in=None, logr_out=None):
        if logr_in is None:
            logr_in = np.log10(self.r_in)
        if logr_out is None:
            logr_out = np.log10(self.r_hot)

        if logr_out <= logr_in:
            op_hot_compton_component = {'L_hot': 0*u.erg/u.s, 'Fnu_hot': np.zeros(len(self.nu_arr)) * u.erg / u.s / u.Hz}
            self.SED_hot_component = op_hot_compton_component
            if self.verbose: print("No hot corona region, return zero spectrum and luminosity")
            return op_hot_compton_component
               
        hot_compton_spectrum_tmp = self._hot_compton_emission_total(logr_in, logr_out)
        hot_compton_spectrum_integrated = integrate.trapezoid(hot_compton_spectrum_tmp.value, self.nu_arr.value) 
        hot_compton_luminosity = self._hot_compton_luminosity() # hot corona is spherical (lampost model) in SSD
        ratio = hot_compton_luminosity.value / hot_compton_spectrum_integrated
        hot_compton_spectrum = hot_compton_spectrum_tmp * ratio
        
        op_hot_compton_component = {'L_hot': hot_compton_luminosity, 'Fnu_hot': hot_compton_spectrum}
        self.SED_hot_component = op_hot_compton_component
        return op_hot_compton_component

    """
    Compute the total SED
    """
    def get_total_SED(self):
        Fnu_total = self.get_disk_component_intrinsic()['Fnu_disk'] + \
                    self.get_warm_component_intrinsic()['Fnu_warm'] + \
                    self.get_hot_component_intrinsic()['Fnu_hot']
        self.SED_total = Fnu_total
        return Fnu_total
    
    def get_total_SED_obs(self):
        Fnu_total = (self.get_disk_component_intrinsic()['Fnu_disk'] + \
                        self.get_warm_component_intrinsic()['Fnu_warm'] + \
                        self.get_hot_component_intrinsic()['Fnu_hot']) * self.cosi
        return Fnu_total
