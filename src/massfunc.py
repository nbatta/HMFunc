import numpy as np
from scipy.interpolate import interp2d
import camb
from camb import model
from src.cosmology import Cosmology
import src.cosmology as cosmo
from src.tinker import dn_dlogM

np.seterr(divide='ignore', invalid='ignore')

class HMF(Cosmology):
    def __init__(self, paramDict=cosmo.defaultCosmology, constDict=cosmo.defaultConstants, version='camb', zarr=None):
        Cosmology.__init__(self, paramDict, constDict)
        if zarr is None:
            self.zarr_edges = np.arange(0.000001, 2.01, 0.1)
        else:
            self.zarr_edges = zarr

        self.zarr = (self.zarr_edges[1:] + self.zarr_edges[:-1])/2.

        print (self.zarr)

        self.h = self.H0/100.

        self.M = 10**np.arange(13.5, 15.7, 0.02) 

        self.rhoc0om = self.rho_crit0H100*self.om

        self.kh, self.pk = self._pk(self.zarr)

        if (version == 'camb'):            
            self.version = 'camb'
        if (version == 'col'):
            from colossus.cosmology import cosmology
            internal_cosmo = {'flat': True,
                              'H0': self.H0,
                              'Om0': self.om,
                              'Ob0': self.ob,
                              'sigma8': self.s8,
                              'ns': self.ns}

            cosmo = cosmology.setCosmology('internal_cosmo', internal_cosmo)
            self.version = 'col'

        self.DAz = self.results.angular_diameter_distance(self.zarr)
        self.DMz = self.results.comoving_radial_distance(self.zarr_edges)
        self._initdVdz(self.zarr)
        self._initdVdz2(self.zarr)
        
    def E_z(self, z):
        # hubble function
        ans = self.results.hubble_parameter(z)/self.H0  # 0.1% different from sqrt(om*(1+z)^3+ol)
        return ans

    def rhoc(self, z):
        # critical density as a function of z
        ans = self.rho_crit0H100*self.E_z(z)**2.
        return ans

    def rhom(self, z):
        # mean matter density as a function of z
        ans = self.rhoc0om*(1.0 + z)**3
        return ans

    def _pk(self, zarr, kmin=1e-5, kmax=50, knum=400):
        self.pars.set_matter_power(redshifts=np.append(self.zarr, 0),  kmax=kmax, silent=True)
        self.pars.Transfer.high_precision = False  # True
        self.pars.NonLinear = model.NonLinear_none
        results = camb.get_results(self.pars)
        self.s8 = results.get_sigma8()[-1]

        kh, z, powerZK = results.get_matter_power_spectrum(minkh=kmin, maxkh=kmax, npoints=knum, var1='delta_nonu', var2='delta_nonu')
        return kh, powerZK[1:, :]  # remove z = 0 from output

    def _initdVdz(self,z_arr):
        #dV/dzdOmega 
        DA_z = self.DAz
        dV_dz = DA_z**2 * (1.+z_arr)**2
        #for i in range (z_arr.size):
        dV_dz /= (self.results.h_of_z(z_arr))
        #dV_dz *= (self.H0/100.)**3. # was h0
        self.dVdz = dV_dz

    def _initdVdz2(self,z_arr):
        #dV/dzdOmega 
        DM_z = self.DMz
        dV_dz = DM_z**3 * 4./3. *np.pi 
        #for i in range (z_arr.size):
        #dV_dz *= (self.H0/100.)**3. # was h0
        self.dVdz2 = dV_dz[1:] - dV_dz[:-1]

    def critdensThreshold(self, z, deltac):
        rho_treshold = deltac * self.rhoc(z)/self.rhom(z)
        return rho_treshold

    def dn_dM(self, M, delta):
        '''
        dN/dmdV Mass Function
        M here is in MDeltam but we can convert
        '''
        if (self.version == 'camb'):
            # delts = self.zarr*0. + delta
            delts = self.critdensThreshold(self.zarr, delta)
            dn_dlnm = dn_dlogM(M, self.zarr, self.rhoc0om, delts, self.kh, self.pk, 'comoving')
            dn_dm = dn_dlnm/M[:, None]
        elif (self.version == 'col'):
            # Note Collosus uses the MPS computed using the Eisenstein & Hu 1998 transfer function,
            # with this approximation, the variance (sigma) is accurate to about 2% or better
            dn_dlnm = np.array([])
            from colossus.lss import mass_function
            from colossus.cosmology import cosmology
            from colossus.halo import mass_so
            internal_cosmo = {'flat': True,
                              'H0': self.H0,
                              'Om0': self.om,
                              'Ob0': self.ob,
                              'sigma8': self.s8,
                              'ns': self.ns}
            cosmo = cosmology.setCosmology('internal_cosmo', internal_cosmo)

            for i in range(len(self.zarr)):
                mfunc = mass_function.massFunction(M/self.h, self.zarr[i],
                                                   mdef=str(delta)+'c', model='tinker08', q_out='dndlnM')
                dn_dlnm = np.append(dn_dlnm, mfunc, axis=0)

            mlen = len(dn_dlnm)
            zlen = len(self.zarr)
            dn_dlnm = np.transpose(np.reshape(dn_dlnm, (zlen, np.int(mlen/zlen))))
            dn_dm = dn_dlnm/(M[:, None]/(self.h))

        return dn_dm * (self.h)**4

    def N_of_Mz(self,M,delta):
        #dN/dzdOmega
        dn_dm = self.dn_dM(M,delta)
        dV_dz = self.dVdz
        N_dzdm = dn_dm * dV_dz[None,:]

        return N_dzdm

    def N_of_z(self,M,delta):
        # dN/dz(z) = 4pi fsky \int dm dN/dzdmdOmega
        z_arr = self.zarr
        dn_dm = self.dn_dM(M,delta)
        #dn_dzdm = self.N_of_Mz(self.M,delta)
        N_z = np.zeros(z_arr.size)
        #for i in range(z_arr.size):
        #N_z = np.trapz(dn_dzdm[:,i],np.diff(self.M))
        #N_z = np.trapz(dn_dzdm, dx=np.diff(self.M[:,None]), axis=0)
        N_z = np.trapz(dn_dm*np.ones([len(M),len(self.zarr)]),x=M/self.h,axis=0)
        #N_z = np.sum(dn_dzdm*np.gradient(self.M)[:,None],axis=0)
        #N_z = np.trapz(dn_dzdm,dx=np.diff(self.M200),axis=0)
        return N_z#*self.dVdz2

    def inter_dndmLogm(self, delta, M=None):
        """
        interpolating over M500c becasue that's a constant vector at every redshift, log10 M500c
        """
        if M is None:
            M = self.M
        dndM = self.dn_dM(M, delta)
        ans = interp2d(self.zarr, np.log10(M), np.log10(dndM), kind='cubic', fill_value=0)
        # ans = RectBivariateSpline(self.zarr, np.log10(M), np.log10(dndM).T)
        return ans
