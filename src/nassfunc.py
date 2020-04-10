import numpy as np
from src.cosmology import Cosmology
import src.cosmology as cosmo
from colossus.cosmology import cosmology
from colossus.lss import mass_function

class NMF(Cosmology):
    def __init__(self, areaDeg2=600., zMin= 0., zMax=2., zStep=0.1, paramDict=cosmo.defaultCosmology, constDict=cosmo.defaultConstants):
        Cosmology.__init__(self, paramDict, constDict)

        zRange=np.arange(zMin, zMax+zStep, zStep)
        areaSr=np.radians(np.sqrt(areaDeg2))**2
        self.areaSr=areaSr
        self.areaDeg2=areaDeg2
        self.zBinEdges=zRange
        self.z=(zRange[:-1]+zRange[1:])/2.

        internal_cosmo = {'flat': True,
                          'H0': 70,
                          'Om0': 0.3,
                          'Ob0': self.ob,
                          'sigma8': 0.8, #self.s8,
                          'ns': self.ns}

        print (internal_cosmo,self.s8)
        self.cosmoModel=cosmology.setCosmology('nemo', internal_cosmo)
        self.h = self.cosmoModel.h


        self.log10M=np.arange(np.log10(5e13), 16, 0.01)
        self.M=np.power(10, self.log10M)*self.h
        self.mdef='500c'
        self.model='tinker08'

    def _comovingVolume(self, z):
        """Returns co-moving volume in Mpc^3 (all sky) to some redshift z, using Colossus routines (taking
        care of the fact that Colossus returns all distances in Mpc/h).
        
        NOTE: Assumes flat cosmology
        
        """
        return (4/3)*np.pi*np.power(self.cosmoModel.comovingDistance(0, z)/self.h, 3)

    def _doClusterCount(self):
        """Updates cluster count etc. after mass function object is updated.
        
        """

        zRange=self.zBinEdges
        self.M=np.power(10, self.log10M)*self.h
        
        # Number density by z and total cluster count (in redshift shells)
        # Can use to make P(m, z) plane
        numberDensity=[]
        clusterCount=[]
        totalVolumeMpc3=0.
        for i in range(len(zRange)-1):
            zShellMin=zRange[i]
            zShellMax=zRange[i+1]
            zShellMid=(zShellMax+zShellMin)/2.  
            dndlnM=mass_function.massFunction(self.M/self.h, zShellMid, mdef = self.mdef, 
                                              model = self.model, q_out = 'dndlnM')
            dndM=dndlnM/self.M
            # NOTE: this differs from hmf by several % at the high-mass end (binning or interpolation?)
            n=(dndM*self.h**4)*np.gradient(self.M/self.h)
            numberDensity.append(n)
            shellVolumeMpc3=self._comovingVolume(zShellMax)-self._comovingVolume(zShellMin)
            shellVolumeMpc3=shellVolumeMpc3*(self.areaSr/(4*np.pi))
            totalVolumeMpc3=totalVolumeMpc3+shellVolumeMpc3
            clusterCount.append(n*shellVolumeMpc3)
        numberDensity=np.array(numberDensity)
        clusterCount=np.array(clusterCount)  
        self.volumeMpc3=totalVolumeMpc3
        self.numberDensity=numberDensity
        self.clusterCount=clusterCount
        self.numClusters=np.sum(clusterCount)
        self.numClustersByRedshift=np.sum(clusterCount, axis = 1)
