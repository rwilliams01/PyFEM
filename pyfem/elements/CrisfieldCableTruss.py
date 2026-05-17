from .CrisfieldTruss import CrisfieldTruss

class CrisfieldCableTruss(CrisfieldTruss):

    def getStrain(self, elemdat):
        E11 = super().getStrain(elemdat)
        # only return tension values
        return max(E11, 0.0)