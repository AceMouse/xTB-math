from math import sqrt, exp
from gfn2 import kExpLight, kExpHeavy, repAlpha, repZeff, nShell, chemicalHardness, shellHardness, thirdOrderAtom, kshell

H = 0
He = 1
C = 5


element_ids = [C,C,C]
positions = [(1,0,0),(0,1,0),(0,0,1)]
atoms = zip(element_ids, positions)
 
def dist(v1, v2): #euclidean distance. 
    d = 0
    for a,b in zip(v1,v2):
        d += (a-b)**2
    d = sqrt(d)
    return d


def repulsion_energy(atoms):
    acc = 0
    for A,v1 in atoms:
        for B,v2 in atoms:
            kf = kExpLight if A in {H,He} and B in {H,He} else kExpHeavy
            R_AB = dist(v1,v2)
            frac = (repZeff[A] * repZeff[B])/R_AB
            acc += frac*exp(-sqrt(repAlpha[A],repAlpha[B])*(R_AB**kf)) 

    acc *= 0.5
    return acc

def isotropic_electrostatic_and_XC_energy_second_order(atoms, charges):
    acc = 0
    for A,v1 in atoms:
        etaA = chemicalHardness[A]
        for B,v2 in atoms:
            etaB = chemicalHardness[B]
            R_AB2 = dist(v1,v2)**2
            for l in range(nShell[A]):
                klA = shellHardness[A][l]
                for lp in range(nShell[B]):
                    klpB = shellHardness[B][lp]
                    eta_ABllp = 0.5*(etaA*(1+klA)+etaB*(1+klpB))
                    gamma_ABllp = 1./sqrt(R_AB2+eta_ABllp**(-2))
                    acc += charges[A][l]*charges[B][lp]*gamma_ABllp
    acc *= 0.5
    return acc

def isotropic_electrostatic_and_XC_energy_third_order(atoms, charges):
    acc = 0
    for A,_ in atoms:
        for l in range(nShell[A]):
            acc += (charges[A][l]**3)*kshell[l]*thirdOrderAtom[A]
    acc *= 1./3
    return acc

            
