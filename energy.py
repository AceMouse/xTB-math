from math import sqrt, exp
from enum import Enum
from gfn2 import kExpLight, kExpHeavy, repAlpha, repZeff, nShell, chemicalHardness, shellHardness, thirdOrderAtom, kshell, selfEnergy, kCN, shellPoly

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
            acc += frac*exp(-sqrt(repAlpha[A]*repAlpha[B])*(R_AB**kf)) 

    acc *= 0.5
    return acc

def isotropic_electrostatic_and_XC_energy_second_order(atoms, charges):
    acc = 0
    for A,v1 in atoms:
        etaA = chemicalHardness[A]
        for B,v2 in atoms:
            etaB = chemicalHardness[B]
            R_AB2 = dist(v1,v2)**2
            for u in range(nShell[A]):
                kuA = shellHardness[A][u]
                for v in range(nShell[B]):
                    kvB = shellHardness[B][v]
                    eta_ABuv = 0.5*(etaA*(1+kuA)+etaB*(1+kvB))
                    gamma_ABuv = 1./sqrt(R_AB2+eta_ABuv**(-2))
                    acc += charges[A][u]*charges[B][v]*gamma_ABuv
    acc *= 0.5
    return acc

def isotropic_electrostatic_and_XC_energy_third_order(atoms, charges):
    acc = 0
    for A,_ in atoms:
        for u in range(nShell[A]):
            acc += (charges[A][u]**3)*kshell[u]*thirdOrderAtom[A]
    acc *= 1./3
    return acc

Kll_AB = [
    [1.85,2.04,2.00],
    [2.04,2.23,2.00],
    [2.00,2.00,2.23]
]
class Energy(Enum):
    REPULSION = 1.5


# energy_type: The type of energy to compute for
# s_uv: overlap of the orbitals. how do we get this?? (To get the coefficients to compute the slater orbites I think we need to compute the zeroth iteration for the wavefunction with a start guess?)
def extended_huckel_energy(atoms, s_uv, energy_type: Energy):
    acc = 0
    for A,v1 in atoms:
        for B,v2 in atoms:
            for u in range(nShell[A]):
                for v in range(nShell[B]):
                    # TODO: We need to compute the density matrix P
                    P_uv = 0
                    acc += P_uv * H_EHT(A, B, u, v, v1, v2, atoms, energy_type)
    return acc


def H_EHT(A, B, u, v, v1, v2, atoms, s_uv, energy_type: Energy):
    Kuv_AB = Kll_AB[u][v] 

    hl_A = selfEnergy[A]
    delta_hl_CNA = kCN[A][GFN2_coordinate_number(A,v1, atoms)]
    H_uu = hl_A - delta_hl_CNA * GFN2_coordinate_number(A,v1, atoms)
    H_vv = hl_A - delta_hl_CNA * GFN2_coordinate_number(B,v2, atoms)

    X_electronegativity = 1 if A[0] == B[0] else 1 + 0.02 * (0.35**2)
    R_AB = dist(v1,v2)**2
    # TODO: What is Rcov_AB?
    k_polyA = shellPoly[A][u]
    k_polyB = shellPoly[B][v]
    Rcov_AB = 0
    II = (1 + k_polyA (R_AB / Rcov_AB)**0.5)(1 + k_polyB (R_AB / Rcov_AB)**0.5)

    return 0.5 * Kuv_AB * s_uv * (H_uu + H_vv) * X_electronegativity * II


# CN'_A
# atom: The atom to compute for
# atoms: All atoms
def GFN2_coordinate_number(A, v1, atoms):
    # TODO: Find these covalent radii values
    R_Acov = 0
    R_Bcov = 0

    for B,v2 in atoms:
        if A != B:
            R_AB = dist(v1,v2)**2
            (1 + exp(-10 * (4 * (R_Acov + R_Bcov)/3 * R_AB - 1)))**-1 * (1 + exp(-20 * (4 * (R_Acov + R_Bcov + 2)/3 * R_AB - 1)))**-1

