from math import sqrt, exp, e
from enum import Enum
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


class Energy(Enum):
    REPULSION = 1.5

# energy_type: The type of energy to compute for
# s_uv: overlap of the orbitals. how do we get this?? (To get the coefficients to compute the slater orbites I think we need to compute the zeroth iteration for the wavefunction with a start guess?)
def extended_huckel_energy(atoms, s_uv, energy_type: Energy):
    acc = 0
    
    # TODO: We need to compute the density matrix P
    P_uv = 0

    for A in atoms:
        for B in atoms:
            acc += P_uv * H_EHT(A, B, atoms, s_uv, energy_type)
    return acc


def H_EHT(A, B, atoms, s_uv, energy_type: Energy):
    # kll_AB: effective scaling factor common to all EHT methods. Find values in Table 2.
    # TODO: How do we determine which of the fitted constants from Table 2 to use?
    # I am guessing for repulsion energy we use K_rep and so on.
    Kll_AB = energy_type.value # K^{ll'}_AB

    # TODO: Find these fitted constants
    hl_A = 0
    # TODO: How do we get the delta? seems like it should depend on the result of CN'_A somehow?
    delta_hl_CNA = 0
    # TODO: Is it correct that these are just the same but for A and B respectively?
    H_uu = hl_A - delta_hl_CNA * GFN2_coordinate_number(A, atoms)
    H_vv = hl_A - delta_hl_CNA * GFN2_coordinate_number(B, atoms)

    X_electronegativity = 1 if A[0] == B[0] else 1 + 0.02 * (0.35**2)
    R_AB = dist(v1,v2)**2
    # TODO: What is Rcov_AB? and k_poly?
    k_polyA = 0
    k_polyB = 0
    Rcov_AB = 0
    II = (1 + k_polyA (R_AB / Rcov_AB)**0.5)(1 + k_polyB (R_AB / Rcov_AB)**0.5)

    return 0.5 * Kll_AB * s_uv * (H_uu + H_vv) * X_electronegativity * II


# CN'_A
# atom: The atom to compute for
# atoms: All atoms
def GFN2_coordinate_number(atom, atoms):
    # TODO: Find these covalent radii values
    R_Acov = 0
    R_Bcov = 0

    A,v1 = atom
    for B,v2 in atoms:
        if A != B:
            R_AB = dist(v1,v2)**2
            (1 + e**(-10 * (4 * (R_Acov + R_Bcov)/3 * R_AB - 1)))**-1 * (1 + e**(-20 * (4 * (R_Acov + R_Bcov + 2)/3 * R_AB - 1)))**-1

