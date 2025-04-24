from gfn2 import kExpLight, kExpHeavy, repAlpha, repZeff, nShell, chemicalHardness, shellHardness, thirdOrderAtom, kshell, selfEnergy, kCN, shellPoly, slaterExponent, atomicRadii, paulingEN, kEN
import numpy as np
import time
from math import sqrt
from util import euclidian_dist, dist, print_res1, density_initial_guess, overlap_initial_guess, get_partial_mulliken_charges

def huckel_matrix_np(element_ids, positions):
    hl_X = selfEnergy[element_ids].flatten()
    delta_hl_CNX = kCN[element_ids, :3].flatten()
    coordination_numbers = np.repeat(GFN2_coordination_numbers_np(element_ids, positions),3)
    H_xx = hl_X - delta_hl_CNX * coordination_numbers
    H_xxs = np.broadcast_to(H_xx, (H_xx.shape[0], H_xx.shape[0]))
    H_uuvv = H_xxs + H_xxs.transpose()
    return H_uuvv


def GFN2_coordination_numbers_np(element_ids, positions):
    R_cov = atomicRadii[element_ids]
    R_cov_stack = np.broadcast_to(R_cov, (R_cov.shape[0], R_cov.shape[0])) 
    R_covs = R_cov_stack + R_cov_stack.transpose()
    distances = euclidian_dist(positions)**2
    res = (1 + np.exp(-10 * (4 * R_covs/3 * distances - 1)))**-1 * (1 + np.exp(-20 * (4 * (R_covs+2)/3 * distances - 1)))**-1
    np.fill_diagonal(res,0)
    coordination_numbers = np.sum(res, axis=1)
    return coordination_numbers

def fock_isotropic_electrostatic_and_exchange_correlation(element_ids, positions, overlap_matrix, partial_mulliken_charges):
    fock_matrix = [[0]*len(overlap_matrix[0])]*len(overlap_matrix)
    for i,A in enumerate(element_ids):
        etaA = chemicalHardness[A]
        v1 = positions[i]
        for j,B in enumerate(element_ids):
            v2 = positions[j]
            etaB = chemicalHardness[B]
            for l in range(nShell[A]):
                klA = shellHardness[A][l]
                for lp in range(nShell[B]):
                    klpB = shellHardness[B][lp]
                    minus_half_S_uv = -0.5*overlap_matrix[i*3+l][j*3+lp]
                    Gamma_Al = (2.**-l)*thirdOrderAtom[A]
                    Gamma_Blp = (2.**-lp)*thirdOrderAtom[B]
                    fock_matrix[i*3+l][j*3+lp] += minus_half_S_uv*(
                        partial_mulliken_charges[i][l]**2*Gamma_Al+
                        partial_mulliken_charges[j][lp]**2*Gamma_Blp)
                    for k,C in enumerate(element_ids):
                        v3 = positions[k]
                        etaC = chemicalHardness[C]
                        R_AC2 = dist(v1,v3)**2
                        R_BC2 = dist(v2,v3)**2
                        for lpp in range(nShell[C]):
                            klppC = shellHardness[C][lpp]
                            eta_ACllpp = 0.5*(etaA*(1+klA)+etaC*(1+klppC))
                            eta_BClplpp = 0.5*(etaB*(1+klpB)+etaC*(1+klppC))
                            gamma_ACllpp = 1./sqrt(R_AC2+eta_ACllpp**(-2))
                            gamma_BClplpp = 1./sqrt(R_BC2+eta_BClplpp**(-2))
                            fock_matrix[i*3+l][j*3+lp] += minus_half_S_uv*(
                                    gamma_ACllpp + gamma_BClplpp
                                )*partial_mulliken_charges[k][lpp]
    return fock_matrix
if __name__ == "__main__":
    ISO = True
    EHT = True

    #element_ids = np.array([C,C,C])
    #positions = np.array([[1,0,0],[0,1,0],[0,0,1]])
    rand = np.random.default_rng()
    element_cnt = 100
    element_ids = rand.choice(repZeff.shape[0], size=element_cnt)
    positions = rand.random((element_cnt,3))
    atoms = list(zip(element_ids, positions))
    density_matrix = density_initial_guess(element_cnt)
    overlap_matrix = overlap_initial_guess(element_cnt)
    if ISO:
        partial_mulliken_charges = get_partial_mulliken_charges(density_matrix, overlap_matrix)
        t1 = time.time()
        x1 = fock_isotropic_electrostatic_and_exchange_correlation(element_ids, positions, overlap_matrix, partial_mulliken_charges)
        t2 = time.time()
        print_res1(np.sum(x1),t1,t2,"Isotropic")

