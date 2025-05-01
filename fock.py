from gfn2 import kExpLight, kExpHeavy, repAlpha, repZeff, nShell, chemicalHardness, shellHardness, thirdOrderAtom, kshell, selfEnergy, kCN, shellPoly, slaterExponent, atomicRadii, paulingEN, kEN, multiRad, cnRMax, N_val,cnShift, dipDamp, quadDamp, dipKernel, quadKernel
import numpy as np
import time
from math import sqrt, exp
from util import euclidian_dist, euclidian_dist_sqr, dist, print_res1, density_initial_guess, overlap_initial_guess, get_partial_mulliken_charges, get_atomic_charges

def huckel_matrix_np(element_ids, positions, overlap_matrix):
    Kll_AB = np.array([
        [1.85,2.04,2.00],
        [2.04,2.23,2.00],
        [2.00,2.00,2.23]
    ])
    us = np.repeat(np.array([[0,1,2]]),element_ids.shape[0],axis=0).flatten() #[0,1,2]*n shell idx for every shell u flattened. 
    include_shell = np.repeat(nShell[element_ids], 3) > us # include_shell[i*3+u] = true if that shell is part of the basis set. 
    include_shell = np.outer(include_shell, include_shell) # include_shell[i*3+u][j*3+v] = true if both shells are part of the basis set.
    Kuv_AB = np.tile(Kll_AB, (element_ids.shape[0], element_ids.shape[0]))
    hl_X = selfEnergy[element_ids].flatten()
    delta_hl_CNX = kCN[element_ids, :3].flatten()
    coordination_numbers = np.repeat(GFN2_coordination_numbers_np(element_ids, positions),3)
    H_xx = hl_X - delta_hl_CNX * coordination_numbers
    H_xxs = np.broadcast_to(H_xx, (H_xx.shape[0], H_xx.shape[0]))
    H_uuvv = H_xxs + H_xxs.transpose()
    electronegativity = paulingEN[np.repeat(element_ids, 3)]
    electronegativities = np.broadcast_to(electronegativity, (electronegativity.shape[0], electronegativity.shape[0]))
    X_electronegativities = 1+kEN*((electronegativities - electronegativities.transpose())**2)
    k_polyX = shellPoly[element_ids, :3].flatten()
    k_polyX = np.broadcast_to(k_polyX, (k_polyX.shape[0], k_polyX.shape[0])).transpose()
    R_AB2 = np.repeat(euclidian_dist_sqr(positions),3,axis=0)
    R_AB2 = np.repeat(R_AB2,3,axis=1)
    atomicRadiis = np.repeat(atomicRadii[element_ids], 3)
    atomicRadiis_mat = np.broadcast_to(atomicRadiis, (atomicRadiis.shape[0], atomicRadiis.shape[0]))
    Rcov_AB = atomicRadiis_mat + atomicRadiis_mat.transpose()
    Pi_u = 1 + k_polyX * (R_AB2 / Rcov_AB)**0.5
    Pi_uv = Pi_u * Pi_u.transpose()
    slaterExponents = np.broadcast_to(slaterExponent[element_ids, :3].flatten(), (element_ids.shape[0]*3,element_ids.shape[0]*3))
    slaterExponents_ABs = slaterExponents * slaterExponents.transpose()
    Y = ((2 * np.sqrt(slaterExponents_ABs)) / (slaterExponents + slaterExponents.transpose() + np.logical_not(include_shell)))**0.5
    return 0.5 * Kuv_AB * overlap_matrix * H_uuvv * X_electronegativities * Pi_uv * Y * include_shell


def GFN2_coordination_numbers_np(element_ids, positions):
    R_cov = atomicRadii[element_ids]
    R_cov_stack = np.broadcast_to(R_cov, (R_cov.shape[0], R_cov.shape[0])) 
    R_covs = R_cov_stack + R_cov_stack.transpose()
    distances = euclidian_dist(positions)**2
    res = (1 + np.exp(-10 * (4 * R_covs/3 * distances - 1)))**-1 * (1 + np.exp(-20 * (4 * (R_covs+2)/3 * distances - 1)))**-1
    np.fill_diagonal(res,0)
    coordination_numbers = np.sum(res, axis=1)
    return coordination_numbers

# CN'_A
def GFN2_coordination_number(A_idx, element_ids, positions):
    A = element_ids[A_idx]
    v1 = positions[A_idx]
    R_Acov = atomicRadii[A]
    CNp_A = 0
    for B_idx,B in enumerate(element_ids):
        if B_idx != A_idx:
            v2 = positions[B_idx]
            R_Bcov = atomicRadii[B]
            R_AB = dist(v1,v2)**2
            CNp_A += (1 + exp(-10 * (4 * (R_Acov + R_Bcov)/3 * R_AB - 1)))**-1 * (1 + exp(-20 * (4 * (R_Acov + R_Bcov + 2)/3 * R_AB - 1)))**-1
    return CNp_A

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


def dip_matrix(overlap_matrix, positions):
    n = positions.shape[0]
    pos_u = np.repeat(positions, 3, axis = 0)
    pos_u_stack = np.broadcast_to(pos_u, (n*3,n*3,3))
    D = np.multiply(np.repeat(overlap_matrix,3,axis=-1).reshape((n*3,n*3,3)),pos_u_stack) # D_A_lB_lp^a = D[A_l][B_lp][A_a]
    return D


def V_S(C_idx, element_ids, positions, density_matrix, dip_matrix, charges):
    C = element_ids[C_idx]
    bRC = positions[C_idx]
    RCp_0 = 5.0
    if N_val[C] != 0:
        RCp_0 = multiRad[C] + (cnRMax-multiRad[C])/(1+exp(-4*(GFN2_coordination_number(C_idx, element_ids, positions)-N_val[C]-cnShift)))
    V_S = 0
    for A_idx, A in enumerate(element_ids):
        bRA = positions[A_idx]
        R_AC = dist(bRA,bRC)
        RAp_0 = 5.0
        if N_val[A] != 0:
            RAp_0 = multiRad[A] + (cnRMax-multiRad[A])/(1+exp(-4*(GFN2_coordination_number(A_idx, element_ids, positions)-N_val[A]-cnShift)))

        RAC_0 = 0.5*(RAp_0+RCp_0)
        if A_idx == C_idx: #avoid division by zero
            continue
        f3 = (1/(R_AC**3)) * (1/(1+6*((RAC_0/R_AC)**dipDamp)))
        f5 = (1/(R_AC**5)) * (1/(1+6*((RAC_0/R_AC)**quadDamp)))
        bRAC = bRC-bRA
        bRAC_2 = bRAC**2
        mu_A = np.array([0,0,0])
        theta_A = np.array([[0,0,0],[0,0,0],[0,0,0]])
        for l in range(3):
            for B_idx, B in enumerate(element_ids):
                bRB = positions[B_idx]
                for lp in range(3):
                    for alpha in range(3):
                        DA = dip_matrix[A_idx*3 + l][B_idx*3+lp][alpha] # read as <\phi l | alpha_A |\phi lp> meaning that alpha is from the position of A Q<alpha_pos>

                        DB = dip_matrix[B_idx*3 + lp][A_idx*3+l][alpha]
                        mu_A[alpha] += density_matrix[A_idx*3+l][B_idx*3+lp]*(DA-DB) 
                        for beta in range(3):
                            QAA = DA * bRA[beta] # read as <\phi l | alpha_A beta_A |\phi lp> meaning that alpha is from the position of A and beta from the position of A hence Q<alpha_pos><beta_pos>
                            QAB = DA * bRB[beta]
                            QBA = DB * bRA[beta]
                            QBB = DB * bRB[beta]
                            theta_A[alpha][beta] += density_matrix[A_idx*3+l][B_idx*3+lp]*(QAB+QBA-QAA-QBB)
        braket =f5*mu_A*R_AC - bRAC*3*f5*np.inner(mu_A,bRAC_2) - f3*charges[A_idx]*bRAC 
        V_S_A = np.inner(bRC, braket) \
            - f5*np.inner(bRAC, np.matvec(theta_A, bRAC)) - f3*np.inner(mu_A,bRAC) \
            + charges[A_idx]*f5*0.5*np.inner(bRC**2,bRAC_2) \
            - 3./2.*charges[A_idx]*f5*np.sum(np.outer(bRAC,bRAC)*np.outer(bRC,bRC)) #TODO: should this be really be bRAC not bRAB? where would we get B?
        V_S += V_S_A
    mu_C = np.array([0,0,0])
    theta_C = np.array([[0,0,0],[0,0,0],[0,0,0]])
    for l in range(3):
        for B_idx, B in enumerate(element_ids):
            bRB = positions[B_idx]
            for lp in range(3):
                for alpha in range(3):
                    DC = dip_matrix[C_idx*3 + l][B_idx*3+lp][alpha]# read as <\phi l | alpha_A |\phi lp> meaning that alpha is from the position of A Q<alpha_pos>
                    DB = dip_matrix[B_idx*3 + lp][C_idx*3+l][alpha]
                    mu_C[alpha] += density_matrix[C_idx*3+l][B_idx*3+lp]*(DC-DB) 
                    for beta in range(3):
                        QCC = DC * bRC[beta] # read as <\phi l | alpha_A beta_A |\phi lp> meaning that alpha is from the position of A and beta from the position of A hence Q<alpha_pos><beta_pos>
                        QCB = DC * bRB[beta]
                        QBC = DB * bRC[beta]
                        QBB = DB * bRB[beta]
                        theta_C[alpha][beta] += density_matrix[C_idx*3+l][B_idx*3+lp]*(QCB+QBC-QCC-QBB)
    V_S += 2*dipKernel[C]*np.inner(bRC,mu_C) - quadKernel[C]*np.inner(bRC,np.matvec(3*theta_C-np.trace(theta_C)*np.eye(3,3),bRC)) # do the part outside of the sum of V_S_A 
    return V_S
                    
def fock_anisotropic_electrostatic_and_exchange_correlation(element_ids, positions, density_matrix,  overlap_matrix, dip_matrix, atomic_charges):
    fock_matrix = np.zeros((overlap_matrix.shape))
    for B_idx, B in enumerate(element_ids):
        for l in range(3):
            for C_idx, C in enumerate(element_ids):
                for lp in range(3):
                    fock_matrix[B_idx*3+l][C_idx*3+lp] = 0.5*overlap_matrix[B_idx*3+l][C_idx*3+lp]*(
                        V_S(B_idx, element_ids, positions, density_matrix, dip_matrix, atomic_charges)\
                        +V_S(C_idx, element_ids, positions, density_matrix, dip_matrix, atomic_charges))
                    # TODO: V_D
                    # TODO: V_Q
    return fock_matrix
    
if __name__ == "__main__":
    ISO = False
    ANISO = True
    EHT = False
    VS = True
    #element_ids = np.array([C,C,C])
    #positions = np.array([[1,0,0],[0,1,0],[0,0,1]])
    rand = np.random.default_rng()
    element_cnt = 10
    element_ids = rand.choice(repZeff.shape[0], size=element_cnt)
    positions = rand.random((element_cnt,3))
    atoms = list(zip(element_ids, positions))
    density_matrix = density_initial_guess(element_cnt)
    overlap_matrix = overlap_initial_guess(element_cnt)
    dip_matrix = dip_matrix(overlap_matrix, positions)
    atomic_charges = get_atomic_charges(density_matrix, overlap_matrix, element_ids)
    if VS:
        t1 = time.time()
        x1 = V_S(0, element_ids, positions, density_matrix, dip_matrix, atomic_charges)
        t2 = time.time()
        print_res1(np.sum(x1),t1,t2,"Anisotropic V_S")
    if ANISO:
        t1 = time.time()
        x1 = fock_anisotropic_electrostatic_and_exchange_correlation(element_ids, positions, density_matrix,  overlap_matrix, dip_matrix, atomic_charges)
        t2 = time.time()
        print_res1(np.sum(x1),t1,t2,"Anisotropic fock matrix")
    if ISO:
        partial_mulliken_charges = get_partial_mulliken_charges(density_matrix, overlap_matrix)
        t1 = time.time()
        x1 = fock_isotropic_electrostatic_and_exchange_correlation(element_ids, positions, overlap_matrix, partial_mulliken_charges)
        t2 = time.time()
        print_res1(np.sum(x1),t1,t2,"Isotropic fock matrix")

