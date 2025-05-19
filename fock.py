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
#    res = (1 + np.exp(-10 * (4 * R_covs/3 * distances - 1)))**-1 * (1 + np.exp(-20 * (4 * (R_covs+2)/3 * distances - 1)))**-1
    res = 1.0/(1.0+np.exp(-10*(R_covs/distances-1.0)))*1.0/(1.0+np.exp(-2*10.0*((R_covs+2)/distances-1.0)))
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
            #CNp_A += (1 + exp(-10 * (4 * (R_Acov + R_Bcov)/3 * R_AB - 1)))**-1 * (1 + exp(-20 * (4 * (R_Acov + R_Bcov + 2)/3 * R_AB - 1)))**-1
            CNp_A += 1.0/(1.0+np.exp(-10*((R_Acov+R_Bcov)/R_AB-1.0)))*1.0/(1.0+np.exp(-2*10.0*((R_Acov+R_Bcov+2)/R_AB-1.0)))
    return CNp_A

def ncoordLatP(element_ids, positions):
    rcov = np.array([0.80628307170014579, 1.1590319155689597, 3.0235615188755465, 2.3684565231191779, 1.9401186412784759, 1.8897259492972165, 1.7889405653346984, 1.5873697974096619, 1.6125661434002916, 1.6881551813721805, 3.5274884386881378, 3.1495432488286945, 2.8471870969411395, 2.6204199830254740, 2.7715980589692513, 2.5700272910442146, 2.4944382530723259, 2.4188492151004373, 4.4345568943508020, 3.8802372825569518, 3.3511140167537312, 3.0739542108568059, 3.0487578648661766, 2.7715980589692513, 2.6960090209973626, 2.6204199830254740, 2.5196345990629556, 2.4944382530723259, 2.5448309450535853, 2.7464017129786220, 2.8219907509505107, 2.7464017129786220, 2.8975797889223984, 2.7715980589692513, 2.8723834429317692, 2.9479724809036578, 4.7621093922289859, 4.2077897804351361, 3.7038628606225448, 3.5022920926975076, 3.3259176707631020, 3.1243469028380648, 2.8975797889223984, 2.8471870969411395, 2.8471870969411395, 2.7212053669879919, 2.8975797889223984, 3.0991505568474356, 3.2251322868005832, 3.1747395948193238, 3.1747395948193238, 3.0991505568474356, 3.3259176707631020, 3.3007213247724718, 5.2660363120415772, 4.4345568943508020, 4.0818080504819880, 3.7038628606225448, 3.9810226665194701, 3.9558263205288404, 3.9306299745382112, 3.9054336285475810, 3.8046482445850631, 3.8298445905756928, 3.8046482445850631, 3.7794518985944330, 3.7542555526038037, 3.7542555526038037, 3.7290592066131740, 3.8550409365663221, 3.6786665146319151, 3.4518994007162491, 3.3007213247724718, 3.0991505568474356, 2.9731688268942875, 2.9227761349130286, 2.7967944049598810, 2.8219907509505107, 2.8471870969411395, 3.3259176707631020, 3.2755249787818421, 3.2755249787818421, 3.4267030547256199, 3.3007213247724718, 3.4770957467068784, 3.5778811306693967, 5.0644655441165405, 4.5605386243039501, 4.2077897804351361, 3.9810226665194701, 3.8298445905756928, 3.8550409365663221, 3.8802372825569518, 3.9054336285475810, 3.7542555526038037, 3.7542555526038037, 3.8046482445850631, 3.8046482445850631, 3.7290592066131740, 3.7794518985944330, 3.9306299745382112, 3.9810226665194701, 3.6534701686412858, 3.5526847846787675, 3.3763103627443609, 3.2503286327912129, 3.1999359408099539, 3.0487578648661766, 2.9227761349130286, 2.8975797889223984, 2.7464017129786220, 3.0739542108568059, 3.4267030547256199, 3.6030774766600264, 3.6786665146319151, 3.9810226665194701, 3.7290592066131740, 3.9558263205288404], dtype = np.float64)
    cutoff = 40.0
#   cn = 0.0_wp
    cn = np.zeros(element_ids.shape[0])
#   dcndr = 0.0_wp
#   dcndL = 0.0_wp
#   cutoff2 = cutoff**2
    cutoff2 = cutoff**2
#
#   !$omp parallel do default(none) private(den) shared(enscale, rcov, en)&
#   !$omp reduction(+:cn, dcndr, dcndL) shared(mol, kcn, trans, cutoff2) &
#   !$omp private(jat, itr, ati, atj, r2, rij, r1, rc, countf, countd, stress)
    
#   !> Parameter for electronegativity scaling
    k4=4.10451

#   !> Parameter for electronegativity scaling
    k5=19.08857

#   !> Parameter for electronegativity scaling
    k6=2*11.28174**2
#   do iat = 1, len(mol)
    for iat in range(element_ids.shape[0]):
#      ati = mol%at(iat)
        ati = element_ids[iat]
#      do jat = 1, iat
        for jat in range(iat+1):
#         atj = mol%at(jat)
            atj = element_ids[jat]
#
#         if (enscale) then
#            den = k4*exp(-(abs(en(ati)-en(atj)) + k5)**2/k6)
#            den = k4*np.exp(-(np.abs(paulingEN[ati]-paulingEN[atj]) + k5)**2/k6)
            den = 1
#         else
#            den = 1.0_wp
#         end if
#
#         do itr = 1, size(trans, dim=2)
#            rij = mol%xyz(:, iat) - (mol%xyz(:, jat) + trans(:, itr))
            rij = positions[iat, :] - positions[jat, :]
#            r2 = sum(rij**2)
            r2 = np.sum(rij**2)
#            if (r2 > cutoff2 .or. r2 < 1.0e-12_wp) cycle
            if (r2 > cutoff2 or r2 < 1.0e-12):
                continue
#            r1 = sqrt(r2)
            r1 = np.sqrt(r2)
#
#            rc = rcov(ati) + rcov(atj)
            rc = rcov[ati] + rcov[atj]
            
#
#            countf = den * cfunc(kcn, r1, rc)
            kcn = 10.0
            countf = den * (1.0/(1.0+np.exp(-kcn*(rc/r1-1.0))))*(1.0/(1.0+np.exp(-2*kcn*((rc+2)/r1-1.0))))
#            countd = den * dfunc(kcn, r1, rc) * rij/r1
#
#            cn(iat) = cn(iat) + countf
            cn[iat] += countf
#            if (iat /= jat) then
#               cn(jat) = cn(jat) + countf
#            end if
            if iat != jat:
                cn[jat] += countf
    return cn
#
#            dcndr(:, iat, iat) = dcndr(:, iat, iat) + countd
#            dcndr(:, jat, jat) = dcndr(:, jat, jat) - countd
#            dcndr(:, iat, jat) = dcndr(:, iat, jat) + countd
#            dcndr(:, jat, iat) = dcndr(:, jat, iat) - countd
#
#            stress = spread(countd, 1, 3) * spread(rij, 2, 3)
#
#            dcndL(:, :, iat) = dcndL(:, :, iat) + stress
#            if (iat /= jat) then
#               dcndL(:, :, jat) = dcndL(:, :, jat) + stress
#            end if
#
#         end do
#      end do
#   end do
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

