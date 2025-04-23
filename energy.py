from math import sqrt, exp
from gfn2 import kExpLight, kExpHeavy, repAlpha, repZeff, nShell, chemicalHardness, shellHardness, thirdOrderAtom, kshell, selfEnergy, kCN, shellPoly, slaterExponent, atomicRadii, paulingEN, kEN
import numpy as np
import time
H = 0
He = 1
C = 5

def density_initial_guess(element_cnt): #TODO: Make good initial guess
    return np.ones((element_cnt*3,element_cnt*3))/element_cnt*3

def overlap_initial_guess(element_cnt):
    return np.eye(element_cnt*3)

REP = True
ISO2 = True
ISO3 = True
EHT = True
CN = True

#element_ids = np.array([C,C,C])
#positions = np.array([[1,0,0],[0,1,0],[0,0,1]])
rand = np.random.default_rng()
element_cnt = 100
element_ids = rand.choice(repZeff.shape[0], size=element_cnt)
positions = rand.random((element_cnt,3))
atoms = list(zip(element_ids, positions))
density_matrix = density_initial_guess(element_cnt)
overlap_matrix = overlap_initial_guess(element_cnt)
def print_res(x1,x2,t1,t2,t3,label):
    print(f"{label}:")
    print(f"  normal  = {x1:.3f} in {t2-t1:.10f} sec")
    print(f"  numpy   = {x2:.3f} in {t3-t2:.10f} sec")
    print(f"  diff    = {((x1-x2)*100)/x2:.02f}%")
    print(f"  speedup = {((t2-t1))/(t3-t2):.02f}x")
    
def dist(v1, v2): #euclidean distance. 
    d = 0
    for a,b in zip(v1,v2):
        d += (a-b)**2
    d = sqrt(d)
    return d

def euclidian_dist_sqr(positions):
    pos_sqr = np.broadcast_to(np.sum(positions**2, axis=-1), (positions.shape[0], positions.shape[0])) 
    pos_pairs = np.matmul(positions, positions.transpose())
    dist_sqr = pos_sqr-2*pos_pairs+pos_sqr.transpose()
    dist_sqr = dist_sqr * (dist_sqr > 0) # remove sligthly negative values so the sqrt works fine. 
    return dist_sqr

def euclidian_dist(positions): 
    # res_1,2 = sqrt(sum((v1-v2)^2)) = sqrt(sum(v1^2)-2*v1.v2+sum(v2^2))
    '''
    [
    [sum(v1^2), sum(v2^2), sum(v3^2), ...],
    [sum(v1^2), sum(v2^2), sum(v3^2), ...],
    [sum(v1^2), sum(v2^2), sum(v3^2), ...],
    ...
    ]
    = 
    sum([
    [v1^2, v2^2, v3^2, ...],
    [v1^2, v2^2, v3^2, ...],
    [v1^2, v2^2, v3^2, ...],
    ...
    ], axis = -1)
    '''
    pos_sqr = np.broadcast_to(np.sum(positions**2, axis=-1), (positions.shape[0], positions.shape[0])) 
    '''
    [
    [v1.v1, v2.v1, v3.v1, ...],
    [v1.v2, v2.v2, v3.v2, ...],
    [v1.v3, v2.v3, v3.v3, ...],
    ...
    ]
    '''
    pos_pairs = np.matmul(positions, positions.transpose())
    
    '''
    [
    [sum(v1^2)-2*v1.v1+sum(v1^2), sum(v2^2)-2*v2.v1+sum(v1^2), sum(v3^2)-2*v3.v1+sum(v1^2), ...],
    [sum(v1^2)-2*v1.v2+sum(v2^2), sum(v2^2)-2*v2.v2+sum(v2^2), sum(v3^2)-2*v3.v2+sum(v2^2), ...],
    [sum(v1^2)-2*v1.v3+sum(v3^2), sum(v2^2)-2*v2.v3+sum(v3^2), sum(v3^2)-2*v3.v3+sum(v3^2), ...],
    ...
    ]
    '''
    dist_sqr = pos_sqr-2*pos_pairs+pos_sqr.transpose()
    dist_sqr = dist_sqr * (dist_sqr > 0) # remove sligthly negative values so the sqrt works fine. 
    dists = np.sqrt(dist_sqr)
    return dists


def repulsion_energy(atoms):
    acc = 0
    for i,(A,v1) in enumerate(atoms):
        for j,(B,v2) in enumerate(atoms):
            if i==j:
                continue
            kf = kExpLight if A in {H,He} and B in {H,He} else kExpHeavy
            R_AB = dist(v1,v2)
            frac = (repZeff[A] * repZeff[B])/R_AB
            acc += frac*exp(-sqrt(repAlpha[A]*repAlpha[B])*(R_AB**kf)) 

    acc *= 0.5
    return acc

def repulsion_energy_np(element_ids, positions):
    heavies = element_ids > He
    kfs = np.outer(heavies, heavies)*(kExpHeavy-kExpLight) + kExpLight
    repZeffs = repZeff[element_ids]
    R_ABs = euclidian_dist(positions)
    np.fill_diagonal(R_ABs,1) #avoid division by zero.
    frac = np.outer(repZeffs,repZeffs)/R_ABs
    repAlphas = repAlpha[element_ids]
    energies = frac*np.exp(-np.sqrt(np.outer(repAlphas,repAlphas))*(R_ABs**kfs))
    np.fill_diagonal(energies,0) # repulsion with it self is excluded. 
    repE = 0.5*np.sum(energies)
    return repE
if REP:
    t1 = time.time()
    x1 = repulsion_energy(atoms)
    t2 = time.time()
    x2 = repulsion_energy_np(element_ids, positions)
    t3 = time.time()
    print_res(x1,x2,t1,t2,t3,"repulsion")

def get_partial_mulliken_charges(density_matrix, overlap_matrix):
    return np.sum(density_matrix*overlap_matrix, axis=-1).reshape(-1,3)

def isotropic_electrostatic_and_XC_energy_second_order(atoms, charges):
#    eta_ABs = np.zeros((element_ids.shape[0]*3,element_ids.shape[0]*3))
#    gamma_ABs = np.zeros((element_ids.shape[0]*3,element_ids.shape[0]*3))
#    energies = np.zeros((element_ids.shape[0]*3,element_ids.shape[0]*3))
    acc = 0
    for i,(A,v1) in enumerate(atoms):
        etaA = chemicalHardness[A]
        for j,(B,v2) in enumerate(atoms):
            etaB = chemicalHardness[B]
            R_AB2 = dist(v1,v2)**2
            for u in range(nShell[A]):
                kuA = shellHardness[A][u]
                for v in range(nShell[B]):
                    kvB = shellHardness[B][v]
                    eta_ABuv = 0.5*(etaA*(1+kuA)+etaB*(1+kvB))
                    gamma_ABuv = 1./sqrt(R_AB2+eta_ABuv**(-2))
                    e = charges[u][u]*charges[j][v]*gamma_ABuv
                    acc += e
#                    eta_ABs[i*3+u,j*3+v] = eta_ABuv
#                    gamma_ABs[i*3+u,j*3+v] = gamma_ABuv
#                    energies[i*3+u,j*3+v] = e
#    print(eta_ABs)
#    print(gamma_ABs)
#    print(energies)
    acc *= 0.5
    return acc

def isotropic_electrostatic_and_XC_energy_second_order_np(element_ids, positions, charges):
    R_AB2 = np.repeat(euclidian_dist_sqr(positions),3,axis=0)
    R_AB2 = np.repeat(R_AB2,3,axis=1)
    ks = (shellHardness[element_ids]+1).flatten()
    etas = np.broadcast_to(np.repeat(chemicalHardness[element_ids],3)*(ks), (ks.shape[0], ks.shape[0]))
    eta_ABs = (etas + etas.transpose())*0.5
    us = np.repeat(np.array([[0,1,2]]),element_ids.shape[0],axis=0).flatten()
    include_shell = np.repeat(nShell[element_ids], 3) > us
    include_shell = np.outer(include_shell, include_shell)
    gamma_ABs = 1./np.sqrt(R_AB2+(eta_ABs**(-2)))
    energies = np.outer(charges.flatten(), charges.flatten())*gamma_ABs
#    print(eta_ABs*include_shell)
#    print(gamma_ABs*include_shell)
#    print(energies*include_shell)
    return np.sum(energies*include_shell)*0.5

if ISO2:
    partial_mulliken_charges = get_partial_mulliken_charges(density_matrix, overlap_matrix)
    t1 = time.time()
    x1 = isotropic_electrostatic_and_XC_energy_second_order(atoms, partial_mulliken_charges)
    t2 = time.time()
    x2 = isotropic_electrostatic_and_XC_energy_second_order_np(element_ids, positions, partial_mulliken_charges)
    t3 = time.time()
    print_res(x1,x2,t1,t2,t3,"isotropic electrostatic and XC energy second order")

def isotropic_electrostatic_and_XC_energy_third_order(atoms, charges):
    acc = 0
    for i,(A,_) in enumerate(atoms):
        for u in range(nShell[A]):
            acc += (charges[i][u]**3)*(2.**-u)*thirdOrderAtom[A]
    acc /= 3.
    return acc

def isotropic_electrostatic_and_XC_energy_third_order_np(element_ids, positions, charges):
    us = np.repeat(np.array([[0,1,2]]),element_ids.shape[0],axis=0).flatten()
    include_shell = np.repeat(nShell[element_ids], 3) > us
    energies = (charges.flatten()**3)*(2.**(-us))*np.repeat(thirdOrderAtom[element_ids],3)
    return np.sum(energies*include_shell)/3.

if ISO3:
    t1 = time.time()
    x1 = isotropic_electrostatic_and_XC_energy_third_order(atoms, partial_mulliken_charges)
    t2 = time.time()
    x2 = isotropic_electrostatic_and_XC_energy_third_order_np(element_ids, positions, partial_mulliken_charges)
    t3 = time.time()
    print_res(x1*10**6,x2*10**6,t1,t2,t3,"isotropic electrostatic and XC energy third order")

Kll_AB = np.array([
    [1.85,2.04,2.00],
    [2.04,2.23,2.00],
    [2.00,2.00,2.23]
])

def extended_huckel_energy(atoms, density_matrix, overlap_matrix):
#    n3 = element_ids.shape[0]*3
#    include_shell = np.zeros((n3,n3))
#    Kuv_ABs = np.zeros((n3,n3))
#    P_uvs = np.zeros((n3,n3))
#    s_uvs = np.zeros((n3,n3))
#    hl_X = np.zeros((n3,))
#    delta_hl_CNX = np.zeros((n3,))
#    coordination_numbers = np.zeros((n3,))
#    H_xx = np.zeros((n3,))
#    H_uuvv = np.zeros((n3,n3))
#    X_electronegativities = np.zeros((n3,n3))
#    k_polyX = np.zeros((n3,n3))
#    R_AB2 = np.zeros((n3,n3))
#    Rcov_ABs = np.zeros((n3,n3))
#    IIs = np.zeros((n3,n3))
#    Ys = np.zeros((n3,n3))
#    res = np.zeros((n3,n3))
    acc = 0
    for i,(A,_) in enumerate(atoms):
        for j,(B,_) in enumerate(atoms):
            for u in range(nShell[A]):
                for v in range(nShell[B]):
                    P_uv = density_matrix[i*3+u][j*3+v]
                    s_uv = overlap_matrix[i*3+u][j*3+v]

                    Kuv_AB = Kll_AB[u][v] 
                    
                    A,v1 = atoms[i]
                    B,v2 = atoms[j]
                    hl_A = selfEnergy[A][u]
                    hl_B = selfEnergy[B][v]
                    delta_hl_CNA = kCN[A][u]
                    delta_hl_CNB = kCN[B][v]
                    H_uu = hl_A - delta_hl_CNA * GFN2_coordination_number(i, atoms)
                    H_vv = hl_B - delta_hl_CNB * GFN2_coordination_number(j, atoms)

                    X_electronegativity = 1 + kEN*((paulingEN[A]-paulingEN[B])**2)
                    R_AB = dist(v1,v2)**2
                    k_polyA = shellPoly[A][u]
                    k_polyB = shellPoly[B][v]
                    Rcov_AB = atomicRadii[A] + atomicRadii[B]
                    II = (1 + k_polyA * (R_AB / Rcov_AB)**0.5) * (1 + k_polyB * (R_AB / Rcov_AB)**0.5)

                    Y = ((2 * sqrt(slaterExponent[A][u] * slaterExponent[B][v])) / (slaterExponent[A][u] + slaterExponent[B][v]))**0.5
                    e = P_uv * (0.5 * Kuv_AB * s_uv * (H_uu + H_vv) * X_electronegativity * II * Y)
                    acc += e
#                    P_uvs[i*3+u][j*3+v] = P_uv
#                    s_uvs[i*3+u][j*3+v] = s_uv
#                    Kuv_ABs[i*3+u][j*3+v] = Kuv_AB
#                    hl_X[i*3+u] = hl_A
#                    delta_hl_CNX[i*3+u] = delta_hl_CNA
#                    coordination_numbers[i*3+u] = GFN2_coordination_number(i, atoms)
#                    H_xx[i*3+u] = H_uu
#                    H_uuvv[i*3+u][j*3+v] = H_uu + H_vv
#                    X_electronegativities[i*3+u][j*3+v] = X_electronegativity
#                    k_polyX[i*3+u][j*3+v] = k_polyA
#                    R_AB2[i*3+u][j*3+v] = R_AB
#                    Rcov_ABs[i*3+u][j*3+v] = Rcov_AB
#                    IIs[i*3+u][j*3+v] = II
#                    Ys[i*3+u][j*3+v] = Y
#                    res[i*3+u][j*3+v] = e
#                    include_shell[i*3+u][j*3+v] = 1
#    print(P_uvs) 
#    print(s_uvs) 
    return acc

def huckel_matrix_np(element_ids):
    hl_X = selfEnergy[element_ids].flatten()
    delta_hl_CNX = kCN[element_ids, :3].flatten()
    coordination_numbers = np.repeat(GFN2_coordination_numbers_np(element_ids, positions),3)
    H_xx = hl_X - delta_hl_CNX * coordination_numbers
    H_xxs = np.broadcast_to(H_xx, (H_xx.shape[0], H_xx.shape[0]))
    H_uuvv = H_xxs + H_xxs.transpose()
    return H_uuvv

def extended_huckel_energy_np(element_ids, positions, density_matrix, overlap_matrix):
    us = np.repeat(np.array([[0,1,2]]),element_ids.shape[0],axis=0).flatten()
    include_shell = np.repeat(nShell[element_ids], 3) > us
    include_shell = np.outer(include_shell, include_shell)
    Kuv_AB = np.tile(Kll_AB, (element_ids.shape[0], element_ids.shape[0]))
    P_uv = density_matrix#np.take(density_matrix[np.repeat(element_ids*3, 3) + us], np.repeat(element_ids*3, 3) + us, axis = 1)
    s_uv = overlap_matrix#np.take(overlap_matrix[np.repeat(element_ids*3, 3) + us], np.repeat(element_ids*3, 3) + us, axis = 1)
    H_EHT = huckel_matrix_np(element_ids)
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
    II = 1 + k_polyX * (R_AB2 / Rcov_AB)**0.5
    II = II * II.transpose()
    slaterExponents = np.broadcast_to(slaterExponent[element_ids, :3].flatten(), (element_ids.shape[0]*3,element_ids.shape[0]*3))
    slaterExponents_ABs = slaterExponents * slaterExponents.transpose()
    Y = ((2 * np.sqrt(slaterExponents_ABs)) / (slaterExponents + slaterExponents.transpose() + np.logical_not(include_shell)))**0.5
    res = P_uv * (0.5 * Kuv_AB * s_uv * H_EHT * X_electronegativities * II * Y * include_shell)
    return np.sum(res)



# CN'_A
# A_idx: The index of the atom to compute for
# atoms: All atoms
def GFN2_coordination_number(A_idx, atoms):
    A,v1 = atoms[A_idx]
    R_Acov = atomicRadii[A]

    acc = 0
    for i,(B,v2) in enumerate(atoms):
        if i != A_idx:
            R_Bcov = atomicRadii[B]
            R_AB = dist(v1,v2)**2
            acc += (1 + exp(-10 * (4 * (R_Acov + R_Bcov)/3 * R_AB - 1)))**-1 * (1 + exp(-20 * (4 * (R_Acov + R_Bcov + 2)/3 * R_AB - 1)))**-1

    return acc

def GFN2_coordination_numbers_np(element_ids, positions):
    R_cov = atomicRadii[element_ids]
    R_cov_stack = np.broadcast_to(R_cov, (R_cov.shape[0], R_cov.shape[0])) 
    R_covs = R_cov_stack + R_cov_stack.transpose()
    distances = euclidian_dist(positions)**2
    res = (1 + np.exp(-10 * (4 * R_covs/3 * distances - 1)))**-1 * (1 + np.exp(-20 * (4 * (R_covs+2)/3 * distances - 1)))**-1
    np.fill_diagonal(res,0)
    coordination_numbers = np.sum(res, axis=1)
    return coordination_numbers

if CN:
    t1 = time.time()
    x1 = 0
    for A_idx,_ in enumerate(atoms):
        x1 += GFN2_coordination_number(A_idx, atoms)
    t2 = time.time()
    x2 = np.sum(GFN2_coordination_numbers_np(element_ids, positions))
    t3 = time.time()
    print_res(x1,x2,t1,t2,t3,"coordination numbers")

if EHT:
    t1 = time.time()
    x1 = extended_huckel_energy(atoms, density_matrix, overlap_matrix)
    t2 = time.time()
    x2 = extended_huckel_energy_np(element_ids, positions, density_matrix, overlap_matrix)
    t3 = time.time()
    print_res(x1,x2,t1,t2,t3,"extended huckel energy")
