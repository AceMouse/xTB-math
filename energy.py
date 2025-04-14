from math import sqrt, exp
from gfn2 import kExpLight, kExpHeavy, repAlpha, repZeff, nShell, chemicalHardness, shellHardness, thirdOrderAtom, kshell, selfEnergy, kCN, shellPoly, slaterExponent, atomicRadii
import numpy as np
import time
H = 0
He = 1
C = 5


#element_ids = np.array([C,C,C])
#positions = np.array([[1,0,0],[0,1,0],[0,0,1]])
rand = np.random.default_rng()
element_ids = rand.choice(repZeff.shape[0], size=10)
positions = rand.random((10,3))
atoms = list(zip(element_ids, positions))
 
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

t1 = time.time()
print(repulsion_energy(atoms))
t2 = time.time()
print(repulsion_energy_np(element_ids, positions))
t3 = time.time()
print("normal:", t2-t1)
print("np:", t3-t2)

def get_partial_mulliken_charges(density_matrix, overlap_matrix):
    return np.sum(density_matrix*overlap_matrix, axis=-1).reshape(-1,3)

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

def isotropic_electrostatic_and_XC_energy_second_order_np(element_ids, positions, charges):
    R_AB2 = np.repeat(euclidian_dist_sqr(positions),3,axis=0)
    R_AB2 = np.repeat(R_AB2,3,axis=1)
    ks = (shellHardness[element_ids]+1).flatten()
    etas = np.broadcast_to(np.repeat(chemicalHardness[element_ids],3)*(ks), (ks.shape[0], ks.shape[0]))
    eta_ABs = (etas + etas.transpose())*0.5
    include_shell = np.repeat(nShell[element_ids], 3) > np.repeat(np.array([[1,2,3]]),element_ids.shape[0],axis=0).flatten()
    include_shell = np.outer(include_shell, include_shell)
    gamma_ABs = 1./np.sqrt(R_AB2+eta_ABs**(-2))
    energies = np.outer(charges[element_ids].flatten(), charges[element_ids].flatten())*gamma_ABs
    return np.sum(energies*include_shell)*0.5

def density_initial_guess(): #TODO: Make good initial guess
    n = repZeff.shape[0]*3
    return np.ones((n,n))/n

def overlap_initial_guess():
    n = repZeff.shape[0]*3
    return np.eye(n)

density_matrix = density_initial_guess()
overlap_matrix = overlap_initial_guess()
partial_mulliken_charges = get_partial_mulliken_charges(density_matrix, overlap_matrix)
t1 = time.time()
print(isotropic_electrostatic_and_XC_energy_second_order(atoms, partial_mulliken_charges))
t2 = time.time()
print(isotropic_electrostatic_and_XC_energy_second_order_np(element_ids, positions, partial_mulliken_charges))
t3 = time.time()
print("normal:", t2-t1)
print("np:", t3-t2)

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

# energy_type: The type of energy to compute for
# s_uv: overlap of the orbitals. how do we get this?? (To get the coefficients to compute the slater orbites I think we need to compute the zeroth iteration for the wavefunction with a start guess?)
def extended_huckel_energy(atoms, s_uv):
    acc = 0
    for i,(A,_) in enumerate(atoms):
        for j,(B,_) in enumerate(atoms):
            for u in range(nShell[A]):
                for v in range(nShell[B]):
                    # TODO: We need to compute the density matrix P
                    P_uv = 0
                    acc += P_uv * H_EHT(i, j, u, v, atoms, s_uv)
    return acc


def H_EHT(A_idx, B_idx, u, v, atoms, s_uv):
    Kuv_AB = Kll_AB[u][v] 

    A,v1 = atoms[A_idx]
    B,v2 = atoms[B_idx]
    hl_A = selfEnergy[A]
    delta_hl_CNA = kCN[A][u]
    H_uu = hl_A - delta_hl_CNA * GFN2_coordination_number(A_idx, atoms)
    H_vv = hl_A - delta_hl_CNA * GFN2_coordination_number(B_idx, atoms)

    X_electronegativity = 1 if A[0] == B[0] else 1 + 0.02 * (0.35**2)
    R_AB = dist(v1,v2)**2
    k_polyA = shellPoly[A][u]
    k_polyB = shellPoly[B][v]
    Rcov_AB = atomicRadii[A] + atomicRadii[B]
    II = (1 + k_polyA * (R_AB / Rcov_AB)**0.5) * (1 + k_polyB * (R_AB / Rcov_AB)**0.5)
    Y = ((2 * sqrt(slaterExponent[A][u] * slaterExponent[B][v])) / (slaterExponent[A][u] + slaterExponent[B][v]))**0.5

    return 0.5 * Kuv_AB * s_uv * (H_uu + H_vv) * X_electronegativity * II * Y


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



t1 = time.time()
for A_idx,_ in enumerate(atoms):
    GFN2_coordination_number(A_idx, atoms)
t2 = time.time()
GFN2_coordination_numbers_np(element_ids, positions)
t3 = time.time()
print("normal:", t2-t1)
print("np:", t3-t2)
