from math import sqrt, exp, pi
from gfn2 import kExpLight, kExpHeavy, repAlpha, repZeff, nShell, chemicalHardness, shellHardness, thirdOrderAtom, selfEnergy, kCN, shellPoly, slaterExponent, atomicRadii, paulingEN, kEN, angShell, llao, llao2, itt, wExp, kdiff, kScale, pairParam, enScale, enScale4, electronegativity, maxElem, trafo, valenceShell
import numpy as np
import scipy
import time
from fock import huckel_matrix_np, GFN2_coordination_numbers_np
from util import euclidian_dist, euclidian_dist_sqr, dist, print_res2, density_initial_guess, overlap_initial_guess, get_partial_mulliken_charges
H = 0
He = 1
C = 5


REP = False
ISO2 = False
ISO3 = False
EHT = False
CN = False

#element_ids = np.array([C,C,C])
#positions = np.array([[1,0,0],[0,1,0],[0,0,1]])
rand = np.random.default_rng()
element_cnt = 100
element_ids = rand.choice(repZeff.shape[0], size=element_cnt)
positions = rand.random((element_cnt,3))
atoms = list(zip(element_ids, positions))
density_matrix = density_initial_guess(element_cnt)
overlap_matrix = overlap_initial_guess(element_cnt)


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
    print_res2(x1,x2,t1,t2,t3,"repulsion")


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
    print_res2(x1,x2,t1,t2,t3,"isotropic electrostatic and XC energy second order")

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
    partial_mulliken_charges = get_partial_mulliken_charges(density_matrix, overlap_matrix)
    t1 = time.time()
    x1 = isotropic_electrostatic_and_XC_energy_third_order(atoms, partial_mulliken_charges)
    t2 = time.time()
    x2 = isotropic_electrostatic_and_XC_energy_third_order_np(element_ids, positions, partial_mulliken_charges)
    t3 = time.time()
    print_res2(x1*10**6,x2*10**6,t1,t2,t3,"isotropic electrostatic and XC energy third order")

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


def extended_huckel_energy_np(element_ids, positions, density_matrix, overlap_matrix):
    res = density_matrix * huckel_matrix_np(element_ids, positions, overlap_matrix)
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


if CN:
    t1 = time.time()
    x1 = 0
    for A_idx,_ in enumerate(atoms):
        x1 += GFN2_coordination_number(A_idx, atoms)
    t2 = time.time()
    x2 = np.sum(GFN2_coordination_numbers_np(element_ids, positions))
    t3 = time.time()
    print_res2(x1,x2,t1,t2,t3,"coordination numbers")

if EHT:
    t1 = time.time()
    x1 = extended_huckel_energy(atoms, density_matrix, overlap_matrix)
    t2 = time.time()
    x2 = extended_huckel_energy_np(element_ids, positions, density_matrix, overlap_matrix)
    t3 = time.time()
    print_res2(x1,x2,t1,t2,t3,"extended huckel energy")


# nShell[len(element_ids)]: Number of shells for each element
# nat: Number of atoms
# nao: Number of spherical AOs (SAOs) get from dim_basis
# nbf: Number of Basis functions get from dim_basis
# H0: Core hamiltonian
# H0_noovlp: Core Hamiltonian without overlap contribution
# trans = np.zeros((1,3)) is always a vector of 3 zeros since we don't have a 3d infinite periodic boundary condition, i.e. we don't try to simulate an infinite grid of our molecule.  
# intcut = max(20.0, 25.0-10.0*log10(acc)) where acc is the accuracy, a number between 1e-4 and 1e+3 (higher than 3.16 results in intcut = 20.0 though). acc is set with -a (--acc) and defaults to 1.0 resulting in intcut=25.0
# 
# What are the rest of the args?
def build_SDQH0(nat, at, nbf, nao, xyz, trans, selfEnergy, \
       intcut, caoshell, saoshell, nprim, primcount, alp, cont): # TODO: We need to find these arg values
    #H0[:] = 0.0
    #H0_noovlp[:] = 0.0
    H0 = np.zeros(nao*(nao+1)//2)
    H0_noovlp = np.zeros(nao*(nao+1)//2)

    sint = np.zeros((nao, nao), dtype=np.float64)     # overlap matrix S
    dpint = np.zeros((nao, nao, 3), dtype=np.float64) # Dipol moment matrix D
    qpint = np.zeros((nao, nao, 6), dtype=np.float64) # Quadropole moment matrix Q
    point = np.zeros(3)

    il = 0
    jl = 0
    hii = 0.0
    hjj = 0.0
    zi = 0.0
    zj = 0.0
    km = 0.0
    ss = np.zeros((6, 6), dtype=np.float64)    
    dd = np.zeros((3, 6, 6), dtype=np.float64)
    qq = np.zeros((6, 6, 6), dtype=np.float64)
    tmp = np.zeros((6, 6), dtype=np.float64)

    #compute the upper triangle of S, D, Q and H0
    for iat in range(nat):
        for jat in range(nat):
            if (jat >= iat):
                continue

            ra = xyz[iat,0:3]
            izp = at[iat]
            jzp = at[jat]
            for ish in range(nShell[izp]):
                ishtyp = angShell[izp, ish]
                icao = caoshell[iat, ish]
                naoi = llao[ishtyp]
                for jsh in range(nShell[jzp]):
                    jshtyp = angShell[jzp, jsh]
                    jcao = caoshell[jat, jsh]
                    naoj = llao[jshtyp]

                    il = ishtyp
                    jl = jshtyp
                    # diagonals are the same for all H0 elements
                    # H_\kappa\kappa = k_A^l - \delta h_{CN'_A}^l * CN'_A equation used to transform the original selfEnergy array into this. Under equation (1)
                    hii = selfEnergy[iat, ish] 
                    hjj = selfEnergy[jat, jsh]

                    # we scale the two shells depending on their exponent
                    zi = slaterExponent[izp][ish]
                    zj = slaterExponent[jzp][jsh]
                    zetaij = (2 * sqrt(zi*zj)/(zi+zj))**wExp # Y term equation (7) in main.pdf
                    km = h0scal(il, jl, izp, jzp, (valenceShell[izp, ish] != 0), (valenceShell[jzp, jsh] != 0)) # X term, see equation (3-5) and K term. 

                    hav = 0.5 * km * (hii + hjj) * zetaij # equation (1)

                    for itr in range(trans.shape[0]): # NOTE: Is the indexing here correct?
                        rb = xyz[jat, 0:3] + trans[itr, :]
                        rab2 = np.sum((rb - ra)**2)

                        # distance dependent polynomial
                        # equation (6)
                        k_polyA = shellPoly[izp][il]
                        k_polyB = shellPoly[jzp][jl]
                        Rcov_AB = atomicRadii[izp] + atomicRadii[jzp]
                        shpoly = (1.0 + 0.01 * k_polyA * (rab2 / Rcov_AB)**0.5) * (1.0 + 0.01 * k_polyB * (rab2 / Rcov_AB)**0.5)

                        ss, dd, qq = get_multiints(icao,jcao,naoi,naoj,ishtyp,jshtyp,ra,rb,point,intcut,nprim,primcount,alp,cont)

                        # transform from CAO to SAO
                        dtrf2(ss, ishtyp, jshtyp)
                        for k in range(0,3):
                            tmp[0:6, 0:6] = dd[0:6, 0:6, k]
                            dtrf2(tmp, ishtyp, jshtyp)
                            dd[0:6, 0:6, k] = tmp[0:6, 0:6]
                        for k in range(0,6):
                            tmp[0:6, 0:6] = qq[0:6, 0:6, k]
                            dtrf2(tmp, ishtyp, jshtyp)
                            qq[0:6, 0:6, k] = tmp[0:6, 0:6]
                        for ii in range(0, llao2[ishtyp]):
                            iao = ii + saoshell[iat, ish]
                            for jj in range(0, llao2[jshtyp]):
                                jao = jj + saoshell[jat, jsh]
                                ij = lin(iao, jao)
                                H0[ij] = H0[ij] + hav * shpoly * ss[ii, jj] # add in remaining Pi and S terms. 
                                H0_noovlp[ij] = H0_noovlp[ij] + hav * shpoly
                                sint[iao, jao] = sint[iao, jao] + ss[ii, jj]
                                dpint[iao, jao, :] = dpint[iao, jao, :] + dd[ii, jj, :]
                                qpint[iao, jao, :] = qpint[iao, jao, :] + qq[ii, jj, :]

    # mirror the upper triangle to the lower of S, D and Q. H0 does not need it as we used the pairing function to index it. 
    for iao in range(0, nao):
        for jao in range(0, iao-1):
            sint[jao, iao] = sint[iao, jao]
            dpint[jao, iao, :] = dpint[iao, jao, :]
            qpint[jao, iao, :] = qpint[iao, jao, :]

    # compute the diagonal elements of S, D, Q, and H0
    for iat in range(0, nat):
        ra = xyz[iat, :]
        izp = at[iat]
        for ish in range(0, nShell[izp]):
            ishtyp = angShell[izp, ish]
            for iao in range(0, llao2[ishtyp]):
                i = iao + saoshell[iat, ish]
                ii = lin(i, i)  # compute the pairing function. 
                sint[i,i] = 1.0 + sint[i,i]
                H0[ii] = H0[ii] + selfEnergy[iat, ish]
                H0_noovlp[ii] = H0_noovlp[ii] + selfEnergy[iat, ish]

            icao = caoshell[iat, ish]
            naoi = llao[ishtyp]
            for jsh in range(0, ish):
                jshtyp = angShell[izp, jsh]
                jcao = caoshell[iat, jsh]
                naoj = llao[jshtyp]
                ss, dd, qq = get_multiints(icao,jcao,naoi,naoj,ishtyp,jshtyp,ra,ra,point,intcut,nprim,primcount,alp,cont) # compute the integrals

                # transform from CAO to SAO
                for k in range(0, 3):
                    tmp[1:6, 1:6] = dd[1:6, 1:6, k]
                    dtrf2(tmp, ishtyp, jshtyp)
                    dd[1:6, 1:6, k] = tmp[1:6, 1:6]
                for k in range(0, 6):
                    tmp[1:6, 1:6] = qq[1:6, 1:6, k]
                    dtrf2(tmp, ishtyp, jshtyp)
                    qq[1:6, 1:6, k] = tmp[1:6, 1:6]
                for ii in range(0, llao2[ishtyp]):
                    iao = ii + saoshell[iat, ish]
                    for jj in range(0, llao2[jshtyp]):
                        jao = jj + saoshell[iat, jsh]
                        if (jao > iao and ish == jsh):
                            continue
                        dpint[jao, iao, 0:3] = dpint[jao, iao, 0:3] + dd[ii, jj, 0:3]
                        if (iao != jao):
                            dpint[iao, jao, 0:3] = dpint[iao, jao, 0:3] + dd[ii, jj, 0:3]
                        qpint[jao, iao, 0:6] = qpint[jao, iao, 0:6] + qq[ii, jj, 0:6]
                        if (jao != iao):
                            qpint[iao, jao, 0:6] = qpint[iao, jao, 0:6] + qq[ii, jj, 0:6]
    return sint, dpint, qpint, H0, H0_noovlp

# a symmetric version (Cantor) paring function assigning a unique number to each unordered pair of numbers. i.e lin(a,b) = lin(x,y) iff. a = x, b = y or a = y, b = x.
def lin(i1, i2): # TODO: Test? They duplicate this function a bunch of times
    idum1 = max(i1,i2)
    idum2 = min(i1,i2)
    return idum2 + idum1 * (idum1-1)//2


def h0scal(il, jl, izp, jzp, valaoi, valaoj):
    km = 0.0

    # Valence
    if (valaoi and valaoj):
        electronegativity_izp = electronegativity[izp]
        electronegativity_jzp = electronegativity[jzp]
        den = (electronegativity_izp - electronegativity_jzp)**2
        enpoly = (1.0 + enScale[il-1, jl-1] * den * (1.0 + enScale4 * den))
        km = kScale[il-1, jl-1] * enpoly * pairParam[jzp, izp]
        return km

    # "DZ" functions (on H for GFN or 3S for EA calc on all atoms)
    if (not valaoi and not valaoj):
        km = kdiff
        return km
    if (not valaoi and valaoj):
        km = 0.5 * (kScale[jl-1, jl-1] + kdiff)
        return km
    if (not valaoj and valaoi):
        km = 0.5 * (kScale[il-1, il-1] + kdiff)
    return km



# Cartesian components of the angular momentum vector.
lx = np.array([
  0,
  1,0,0,
  2,0,0,1,1,0,
  3,0,0,2,2,1,0,1,0,1,
  4,0,0,3,3,1,0,1,0,2,2,0,2,1,1,
  5,0,0,3,3,2,2,0,0,4,4,1,0,0,1,1,3,1,2,2,1,
  6,0,0,3,3,0,5,5,1,0,0,1,4,4,2,0,2,0,3,3,1,2,2,1,4,1,1,2
])
ly = [
  0,
  0,1,0,
  0,2,0,1,0,1,
  0,3,0,1,0,2,2,0,1,1,
  0,4,0,1,0,3,3,0,1,2,0,2,1,2,1,
  0,5,0,2,0,3,0,3,2,1,0,4,4,1,0,1,1,3,2,1,2,
  0,6,0,3,0,3,1,0,0,1,5,5,2,0,0,2,4,4,2,1,3,1,3,2,1,4,1,2
]
lz = [
  0,
  0,0,1,
  0,0,2,0,1,1,
  0,0,3,0,1,0,1,2,2,1,
  0,0,4,0,1,0,1,3,3,0,2,2,1,1,2,
  0,0,5,0,2,0,3,2,3,0,1,0,1,4,4,3,1,1,1,2,2,
  0,0,6,0,3,3,0,1,5,5,1,0,0,2,4,4,0,2,1,2,2,3,1,3,1,1,4,2
]
lxyz = np.vstack([lx, ly, lz]).T

def get_multiints(icao,jcao,naoi,naoj,ishtyp,jshtyp,ri,rj,point,intcut,nprim,primcount,alp,cont):
    ss = np.zeros((6, 6), dtype=np.float64)
    dd = np.zeros((6, 6, 3), dtype=np.float64)
    qq = np.zeros((6, 6, 6), dtype=np.float64)
    iptyp = itt[ishtyp]
    jptyp = itt[jshtyp]

    rij = ri - rj
    rij2 = rij[0]**2 + rij[1]**2 + rij[2]**2

    max_r2 = 2000.0
    sqrtpi = sqrt(pi)
    t = np.zeros(9)

    if (rij2 > max_r2):
        return ss, dd, qq
    # nprim[icao+1] is the number of primitive functions we expand icao into
    # primcount[icao+1] is the offset into alp for getting the STO exponents and coefficients associated with icao 

    # we go through the primitives (because the screening is the same for all of them)
    for ip in range(nprim[icao]): # NOTE: should this still be icao+1?
        iprim = ip + primcount[icao]
        # exponent the same for each l component
        alpi = alp[iprim]
        for jp in range(nprim[jcao]):
            jprim = jp + primcount[jcao]
            # exponent the same for each l component
            alpj = alp[jprim]
            ab = 1.0 / (alpi + alpj)
            est = alpi * alpj * rij2 * ab
            if (est > intcut):
                continue
            kab = exp(-est) * (sqrtpi * sqrt(ab))**3
            rp = (alpi*ri + alpj*rj) * ab
            for k in range(ishtyp + jshtyp + 3):
                t[k] = olapp([k], alpi+alpj)[0]

            #--------------- compute gradient ----------
            # now compute integrals  for different components of i(e.g., px,py,pz)
            for mli in range(naoi):
                iprim = ip + primcount[icao + mli]
                # coefficients NOT the same (contain CAO2SAO lin. comb. coefficients)
                ci = cont[iprim]
                for mlj in range(naoj):
                    jprim = jp + primcount[jcao + mlj]
                    cj = cont[jprim]
                    cc = kab * cj * ci
                    saw = np.zeros(10)
                    multipole_3d(ri,rj,point,rp,lxyz[iptyp+mli,:],lxyz[jptyp+mlj,:],t,saw)
                    ss[mli,mlj] = ss[mli,mlj] + saw[0] * cc
                    dd[mli,mlj,:] = dd[mli,mlj,:] + saw[1:4]*cc
                    qq[mli,mlj,:] = qq[mli,mlj,:] + saw[4:10]*cc

    return ss,dd,qq


# calculates a partial overlap in one cartesian direction
dftr = [1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0]
def olapp(l, gama):
    s = []
    for lx in l:
        if (lx % 2 != 0):
            s.append(0.0)
            continue

        lh = lx//2
        gm = 0.5 / gama
        s.append(gm**lh*dftr[lh])
    return s

maxl = 6
maxl2 = maxl*2

def multipole_3d(ri, rj, rc, rp, li, lj, s1d, s3d):
    val = np.zeros((3,3))

    for k in range(3):
        vv = np.zeros(maxl2)
        vi = np.zeros(maxl)
        vj = np.zeros(maxl)
        vi[li[k]] = 1.0
        vj[lj[k]] = 1.0
        rpc = rp[k] - rc[k]

        horizontal_shift(rp[k] - ri[k], li[k], vi)
        horizontal_shift(rp[k] - rj[k], lj[k], vj)
        form_product(vi, vj, li[k], lj[k], vv)
        for l in range(li[k] + lj[k] + 1):
            val[0,k] = val[0,k] + s1d[l] * vv[l]
            val[1,k] = val[1,k] + (s1d[l+1] + rpc*s1d[l]) * vv[l]
            val[2,k] = val[2,k] + (s1d[l+2] + 2*rpc*s1d[l+1] + rpc*rpc*s1d[l]) * vv[l]

    s3d[0] = val[0,0] * val[0,1] * val[0,2]
    s3d[1] = val[1,0] * val[0,1] * val[0,2]
    s3d[2] = val[0,0] * val[1,1] * val[0,2]
    s3d[3] = val[0,0] * val[0,1] * val[1,2]
    s3d[4] = val[2,0] * val[0,1] * val[0,2]
    s3d[5] = val[0,0] * val[2,1] * val[0,2]
    s3d[6] = val[0,0] * val[0,1] * val[2,2]
    s3d[7] = val[1,0] * val[1,1] * val[0,2]
    s3d[8] = val[1,0] * val[0,1] * val[1,2]
    s3d[9] = val[0,0] * val[1,1] * val[1,2]

def horizontal_shift(ae, l, cfs):
    match l:
        #case 0: # s
        #    pass
        case 1: # p
            cfs[0] = cfs[0] + ae * cfs[1]
        case 2: # d
            cfs[0] = cfs[0] + ae * ae * cfs[2]
            cfs[1] = cfs[1] + 2 * ae * cfs[2]
        case 3: # f
            cfs[0] = cfs[0] + ae * ae * ae * cfs[3]
            cfs[1] = cfs[1] + 3 * ae * ae * cfs[3]
            cfs[2] = cfs[2] + 3 * ae * cfs[3]
        case 4: # g
            cfs[0] = cfs[0] + ae * ae * ae * ae * cfs[4]
            cfs[1] = cfs[1] + 4 * ae * ae * ae * cfs[4]
            cfs[2] = cfs[2] + 6 * ae * ae * cfs[4]
            cfs[3] = cfs[3] + 4 * ae * cfs[4]

def form_product(a, b, la, lb, d):
    if (la > 4 or lb > 4):
        # <s|g> = <s|*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g>
        d[0] = a[0] * b[0]
        d[1] = a[0] * b[1] + a[1] * b[0]
        d[2] = a[0] * b[2] + a[2] * b[0]
        d[3] = a[0] * b[3] + a[3] * b[0]
        d[4] = a[0] * b[4] + a[4] * b[0]
        if (la == 0 or lb == 0):
            return
        
        # <p|g> = (<s|+<p|)*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g> + <h>
        d[2] = d[2]+a[1]*b[1]
        d[3] = d[3]+a[1]*b[2]+a[2]*b[1]
        d[4] = d[4]+a[1]*b[3]+a[3]*b[1]
        d[5] = a[1]*b[4]+a[4]*b[1]
        if(la <= 1 or lb <= 1):
            return

        # <d|g> = (<s|+<p|+<d|)*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g> + <h> + <i>
        d[4] = d[4]+a[2]*b[2]
        d[5] = d[4]+a[2]*b[3]+a[3]*b[2]
        d[6] = a[2]*b[4]+a[4]*b[2]
        if(la <= 2 or lb <= 2):
            return

        # <f|g> = (<s|+<p|+<d|+<f|)*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g> + <h> + <i> + <k>
        d[6] = d[6]+a[3]*b[3]
        d[7] = a[3]*b[4]+a[4]*b[3]
        if(la <= 3 or lb <= 3):
            return

        # <g|g> = (<s|+<p|+<d|+<f|+<g|)*(|s>+|p>+|d>+|f>+|g>)
        #       = <s> + <p> + <d> + <f> + <g> + <h> + <i> + <k> + <l>
        d[8] = a[4]*b[4]
        return


    if (la >= 3 or lb >= 3):
        # <s|f> = <s|*(|s>+|p>+|d>+|f>)
        #       = <s> + <p> + <d> + <f>
        d[0] = a[0]*b[0]
        d[1] = a[0]*b[1]+a[1]*b[0]
        d[2] = a[0]*b[2]+a[2]*b[0]
        d[3] = a[0]*b[3]+a[3]*b[0]
        if (la == 0 or lb == 0):
            return
        # <p|f> = (<s|+<p|)*(|s>+|p>+|d>+|f>)
        #       = <s> + <p> + <d> + <f> + <g>
        d[2] = d[2]+a[1]*b[1]
        d[3] = d[3]+a[1]*b[2]+a[2]*b[1]
        d[4] = a[1]*b[3]+a[3]*b[1]
        if (la <= 1 or lb <= 1):
            return
        # <d|f>  =  (<s|+<p|+<d|)*(|s>+|p>+|d>+|f>)
        #        =  <s> + <p> + <d> + <f> + <g> + <h>
        d[4] = d[4]+a[2]*b[2]
        d[5] = a[2]*b[3]+a[3]*b[2]
        if(la <= 2 or lb <= 2):
            return
        # <f|f>  =  (<s|+<p|+<d|+<f|)*(|s>+|p>+|d>+|f>)
        #        =  <s> + <p> + <d> + <f> + <g> + <h> + <i>
        d[6] = a[3]*b[3]
        return

    if (la >= 2 or lb >= 2):
        # <s|d> = <s|*(|s>+|p>+|d>)
        #       = <s> + <p> + <d>
        d[0] = a[0]*b[0]
        d[1] = a[0]*b[1]+a[1]*b[0]
        d[2] = a[0]*b[2]+a[2]*b[0]
        if(la == 0 or lb == 0):
            return

        # <p|d> = (<s|+<p|)*(|s>+|p>+|d>)
        #       = <s> + <p> + <d> + <f>
        d[2] = d[2]+a[1]*b[1]
        d[3] = a[1]*b[2]+a[2]*b[1]
        if(la <= 1 or lb <= 1):
            return

        # <d|d> = (<s|+<p|+<d|)*(|s>+|p>+|d>)
        #       = <s> + <p> + <d> + <f> + <g>
        d[4] = a[2]*b[2]
        return


    # <s|s> = <s>
    d[0] = a[0]*b[0]
    if (la == 0 and lb == 0):
        return

    # <s|p> = <s|*(|s>+|p>)
    #       = <s> + <p>
    d[1] = a[0]*b[1]+a[1]*b[0]
    if (la == 0 or lb == 0):
        return

    # <p|p> = (<s|+<p|)*(|s>+|p>)
    #       = <s> + <p> + <d>
    d[2] = a[1]*b[1]


def dtrf2(s,li,lj):
    # transformation not needed for pure s/p overlap -> do nothing
    if (li < 2 and lj < 2):
        return

    s2 = np.zeros((6,6))

    # At this point li >= 2 or lj >= 2, so one of them is a d-shell
    # assuming its on jat ... a wild guess
    match li:
        case 0: # s-d
            for jj in range(6):
                sspher = 0
                for m in range(6):
                    sspher = sspher + trafo[jj,m] * s[0,m]
                s2[0,jj] = sspher
            s[0,0:5] = s2[0,1:6]
            return
        case 1: # p-d
            for ii in range(3):
                for jj in range(6):
                    sspher = 0
                    for m in range(6):
                        sspher = sspher + trafo[jj,m] * s[ii,m]
                    s2[ii,jj] = sspher
                s[ii,0:5] = s2[ii, 1:6]
            return

    # wasn't there, then try iat ...
    match lj:
        case 0: # d-s
            for jj in range(6):
                sspher = 0
                for m in range(6):
                    sspher = sspher + trafo[jj,m] * s[m,0]
                s2[jj,0] = sspher
            s[0:5, 0] = s2[1:6, 0]
            return
        case 1: # d-p
            for ii in range(3):
                for jj in range(6):
                    sspher = 0
                    for m in range(6):
                        sspher = sspher + trafo[jj,m] * s[m,ii]
                    s2[jj,ii] = sspher
                s[0:5,ii] = s2[1:6,ii]
            return

    # if not returned up to here -> d-d
    # CB: transposing s in first dgemm is important for integrals other than S
    dum = np.zeros((6,6))
    dum = mctc_dgemm(trafo, s, dum, transa=True)
    s2 = mctc_dgemm(dum, trafo, s2, transb=False)
    s[0:5,0:5] = s2[1:6,1:6]

def mctc_dgemm(amat, bmat, cmat, transa=False, transb=False, alpha=1.0, beta=0.0):
    return scipy.linalg.blas.dgemm(alpha, amat, bmat, beta, cmat, transa, transb)
#subroutine getSelfEnergy2D(hData, nShell, at, cn, qat, selfEnergy, dSEdcn, dSEdq)
#   type(THamiltonianData), intent(in) :: hData
#   integer, intent(in) :: nShell(:)
#   integer, intent(in) :: at(:)
#   real(wp), intent(in), optional :: cn(:)
#   real(wp), intent(in), optional :: qat(:)
#   real(wp), intent(out) :: selfEnergy(:, :)
#   real(wp), intent(out), optional :: dSEdcn(:, :)
#   real(wp), intent(out), optional :: dSEdq(:, :)
#
#   integer :: iAt, iZp, iSh
#
#   selfEnergy(:, :) = 0.0_wp
#   if (present(dSEdcn)) dSEdcn(:, :) = 0.0_wp
#   if (present(dSEdq)) dSEdq(:, :) = 0.0_wp
#   do iAt = 1, size(cn)
#      iZp = at(iAt)
#      do iSh = 1, nShell(iZp)
#         selfEnergy(iSh, iAt) = hData%selfEnergy(iSh, iZp)
#      end do
#   end do
#   if (present(dSEdcn) .and. present(cn)) then
#      do iAt = 1, size(cn)
#         iZp = at(iAt)
#         do iSh = 1, nShell(iZp)
#            selfEnergy(iSh, iAt) = selfEnergy(iSh, iAt) &
#               & - hData%kCN(iSh, iZp) * cn(iAt)
#            dSEdcn(iSh, iAt) = -hData%kCN(iSh, iZp)
#         end do
#      end do
#   end if
#   if (present(dSEdq) .and. present(qat)) then
#      do iAt = 1, size(cn)
#         iZp = at(iAt)
#         do iSh = 1, nShell(iZp)
#            selfEnergy(iSh, iAt) = selfEnergy(iSh, iAt) &
#               & - hData%kQShell(iSh,iZp)*qat(iAt) - hData%kQAtom(iZp)*qat(iAt)**2
#            dSEdq(iSh, iAt) = -hData%kQShell(iSh,iZp) &
#               & - hData%kQAtom(iZp)*2*qat(iAt)
#         end do
#      end do
#   end if
#
#end subroutine getSelfEnergy2D
def getSelfEnergy(element_ids, cn): # actually the H_\kappa\kappa diagonal terms from the GFN2 Hamiltonian. Weird naming it selfEnergy. 
    selfEnergyCopy = np.zeros((element_ids.shape[0], np.max(nShell)))
    for iAt in range(element_ids.shape[0]):
        iZp = element_ids[iAt]
        for iSh in range(nShell[iZp]):
            selfEnergyCopy[iAt, iSh] = selfEnergy[iZp, iSh]
    for iAt in range(element_ids.shape[0]):
        iZp = element_ids[iAt]
        for iSh in range(nShell[iZp]):
            selfEnergyCopy[iAt, iSh] -= kCN[iZp,iSh]*cn[iAt] 
    return selfEnergyCopy



BUILD = True
if BUILD:
    trans = np.zeros((1,3))
    from basisset import new_basis_set_simple, dim_basis_np
    _, basis_nao, basis_nbf = dim_basis_np(element_ids)
    basis_shells, basis_sh2ao, basis_sh2bf, basis_minalp, basis_level, basis_zeta, basis_valsh, basis_hdiag, basis_alp, basis_cont, basis_hdiag2, basis_aoexp, basis_ash, basis_lsh, basis_ao2sh, basis_nprim, basis_primcount, basis_caoshell, basis_saoshell, basis_fila, basis_fila2, basis_lao, basis_aoat, basis_valao, basis_lao2, basis_aoat2, basis_valao2, ok = new_basis_set_simple(element_ids)
    acc = 1.0
    intcut = max(20.0, 25.0-10.0*np.log10(acc))
    cn = GFN2_coordination_numbers_np(element_ids, positions)
    # TODO compare with 'cn' from the fortran code
    # Check the gfn path here. gfn = cn
    # https://github.com/grimme-lab/xtb/blob/09288659551cde6d98e4514bc23892ec0dbee075/src/disp/coordinationnumber.f90#L339
    print(cn.shape)
    selfEnergy_H_kappa_kappa = getSelfEnergy(element_ids, cn)
    S, dpint, qpint, H0, H0_noovlp = build_SDQH0(element_cnt, element_ids, \
      basis_nbf, basis_nao, positions, trans, selfEnergy_H_kappa_kappa, intcut, \
      basis_caoshell, basis_saoshell, basis_nprim, basis_primcount, basis_alp, \
      basis_cont)
    
