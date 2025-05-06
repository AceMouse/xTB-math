from gfn2 import angShell, nShell, repZeff
import numpy as np
import time

from util import euclidian_dist, euclidian_dist_sqr, dist, print_res2, density_initial_guess, overlap_initial_guess, get_partial_mulliken_charges

DIM = True
rand = np.random.default_rng()
element_cnt = 1000
element_ids = rand.choice(repZeff.shape[0], size=element_cnt)
positions = rand.random((element_cnt,3))
atoms = list(zip(element_ids, positions))
density_matrix = density_initial_guess(element_cnt)
overlap_matrix = overlap_initial_guess(element_cnt)

def dim_basis(element_ids):
    n = element_ids.shape[0]
    nao = 0
    nbf = 0
    nshell = 0
    for i in range(0, n):
        k = 0
        ati = element_ids[i]
        for j in range(0, nShell[ati]):
            l = angShell[ati,j] #angular momentum for the shell. 
            k = k + 1
            nshell = nshell + 1
            # from Frank Jensen - Introduction to Computational Chemistry Edition III, right below equation (5.3):
            #   A d-type GTO written in the spherical form has five components (Y_2,2 , Y_2,1 , Y_2,0 , Y_2,−1 , Y_2,−2 ), but there appear to be six components in the Cartesian coordinates 
            #   (x^2 , y^2 , z^2 , xy, xz, yz). The latter six functions, however, may be transformed to the five spherical d-functions and one additional s-function (x^2 + y^2 + z^2 ). 
            #   Similarly, there are ten Cartesian “f-functions” that may be transformed into seven spherical f-functions and one set of spherical p-functions.
            match l:
                case 0: # s-type GTO
                    nbf = nbf + 1 # Cartesian component to describe 1 s-function
                    nao = nao + 1 # spherical component to describe 1 spherical s-function
                case 1: # p-type GTO
                    nbf = nbf + 3 # Cartesian components to describe 3 p-functions
                    nao = nao + 3 # spherical components to describe 3 spherical p-functions
                case 2: # d-type GTO
                    nbf = nbf + 6 # Cartesian components to describe 6 d-functions or possibly 5 d-functions and one s-function
                    nao = nao + 5 # spherical components to describe 5 spherical d-functions
                case 3: # f-type GTO
                    nbf = nbf + 10 # Cartesian components to describe 10 f-functions or possibly 7 f-functions and 3 p-function
                    nao = nao + 7  # spherical components to describe 7 spherical f-functions
                case 4: # g-type GTO
                    nbf = nbf + 15 # Cartesian components to describe 15 g-functions or possibly 9 g-functions and 5 d-functions 
                    nao = nao + 9  # spherical components to describe 9 spherical g-functions
        if k == 0:
             print(f'no basis found for atom {i} Z = {ati}')
             quit(1)
    return nshell, nao, nbf #total number of shells, number of spherical atomic orbitals, number of basis functions. 

def dim_basis_np(element_ids):
    n = element_ids.shape[0]
    shells = nShell[element_ids]
    nshell = np.sum(shells)
    max_shells = np.max(shells)
    js = np.broadcast_to(np.arange(max_shells), (n,max_shells))
    include_shell = js < np.reshape(np.repeat(shells, max_shells), (n, max_shells))
    ls = angShell[np.reshape(np.repeat(element_ids, max_shells), (n, max_shells)), js]
    nbf = np.sum((ls+1)*(ls+2)*include_shell)/2 # triangular numbers of Cartesian components needed to describe basis functions. (1,3,6,10,15)
    nao = np.sum(ls*include_shell)*2 + nshell # odd numbers of spherical components needed to describe basis functions as spherical harmonic functions. (1,3,5,7,9) 
    return nshell, nao, nbf #total number of shells, number of spherical atomic orbitals, number of basis functions. 

if DIM:
    t1 = time.time()
    x1 = np.sum(dim_basis(element_ids))
    t2 = time.time()
    x2 = np.sum(dim_basis_np(element_ids))
    t3 = time.time()
    print_res2(x1,x2,t1,t2,t3,"Sum of dim basis output")
