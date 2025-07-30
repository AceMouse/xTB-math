from math import sqrt
import numpy as np
def dist(v1, v2): #euclidean distance. 
    d = 0
    for a,b in zip(v1,v2):
        d += (a-b)**2
    d = sqrt(d)
    return d

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

def euclidian_dist_sqr(positions):
    pos_sqr = np.broadcast_to(np.sum(positions**2, axis=-1), (positions.shape[0], positions.shape[0]))
    pos_pairs = np.matmul(positions, positions.transpose())
    dist_sqr = pos_sqr-2*pos_pairs+pos_sqr.transpose()
    dist_sqr = dist_sqr * (dist_sqr > 0) # remove sligthly negative values so the sqrt works fine.
    return dist_sqr

def density_initial_guess(element_ids, nShell, angShell):
    occs = []
    for atom_A in element_ids:
        Z = atom_A + 1
        electrons_left = Z
        for j in range(0, nShell[atom_A]):
            l = angShell[atom_A,j] #angular momentum for the shell. 
            orbitals_in_shell = l*2+1 
            max_electrons_in_shell = orbitals_in_shell * 2
            electrons_in_shell = min(max_electrons_in_shell, electrons_left)
            electrons_per_orbital = electrons_in_shell/orbitals_in_shell
            occs += [electrons_per_orbital]*orbitals_in_shell
            electrons_left -= electrons_in_shell
    return np.diag(occs)
#def density_initial_guess(element_cnt): #TODO: Make good initial guess
#    return np.ones((element_cnt*3,element_cnt*3))/element_cnt*3

def overlap_initial_guess(element_cnt):
    return np.eye(element_cnt*3)

def print_res2(x1,x2,t1,t2,t3,label):
    print(f"{label}:")
    print(f"  normal  = {x1:.3f} in {t2-t1:.10f} sec")
    print(f"  numpy   = {x2:.3f} in {t3-t2:.10f} sec")
    print(f"  diff    = {((x1-x2)*100)/x2:.02f}%")
    print(f"  speedup = {((t2-t1))/(t3-t2):.02f}x")

def print_res1(x1,t1,t2,label):
    print(f"{label}:")
    print(f"  normal  = {x1:.3f} in {t2-t1:.10f} sec")

def get_partial_mulliken_charges(density_matrix, overlap_matrix):
    return np.sum(density_matrix*overlap_matrix, axis=-1).reshape(-1,3)

def get_atomic_charges(density_matrix, overlap_matrix, element_ids):
    GAPs = np.sum(get_partial_mulliken_charges(density_matrix, overlap_matrix),axis=-1)
    Zs = element_ids + 1
    return Zs - GAPs 
