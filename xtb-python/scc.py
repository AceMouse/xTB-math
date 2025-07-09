from dftd4 import evtoau
import cvxopt

# total scc energy for GFN1
def electro(nbf, H0, P, dq, dqsh, atomicGam, shellGam, jmat, shift):

    es = get_isotropic_electrostatic_energy(dq, dqsh, atomicGam, shellGam, jmat, shift)

    k = 0
    h = 0.0
    for i in range(nbf):
        for j in range(i):
            h += P[i,j] * H0[k]
            k += 1
        h += P[i,i] * H0[k] * 0.5
        k += 1

    scc = es + 2.0 * h * evtoau
    return es, scc

# Coalesced access
# block * block_size + thread
# example: 0 * 256 + 2 = thread 2 of block 0

# idx = block * block_size + thread
# P[block, thread] => P[idx] (Flattened)

# thread * block
# [ ...
#   ...
#   ... ]

# NOTE: Pseudo code for lock step parallel version of electro
#def electro(nbf, H0, P, dq, dqsh, atomicGam, shellGam, jmat, shift):
#    es = get_isotropic_electrostatic_energy(dq, dqsh, atomicGam, shellGam, jmat, shift)
#
#    h_dev = sycl_malloc(number_of_blocks * block_size)
#
#    ######### KERNEL ##########
#    idx = block * blocksize + thread
#
#    if (thread < block):
#        h_dev[idx] = P[idx] * H0[idx]
#
#    if (thread == block):
#        h_dev[idx] = P[idx] * H0[idx] * 0.5
#
#    # barrier
#    reduce(h_dev)
#    ###########################
#
#    # copy hr from device to host
#    h = sycl_move(h_dev)
#
#    scc = es + 2.0 * h * evtoau
#    return es, scc


def get_isotropic_electrostatic_energy(qat, qsh, atomicGam, shellGam, jmat, shift):
    eThird = get_third_order_energy(qat, qsh, atomicGam, shellGam)

    jmat_cvx = cvxopt.matrix(jmat)
    qsh_cvx = cvxopt.matrix(qsh)
    shift_cvx = cvxopt.matrix(shift)

    # y := alpha * Ax + beta * y
    # shift_cvx = 1.0 * jmat_cvx * qsh_cvx + 0.0 * shift_cvx
    cvxopt.blas.symv(jmat_cvx, qsh_cvx, shift_cvx, uplo='L', alpha=1.0, beta=0.0)

    return 0.5 * cvxopt.blas.dot(shift_cvx, qsh_cvx) + eThird

#def symv(A, x, y, uplo='L', alpha=1.0, beta=0.0):
#    y_dev = sycl_malloc(y)
#    A_dev = sycl_malloc(A.flatten)
##    ######### KERNEL ##########
##    idx = block * blocksize + thread
#    alpha * A_dev[idx]
##    ###########################


def get_third_order_energy(qat, qsh, atomicGam, shellGam):
    energy = 0.0

    if (atomicGam is not None):
        for ii in range(qat.shape[0]):
            energy += qat[ii]**3 * atomicGam[ii] / 3.0

    if (shellGam is not None):
        for ii in range(qsh.shape[0]):
            energy += qsh[ii]**3 * shellGam[ii] / 3.0

    return energy
