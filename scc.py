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


def get_isotropic_electrostatic_energy(qat, qsh, atomicGam, shellGam, jmat, shift):
    eThird = get_third_order_energy(qat, qsh, atomicGam, shellGam)

    jmat_cvx = cvxopt.matrix(jmat)
    qsh_cvx = cvxopt.matrix(qsh)
    shift_cvx = cvxopt.matrix(shift)
    cvxopt.blas.symv(jmat_cvx, qsh_cvx, shift_cvx, uplo='L', alpha=1.0, beta=0.0)

    return 0.5 * cvxopt.blas.dot(shift_cvx, qsh_cvx) + eThird


def get_third_order_energy(qat, qsh, atomicGam, shellGam):
    energy = 0.0

    if (atomicGam is not None):
        for ii in range(qat.shape[0]):
            energy += qat[ii]**3 * atomicGam[ii] / 3.0

    if (shellGam is not None):
        for ii in range(qsh.shape[0]):
            energy += qsh[ii]**3 * shellGam[ii] / 3.0

    return energy
