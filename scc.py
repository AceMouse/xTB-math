from dftd4 import evtoau
import cvxopt

# total energy for GFN1
def electro(nbf, H0, P, dq, dqsh, atomicGam, shellGam, jmat, shift):

    es = get_isotropic_electrostatic_energy(dq, dqsh, atomicGam, shellGam, jmat, shift)

    k = 0
    h = 0.0
    for i in range(nbf):
        for j in range(i):
            k += 1
            h += P[i,j] * H0[k]
        k += 1
        h += P[i,i] * H0[k] * 0.5

    scc = es + 2.0 * h * evtoau
    return scc


def get_isotropic_electrostatic_energy(qat, qsh, atomicGam, shellGam, jmat, shift):
    eThird = get_third_order_energy(qat, qsh, atomicGam, shellGam)

    cvxopt.blas.symv(jmat, qsh, shift, uplo='l', alpha=1.0, beta=0.0)

    return 0.5 * cvxopt.blas.dot(shift, qsh) + eThird


def get_third_order_energy(qat, qsh, atomicGam, shellGam):
    energy = 0.0

    if (atomicGam != None):
        for ii in range(qat.shape[0]):
            energy += qat[ii]**3 * atomicGam[ii] / 3.0

    if (shellGam != None):
        for ii in range(qsh.shape[0]):
            energy += qsh[ii]**3 * shellGam[ii] / 3.0

    return energy
