import scipy

def mchrg_dsytrf(amat, uplo):
   #real(dp), intent(inout) :: amat(:, :)
   #integer(ik), intent(out) :: ipiv(:)
   #character(len=1), intent(in), optional :: uplo
   #integer(ik), intent(out), optional :: info

    if (uplo != None):
        ula =  uplo
    else:
        ula = 'u'

    lwork = -1
    _, ipiv, info = scipy.linalg.lapack.dsytrf(amat, ula, lwork=lwork)

    if (info == 0): # NOTE: Why call dsytrf again?
        _, ipiv, info = scipy.linalg.lapack.dsytrf(amat, ula, lwork=lwork)

    return ipiv, info


#import cython
#cimport scipy.linalg.cython_lapack

def mchrg_dsytri(amat, info):
   #real(dp), intent(inout) :: amat(:, :)
   #integer(ik), intent(in) :: ipiv(:)
   #character(len=1), intent(in), optional :: uplo
   #integer(ik), intent(out), optional :: info

    # NOTE: There is no wrapper for the sytri functions in scipy it seems.
    # Instead we use getrf + getri, which is for general matrices instead of for symmetric specifcially?
    # Less efficient, but works?

    lu, piv, info = scipy.linalg.lapack.dgetrf(amat)
    if info != 0:
        raise RuntimeError(f"dgetrf failed: info = {info}")

    inv_amat, info = scipy.linalg.lapack.dgetri(lu, piv)
    #if info != 0:
    #    raise RuntimeError(f"dgetri failed: info = {info}")

    return inv_amat, info

