import cvxopt
import scipy

def mchrg_dsymv(amat, xvec, yvec, uplo, alpha, beta):
   #real(dp), intent(in) :: amat(:, :)
   #real(dp), intent(in) :: xvec(:)
   #real(dp), intent(inout) :: yvec(:)
   #character(len=1), intent(in), optional :: uplo
   #real(dp), intent(in), optional :: alpha
   #real(dp), intent(in), optional :: beta

    if (alpha != None):
        a = alpha
    else:
        a = 1.0

    if (beta != None):
        b = beta
    else:
        b = 0

    if (uplo != None):
        ula = uplo
    else:
        ula = 'u'

    #incx = 1
    #incy = 1
    #lda = max(1, amat.shape[0])
    #n = amat.shape[1]
    cvxopt.blas.symv(amat, xvec, yvec, uplo=ula, alpha=a, beta=b)


def mchrg_dsytrs1(amat, bvec, ipiv, uplo, info):
   #real(dp), intent(in) :: amat(:, :)
   #real(dp), intent(inout), target :: bvec(:)
   #integer(ik), intent(in) :: ipiv(:)
   #character(len=1), intent(in), optional :: uplo
   #integer(ik), intent(out), optional :: info

    bptr = bvec.reshape(-1, 1)
    mchrg_dsytrs(amat, bptr, ipiv, uplo, info)

def mchrg_dsytrs(amat, bmat, ipiv, uplo, info):
   #real(dp), intent(in) :: amat(:, :)
   #real(dp), intent(inout) :: bmat(:, :)
   #integer(ik), intent(in) :: ipiv(:)
   #character(len=1), intent(in), optional :: uplo
   #integer(ik), intent(out), optional :: info

    #if (uplo != None):
    #    ula = uplo
    #else:
    #    ula = 'u'

    #lda = max(1, amat.shape[0])
    #ldb = max(1, bmat.shape[0])
    #n = amat.shape[1]
    #nrhs = bmat.shape[1]
    scipy.linalg.lapack.dsytrs(amat, ipiv, bmat)

