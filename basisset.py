from gfn2 import angShell, nShell, repZeff, valenceShell, slaterExponent, selfEnergy, principalQuantumNumber, numberOfPrimitives
import numpy as np
import time

from slater import slaterToGauss_simple as slaterToGauss
from util import euclidian_dist, euclidian_dist_sqr, dist, print_res2, density_initial_guess, overlap_initial_guess, get_partial_mulliken_charges

DIM = True
rand = np.random.default_rng()
element_cnt = 10
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

#subroutine atovlp(l,npri,nprj,alpa,alpb,conta,contb,ss)
#   integer l,npri,nprj
#   real(wp) alpa(*),alpb(*)
#   real(wp) conta(*),contb(*)
#   real(wp) ss
#
#   integer ii,jj
#   real(wp) ab,s00,sss,ab05
#
#   SS=0.0_wp
#   do ii=1,npri
#      do jj=1,nprj
#         ab =1./(alpa(ii)+alpb(jj))
#         s00=(pi*ab)**1.50_wp
#         if(l.eq.0)then
#            sss=s00
#         endif
#         if(l.eq.1)then
#            ab05=ab*0.5_wp
#            sss=s00*ab05
#         endif
#         SS=SS+SSS*conta(ii)*contb(jj)
#      enddo
#   enddo
#
#end subroutine atovlp

def atovlp(l, npri, nprj, alpa, alpb, conta, contb):
    ss = 0.0
    sss = 0.0
    for ii in range(0,npri):
        for jj in range(0,nprj):
            ab = 1./(alpa[ii]+alpb[jj])
            s00 = (np.pi*ab)**1.5
            if l == 0:
                sss = s00
            if l == 1:
                ab05 = ab*0.5
                sss = s00*ab05
            ss += sss*conta[ii]*contb[jj]
    return ss

def atovlp_np(l,npri,nprj, alpa, alpb, conta, contb):
    alpa_rect = np.broadcast_to(alpa, (npri, nprj))
    alpb_rect = np.broadcast_to(alpa, (nprj, npri))
    ab = 1./(alpa_rect + alpb_rect.transpose())
    sss = (np.pi*ab)**1.5
    if l == 1:
        sss *= ab*0.5
    return np.sum(sss*np.outer(conta,contb))
#subroutine set_d_function(basis,iat,ish,iao,ibf,ipr,npq,l,nprim,zeta,level,valao)
#   type(TBasisset), intent(inout) :: basis
#   integer, intent(in)    :: iat
#   integer, intent(in)    :: ish
#   integer, intent(inout) :: ibf
#   integer, intent(inout) :: iao
#   integer, intent(inout) :: ipr
#   integer, intent(in)    :: npq
#   integer, intent(in)    :: l
#   integer, intent(in)    :: nprim
#   integer, intent(in)    :: valao
#   real(wp),intent(in)    :: zeta
#   real(wp),intent(in)    :: level
#   integer  :: j,p
#   real(wp) :: alp(10),cont(10)
#   real(wp) :: trafo(5:10) = &
#      & [1.0_wp, 1.0_wp, 1.0_wp, sqrt(3.0_wp), sqrt(3.0_wp), sqrt(3.0_wp)]
#   integer :: info
#
#   call slaterToGauss(nprim, npq, l, zeta, alp, cont, .true., info)
#   basis%minalp(ish) = minval(alp(:nprim))
#
#   do j = 5, 10
#
#      ibf = ibf+1
#      basis%primcount(ibf) = ipr
#      basis%valao    (ibf) = valao
#      basis%aoat     (ibf) = iat
#      basis%lao      (ibf) = j
#      basis%nprim    (ibf) = nprim
#      basis%hdiag    (ibf) = level
#
#      do p=1,nprim
#         ipr = ipr+1
#         basis%alp (ipr)=alp (p)
#         basis%cont(ipr)=cont(p)*trafo(j)
#      enddo
#
#      if (j .eq. 5) cycle
#
#      iao = iao+1
#      basis%valao2(iao) = valao
#      basis%aoat2 (iao) = iat
#      basis%lao2  (iao) = j-1
#      basis%hdiag2(iao) = level
#      basis%aoexp (iao) = zeta
#      basis%ao2sh (iao) = ish
#
#   enddo
#end subroutine set_d_function
def set_d_functoin(basis,iat,ish,iao,ibf,ipr,npq,l,nprim,zeta,level,valao):
    pass

#
#subroutine set_f_function(basis,iat,ish,iao,ibf,ipr,npq,l,nprim,zeta,level,valao)
#   type(TBasisset), intent(inout) :: basis
#   integer, intent(in)    :: iat
#   integer, intent(in)    :: ish
#   integer, intent(inout) :: ibf
#   integer, intent(inout) :: iao
#   integer, intent(inout) :: ipr
#   integer, intent(in)    :: npq
#   integer, intent(in)    :: l
#   integer, intent(in)    :: nprim
#   integer, intent(in)    :: valao
#   real(wp),intent(in)    :: zeta
#   real(wp),intent(in)    :: level
#   integer  :: j,p
#   real(wp) :: alp(10),cont(10)
#   real(wp) :: trafo(11:20) = &
#      & [1.0_wp, 1.0_wp, 1.0_wp, sqrt(5.0_wp), sqrt(5.0_wp), &
#      &  sqrt(5.0_wp), sqrt(5.0_wp), sqrt(5.0_wp), sqrt(5.0_wp), sqrt(15.0_wp)]
#   integer :: info
#
#   call slaterToGauss(nprim, npq, l, zeta, alp, cont, .true., info)
#   basis%minalp(ish) = minval(alp(:nprim))
#
#   do j = 11, 20
#
#      ibf = ibf+1
#      basis%primcount(ibf) = ipr
#      basis%valao    (ibf) = valao
#      basis%aoat     (ibf) = iat
#      basis%lao      (ibf) = j
#      basis%nprim    (ibf) = nprim
#      basis%hdiag    (ibf) = level
#
#      do p=1,nprim
#         ipr = ipr+1
#         basis%alp (ipr)=alp (p)
#         basis%cont(ipr)=cont(p)*trafo(j)
#      enddo
#
#      if (j.ge.11 .and. j.le.13) cycle
#
#      iao = iao+1
#      basis%valao2(iao) = valao
#      basis%aoat2 (iao) = iat
#      basis%lao2  (iao) = j-3
#      basis%hdiag2(iao) = level
#      basis%aoexp (iao) = zeta
#      basis%ao2sh (iao) = ish
#
#   enddo
#end subroutine set_f_function
#subroutine newBasisset(xtbData,n,at,basis,ok)

def new_basis_set_simple(element_ids):
    a = np.zeros(10)
    c = np.zeros(10)
    aR = np.zeros(10)
    cR = np.zeros(10)
    aS = np.zeros(10)
    cS = np.zeros(10)

    n = element_ids.shape[0]
    nshell, nao, nbf = dim_basis(element_ids)
    basis_shells = np.zeros((nshell,2))
    basis_sh2ao = np.zeros((nshell,2))
    basis_sh2bf = np.zeros((nshell,2))
    basis_minalp = np.zeros(nshell)
    basis_level = np.zeros(nshell)
    basis_zeta = np.zeros(nshell)
    basis_valsh = np.zeros(nshell)
    basis_hdiag = np.zeros(nbf)
    basis_alp = np.zeros(9*nbf)
    basis_cont = np.zeros(9*nbf)
    basis_hdiag2 = np.zeros(nao)
    basis_aoexp = np.zeros(nao)
    basis_ash = np.zeros(nao)
    basis_lsh = np.zeros(nao)
    basis_ao2sh = np.zeros(nao)
    basis_nprim = np.zeros(nbf, dtype = np.int32)
    basis_primcount = np.zeros(nbf, dtype = np.int32)
    basis_caoshell = np.zeros((n,5), dtype = np.int32)
    basis_saoshell = np.zeros((n,5), dtype = np.int32)
    basis_fila = np.zeros((n,2))
    basis_fila2 = np.zeros((n,2))
    basis_lao = np.zeros(nbf)
    basis_aoat = np.zeros(nbf)
    basis_valao = np.zeros(nbf)
    basis_lao2 = np.zeros(nao)
    basis_aoat2 = np.zeros(nao)
    basis_valao2 = np.zeros(nao)
    basis_hdiag[:] = 1e42
    ibf=0
    iao=0
    ipr=0
    ish=0
    ok=True
    for iat in range(0,n):
        ati = element_ids[iat]
        basis_shells[iat,0] = ish+1 # TODO: should this just be no addition? Is this to fix 1 indexing? 
        basis_fila[iat,0] = ibf+1
        basis_fila2[iat,0] = iao+1
        for m in range(nShell[ati]):
            npq = principalQuantumNumber[ati,m]
            if npq <= 0:
                quit(1)
            l = angShell[ati,m]
            level = selfEnergy[ati,m]
            zeta = slaterExponent[ati,m]
            valao = valenceShell[ati,m]
            if valao != 0:
                nprim = numberOfPrimitives[ati,m]
            else:
                thisprimR = numberOfPrimitives[ati,m]
            basis_lsh[ish] = l
            basis_ash[ish] = iat
            basis_sh2bf[ish,0] = ibf
            basis_sh2ao[ish,0] = iao
            basis_caoshell[iat,m] = ibf
            basis_saoshell[iat,m] = iao
            basis_level[ish] = level
            basis_zeta[ish] = zeta
            basis_valsh[ish] = valao
            
            valao_flip = 1
            j_offset = 0
            trafo = np.array([1,1,1,1,1,1,1,np.sqrt(3.),np.sqrt(3.),np.sqrt(3.), # combined for d and f
                              1.0, 1.0, 1.0, np.sqrt(5.0), np.sqrt(5.0), np.sqrt(5.0), np.sqrt(5.0), np.sqrt(5.0), np.sqrt(5.0), np.sqrt(15.0)],dtype = np.float64)
            additional_prim = 0
            ss=-1
            normalize_basis_cont = False
            match l:
                case 0: # s
                    if valao != 0: 
                        if ati < 2: # H-He 
                            a, c, info = slaterToGauss(nprim, npq, l, zeta, True)
                            basis_minalp[ish] = np.min(a[:nprim])
                        else: 
                            aS, cS, info = slaterToGauss(nprim, npq, l, zeta, True)
                            basis_minalp[ish] = np.min(aS[:nprim])
                    else: # DZ s
                        additional_prim = thisprimR
                        normalize_basis_cont = True
                        aR, cR, info = slaterToGauss(thisprimR, npq, l, zeta, True)
                        if ati < 2: # H-He
                            ss = atovlp(0,nprim,thisprimR,a,aR,c,cR)
                            min_alpa = np.min(a[:nprim])
                        else:
                            ss = atovlp(0,nprim,thisprimR,aS,aR,cS,cR)
                            min_alpa = np.min(aS[:nprim])
                        basis_minalp[ish] = min(min_alpa, np.min(aR[:thisprimR]))
                    j_low = 0
                    j_high = 1
                case 1: # p
                    a, c, info = slaterToGauss(nprim, npq, l, zeta, True)
                    basis_minalp[ish] = np.min(a[:nprim])
                    if ati < 2: # H-He
                        valao_flip = -1
                    j_low = 1
                    j_high = 4
                case 2: # d
                    a, c, info = slaterToGauss(nprim, npq, l, zeta, True)
                    basis_minalp[ish] = np.min(a[:nprim])
                    j_low = 4
                    j_high = 10
                    j_offset = -1
                case 3: # f
                    a, c, info = slaterToGauss(nprim, npq, l, zeta, True)
                    basis_minalp[ish] = np.min(a[:nprim])
                    j_low = 10
                    j_high = 20
                    j_offset = -3
                    valao = 1
            for j in range(j_low,j_high):
                basis_primcount[ibf] = ipr
                basis_valao    [ibf] = valao*valao_flip
                basis_aoat     [ibf] = iat
                basis_lao      [ibf] = j
                basis_nprim    [ibf] = nprim + additional_prim
                basis_hdiag    [ibf] = level

                idum=ipr
                for p in range(0,additional_prim):
                    basis_alp[ipr] = aR[p]
                    basis_cont[ipr] = cR[p]
                    ipr+=1

                for p in range(0,nprim):
                    if l == 0 and ati >= 2:
                        basis_alp[ipr]=aS[p]
                        basis_cont[ipr]=-ss*cS[p]
                    else:
                        basis_alp[ipr]=a[p]
                        basis_cont[ipr]=-ss*c[p]*trafo[j]
                    ipr += 1

                if normalize_basis_cont:
                    ss = atovlp(0,basis_nprim[ibf],basis_nprim[ibf],basis_alp[idum],basis_alp[idum],basis_cont[idum],basis_cont[idum])
                    for p in range(0,basis_nprim[ibf]):
                        basis_cont[idum+p]/=np.sqrt(ss)
                ibf += 1

                if (j >= 10 and j <= 12) or j == 4:
                    continue

                basis_valao2[iao] = valao*valao_flip
                basis_aoat2[iao] = iat
                basis_lao2[iao] = j+j_offset
                basis_hdiag2[iao] = level
                basis_aoexp[iao] = zeta
                basis_ao2sh[iao] = ish

                iao += 1
            basis_sh2bf[ish,1] = ibf-basis_sh2bf[ish,0]
            basis_sh2ao[ish,1] = iao-basis_sh2ao[ish,0]
            ish += 1
        basis_shells[iat,1]=ish
        basis_fila[iat,1]=ibf
        basis_fila2[iat,1]=iao
    ok = np.all(basis_alp[:ipr] > 0) and nbf == ibf and nao == iao
    return basis_shells, basis_sh2ao, basis_sh2bf, basis_minalp, basis_level, basis_zeta, basis_valsh,\
        basis_hdiag, basis_alp, basis_cont, basis_hdiag2, basis_aoexp, basis_ash, basis_lsh, basis_ao2sh, \
        basis_nprim, basis_primcount, basis_caoshell, basis_saoshell, basis_fila, basis_fila2, basis_lao, \
        basis_aoat, basis_valao, basis_lao2, basis_aoat2, basis_valao2, ok


def new_basis_set(element_ids):
#   type(TxTBData), intent(in) :: xtbData
#   type(TBasisset),intent(inout) :: basis
#   integer, intent(in)  :: n
#   integer, intent(in)  :: at(n)
#   logical, intent(out) :: ok
#
#   integer  :: elem,valao
#   integer  :: i,j,m,l,iat,ati,ish,ibf,iao,ipr,p,nprim,thisprimR,idum,npq,npqR,pqn
#   real(wp) :: a(10),c(10),zeta,k1,k2,split1,pp,zqfR,zcnfR,qi,level
#   real(wp) :: aR(10),cR(10),ss
#   real(wp) :: as(10),cs(10)
#   integer :: info
    a = np.zeros(10)
    c = np.zeros(10)
    aR = np.zeros(10)
    cR = np.zeros(10)
    aS = np.zeros(10)
    cS = np.zeros(10)

#  call xbasis0(xtbData,n,at,basis)
    n = element_ids.shape[0]
    nshell, nao, nbf = dim_basis(element_ids)
    basis_shells = np.zeros((nshell,2))
    basis_sh2ao = np.zeros((nshell,2))
    basis_sh2bf = np.zeros((nshell,2))
    basis_minalp = np.zeros(nshell)
    basis_level = np.zeros(nshell)
    basis_zeta = np.zeros(nshell)
    basis_valsh = np.zeros(nshell)
    basis_hdiag = np.zeros(nbf)
    basis_alp = np.zeros(9*nbf)
    basis_cont = np.zeros(9*nbf)
    basis_hdiag2 = np.zeros(nao)
    basis_aoexp = np.zeros(nao)
    basis_ash = np.zeros(nao)
    basis_lsh = np.zeros(nao)
    basis_ao2sh = np.zeros(nao)
    basis_nprim = np.zeros(nbf, dtype = np.int32)
    basis_primcount = np.zeros(nbf, dtype = np.int32)
    basis_caoshell = np.zeros((n,5), dtype = np.int32)
    basis_saoshell = np.zeros((n,5), dtype = np.int32)
    basis_fila = np.zeros((n,2))
    basis_fila2 = np.zeros((n,2))
    basis_lao = np.zeros(nbf)
    basis_aoat = np.zeros(nbf)
    basis_valao = np.zeros(nbf)
    basis_lao2 = np.zeros(nao)
    basis_aoat2 = np.zeros(nao)
    basis_valao2 = np.zeros(nao)
#
#   basis%hdiag(1:basis%nbf)=1.d+42
    basis_hdiag[:] = 1e42
#
#   ibf=0
#   iao=0
#   ipr=0
#   ish=0
#   ok=.true.
    ibf=0
    iao=0
    ipr=0
    ish=0
    ok=True
#
#   atoms: do iat=1,n
    for iat in range(0,n):
#      ati = at(iat)
        ati = element_ids[iat]
#      basis%shells(1,iat)=ish+1
        basis_shells[iat,0] = ish+1 
#      basis%fila  (1,iat)=ibf+1
        basis_fila[iat,0] = ibf+1
#      basis%fila2 (1,iat)=iao+1
        basis_fila2[iat,0] = iao+1
#      shells: do m=1,xtbData%nShell(ati)
        for m in range(nShell[ati]):
#         ish = ish+1
#         ! principle QN
#         npq=xtbData%hamiltonian%principalQuantumNumber(m,ati)
            npq = principalQuantumNumber[ati,m]
#         l=xtbData%hamiltonian%angShell(m,ati)
            l = angShell[ati,m]
#
#         level = xtbData%hamiltonian%selfEnergy(m,ati)
            level = selfEnergy[ati,m]
#         zeta  = xtbData%hamiltonian%slaterExponent(m,ati)
            zeta = slaterExponent[ati,m]
#         valao = xtbData%hamiltonian%valenceShell(m,ati)
            valao = valenceShell[ati,m]
#         if (valao /= 0) then
            if (valao != 0) :
#            nprim = xtbData%hamiltonian%numberOfPrimitives(m,ati)
                nprim = numberOfPrimitives[ati,m]
#         else
            else:
#            thisprimR = xtbData%hamiltonian%numberOfPrimitives(m,ati)
                thisprimR = numberOfPrimitives[ati,m]
#         end if
#
#         basis%lsh(ish) = l
#         basis%ash(ish) = iat
#         basis%sh2bf(1,ish) = ibf
#         basis%sh2ao(1,ish) = iao
#         basis%caoshell(m,iat)=ibf
#         basis%saoshell(m,iat)=iao
            basis_lsh[ish] = l
            basis_ash[ish] = iat
            basis_sh2bf[ish,0] = ibf
            basis_sh2ao[ish,0] = iao
            basis_caoshell[iat,m] = ibf
            basis_saoshell[iat,m] = iao
#
#         ! add new shellwise information, for easier reference
#         basis%level(ish) = level
#         basis%zeta (ish) = zeta
#         basis%valsh(ish) = valao
            basis_level[ish] = level
            basis_zeta[ish] = zeta
            basis_valsh[ish] = valao
#
#         ! H-He
#         if(l.eq.0.and.ati.le.2.and.valao/=0)then
#            ! s
#            call slaterToGauss(nprim, npq, l, zeta, a, c, .true., info)
#            basis%minalp(ish) = minval(a(:nprim))
#
#            ibf =ibf+1
#            basis%primcount(ibf) = ipr
#            basis%valao    (ibf) = valao
#            basis%aoat     (ibf) = iat
#            basis%lao      (ibf) = 1
#            basis%nprim    (ibf) = nprim
#            basis%hdiag    (ibf) = level
#
#            do p=1,nprim
#               ipr=ipr+1
#               basis%alp (ipr)=a(p)
#               basis%cont(ipr)=c(p)
#            enddo
#
#            iao = iao+1
#            basis%valao2(iao) = valao
#            basis%aoat2 (iao) = iat
#            basis%lao2  (iao) = 1
#            basis%hdiag2(iao) = level
#            basis%aoexp (iao) = zeta
#            basis%ao2sh (iao) = ish
#         endif
            if ( l == 0 and ati < 2 and valao != 0):
                a, c, info = slaterToGauss(nprim, npq, l, zeta, True)
                basis_minalp[ish] = np.min(a[:nprim])

                basis_primcount[ibf] = ipr
                basis_valao[ibf] = valao
                basis_aoat[ibf] = iat
                basis_lao[ibf] = 1
                basis_nprim[ibf] = nprim
                basis_hdiag[ibf] = level
                ibf += 1

                for p in range(0,nprim):
                    basis_alp[ipr] = a[p]
                    basis_cont[ipr] = c[p]
                    ipr += 1

                basis_valao2[iao] = valao
                basis_aoat2[iao] = iat
                basis_lao2[iao] = 1
                basis_hdiag2[iao] = level
                basis_aoexp[iao] = zeta
                basis_ao2sh[iao] = ish
                iao += 1
#
#         if(l.eq.0.and.ati.le.2.and.valao==0)then
#            ! diff s
#            call slaterToGauss(thisprimR, npq, l, zeta, aR, cR, .true., info)
#            call atovlp(0,nprim,thisprimR,a,aR,c,cR,ss)
#            basis%minalp(ish) = min(minval(a(:nprim)),minval(aR(:thisprimR)))
#
#            ibf =ibf+1
#            basis%primcount(ibf) = ipr
#            basis%valao    (ibf) = valao
#            basis%aoat     (ibf) = iat
#            basis%lao      (ibf) = 1
#            basis%nprim    (ibf) = thisprimR+nprim
#            basis%hdiag    (ibf) = level
#
#            idum=ipr+1
#            do p=1,thisprimR
#               ipr=ipr+1
#               basis%alp (ipr)=aR(p)
#               basis%cont(ipr)=cR(p)
#            enddo
#            do p=1,nprim
#               ipr=ipr+1
#               basis%alp (ipr)=a(p)
#               basis%cont(ipr)=-ss*c(p)
#            enddo
#            call atovlp(0,basis%nprim(ibf),basis%nprim(ibf), &
#               &        basis%alp(idum),basis%alp(idum), &
#               &        basis%cont(idum),basis%cont(idum),ss)
#            do p=1,basis%nprim(ibf)
#               basis%cont(idum-1+p)=basis%cont(idum-1+p)/sqrt(ss)
#            enddo
#
#            iao = iao+1
#            basis%valao2(iao) = valao
#            basis%aoat2 (iao) = iat
#            basis%lao2  (iao) = 1
#            basis%hdiag2(iao) = level
#            basis%aoexp (iao) = zeta
#            basis%ao2sh (iao) = ish
#         endif
            if ( l == 0 and ati < 2 and valao == 0):
                aR, cR, info = slaterToGauss(thisprimR, npq, zeta, True)
                ss = atovlp(0,nprim,thisprimR,a,aR,c,cR)
                basis_minalp[ish] = min(np.min(a[:nprim]),np.min(aR[:thisprimR]))

                basis_primcount[ibf] = ipr
                basis_valao[ibf] = valao
                basis_aoat[ibf] = iat
                basis_lao[ibf] = 1
                basis_nprim[ibf] = thisprimR+nprim
                basis_hdiag[ibf] = level
                ibf += 1

                idum=ipr
                for p in range(0,thisprimR):
                    basis_alp[ipr] = aR[p]
                    basis_cont[ipr] = cR[p]
                    ipr+=1

                for p in range(0,nprim):
                    basis_alp[ipr] = a[p]
                    basis_cont[ipr] = -ss*c[p]
                    ipr += 1

                ss = atovlp(0,basis_nprim[ibf],basis_nprim[ibf],basis_alp[idum],basis_alp[idum],basis_cont[idum],basis_cont[idum])
                idum+=1
                
                for p in range(0,basis_nprim[ibf]):
                    basis_cont[idum-1+p]=basis_cont[idum-1+p]/np.sqrt(ss)

                basis_valao2[iao] = valao
                basis_aoat2[iao] = iat
                basis_lao2[iao] = 1
                basis_hdiag2[iao] = level
                basis_aoexp[iao] = zeta
                basis_ao2sh[iao] = ish
                iao += 1
#
#         ! p polarization
#         if(l.eq.1.and.ati.le.2)then
#            call slaterToGauss(nprim, npq, l, zeta, a, c, .true., info)
#            basis%minalp(ish) = minval(a(:nprim))
#            do j=2,4
#               ibf=ibf+1
#               basis%primcount(ibf) = ipr
#               basis%aoat     (ibf) = iat
#               basis%lao      (ibf) = j
#               basis%valao    (ibf) = -valao
#               basis%nprim    (ibf) = nprim
#               basis%hdiag    (ibf) = level
#
#               do p=1,nprim
#                  ipr=ipr+1
#                  basis%alp (ipr)=a(p)
#                  basis%cont(ipr)=c(p)
#               enddo
#
#               iao = iao+1
#               basis%valao2(iao) = -valao
#               basis%aoat2 (iao) = iat
#               basis%lao2  (iao) = j
#               basis%hdiag2(iao) = level
#               basis%aoexp (iao) = zeta
#               basis%ao2sh (iao) = ish
#            enddo
#         endif
            if ( l == 1 and ati < 2):
                a, c, info = slaterToGauss(nprim, npq, l, zeta, True)
                basis_minalp[ish] = np.min(a[:nprim])
                for j in range(2,5):
                    basis_primcount[ibf] = ipr
                    basis_valao[ibf] = -valao
                    basis_aoat[ibf] = iat
                    basis_lao[ibf] = j
                    basis_nprim[ibf] = nprim
                    basis_hdiag[ibf] = level
                    ibf += 1

                    for p in range(0,nprim):
                        basis_alp[ipr] = a[p]
                        basis_cont[ipr] = c[p]
                        ipr += 1

                    basis_valao2[iao] = -valao
                    basis_aoat2[iao] = iat
                    basis_lao2[iao] = j
                    basis_hdiag2[iao] = level
                    basis_aoexp[iao] = zeta
                    basis_ao2sh[iao] = ish
                    iao += 1
#
#         ! general sp
#         if(l.eq.0.and.ati.gt.2 .and. valao/=0)then
#            ! s
#            call slaterToGauss(nprim, npq, l, zeta, as, cs, .true., info)
#            basis%minalp(ish) = minval(as(:nprim))
#
#            ibf=ibf+1
#            basis%primcount(ibf) = ipr
#            basis%valao    (ibf) = valao
#            basis%aoat     (ibf) = iat
#            basis%lao      (ibf) = 1
#            basis%nprim    (ibf) = nprim
#            basis%hdiag    (ibf) = level
#
#            do p=1,nprim
#               ipr=ipr+1
#               basis%alp (ipr)=as(p)
#               basis%cont(ipr)=cs(p)
#            enddo
#
#            iao = iao+1
#            basis%valao2(iao) = valao
#            basis%aoat2 (iao) = iat
#            basis%lao2  (iao) = 1
#            basis%hdiag2(iao) = level
#            basis%aoexp (iao) = zeta
#            basis%ao2sh (iao) = ish
#         endif
            if ( l == 0 and ati >= 2 and valao != 0):
                aS, cS, info = slaterToGauss(nprim, npq, l, zeta, True)
                basis_minalp[ish] = np.min(aS[:nprim])

                basis_primcount[ibf] = ipr
                basis_valao[ibf] = valao
                basis_aoat[ibf] = iat
                basis_lao[ibf] = 1
                basis_nprim[ibf] = nprim
                basis_hdiag[ibf] = level
                ibf += 1

                for p in range(0,nprim):
                    basis_alp[ipr] = aS[p]
                    basis_cont[ipr] = cS[p]
                    ipr += 1

                basis_valao2[iao] = valao
                basis_aoat2[iao] = iat
                basis_lao2[iao] = 1
                basis_hdiag2[iao] = level
                basis_aoexp[iao] = zeta
                basis_ao2sh[iao] = ish
                iao += 1
#         ! p
#         if(l.eq.1.and.ati.gt.2)then
#            call slaterToGauss(nprim, npq, l, zeta, a, c, .true., info)
#            basis%minalp(ish) = minval(a(:nprim))
#            do j=2,4
#               ibf=ibf+1
#               basis%primcount(ibf) = ipr
#               basis%valao    (ibf) = valao
#               basis%aoat     (ibf) = iat
#               basis%lao      (ibf) = j
#               basis%nprim    (ibf) = nprim
#               basis%hdiag    (ibf) = level
#
#               do p=1,nprim
#                  ipr=ipr+1
#                  basis%alp (ipr)=a(p)
#                  basis%cont(ipr)=c(p)
#               enddo
#
#               iao = iao+1
#               basis%valao2(iao) = valao
#               basis%aoat2 (iao) = iat
#               basis%lao2  (iao) = j
#               basis%hdiag2(iao) = level
#               basis%aoexp (iao) = zeta
#               basis%ao2sh (iao) = ish
#            enddo
#         endif
            if ( l == 1 and ati >= 2):
                a, c, info = slaterToGauss(nprim, npq, l, zeta, True)
                basis_minalp[ish] = np.min(a[:nprim])
                for j in range(2,5):
                    basis_primcount[ibf] = ipr
                    basis_valao[ibf] = valao
                    basis_aoat[ibf] = iat
                    basis_lao[ibf] = j
                    basis_nprim[ibf] = nprim
                    basis_hdiag[ibf] = level
                    ibf += 1

                    for p in range(0,nprim):
                        basis_alp[ipr] = a[p]
                        basis_cont[ipr] = c[p]
                        ipr += 1

                    basis_valao2[iao] = valao
                    basis_aoat2[iao] = iat
                    basis_lao2[iao] = j
                    basis_hdiag2[iao] = level
                    basis_aoexp[iao] = zeta
                    basis_ao2sh[iao] = ish
                    iao += 1
#
#         ! DZ s
#         if(l.eq.0 .and. ati > 2 .and. valao==0)then
#            call slaterToGauss(thisprimR, npq, l, zeta, aR, cR, .true., info)
#            call atovlp(0,nprim,thisprimR,as,aR,cs,cR,ss)
#            basis%minalp(ish) = min(minval(as(:nprim)),minval(aR(:thisprimR)))
#
#            ibf=ibf+1
#            basis%primcount(ibf) = ipr
#            basis%valao    (ibf) = valao
#            basis%aoat     (ibf) = iat
#            basis%lao      (ibf) = 1
#            basis%nprim    (ibf) = thisprimR+nprim
#            basis%hdiag    (ibf) = level
#
#            idum=ipr+1
#            do p=1,thisprimR
#               ipr=ipr+1
#               basis%alp (ipr)=aR(p)
#               basis%cont(ipr)=cR(p)
#            enddo
#            do p=1,nprim
#               ipr=ipr+1
#               basis%alp (ipr)=as(p)
#               basis%cont(ipr)=-ss*cs(p)
#            enddo
#            call atovlp(0,basis%nprim(ibf),basis%nprim(ibf), &
#               &        basis%alp(idum),basis%alp(idum), &
#               &        basis%cont(idum),basis%cont(idum),ss)
#            do p=1,basis%nprim(ibf)
#               basis%cont(idum-1+p)=basis%cont(idum-1+p)/sqrt(ss)
#            enddo
#
#            iao = iao+1
#            basis%valao2(iao) = valao
#            basis%aoat2 (iao) = iat
#            basis%lao2  (iao) = 1
#            basis%hdiag2(iao) = level
#            basis%aoexp (iao) = zeta
#            basis%ao2sh (iao) = ish
#         endif
            if ( l == 0 and ati >= 2 and valao == 0):
                aR, cR, info = slaterToGauss(thisprimR, npq, zeta, True)
                ss = atovlp(0,nprim,thisprimR,aS,aR,cS,cR)
                basis_minalp[ish] = min(np.min(aS[:nprim]),np.min(aR[:thisprimR]))

                basis_primcount[ibf] = ipr
                basis_valao[ibf] = valao
                basis_aoat[ibf] = iat
                basis_lao[ibf] = 1
                basis_nprim[ibf] = thisprimR+nprim
                basis_hdiag[ibf] = level
                ibf += 1

                idum = ipr
                for p in range(0,thisprimR):
                    basis_alp[ipr] = aR[p]
                    basis_cont[ipr] = cR[p]
                    ipr+=1

                for p in range(0,nprim):
                    basis_alp[ipr] = aS[p]
                    basis_cont[ipr] = -ss*cS[p]
                    ipr += 1

                ss = atovlp(0, basis_nprim[ibf], basis_nprim[ibf], basis_alp[idum], basis_alp[idum], basis_cont[idum], basis_cont[idum])
                idum += 1
                
                for p in range(0,basis_nprim[ibf]):
                    basis_cont[idum-1+p]=basis_cont[idum-1+p]/np.sqrt(ss)

                basis_valao2[iao] = valao
                basis_aoat2[iao] = iat
                basis_lao2[iao] = 1
                basis_hdiag2[iao] = level
                basis_aoexp[iao] = zeta
                basis_ao2sh[iao] = ish
                iao += 1
#
#         ! d
#         if(l.eq.2)then
            if l == 2:
#            call set_d_function(basis,iat,ish,iao,ibf,ipr, &
#               &                npq,l,nprim,zeta,level,valao)
#            MODIFICATIION: inset this function directly to not pass ton of arrays.
#   integer  :: j,p
#   real(wp) :: alp(10),cont(10)
#   real(wp) :: trafo(5:10) = &
#      & [1.0_wp, 1.0_wp, 1.0_wp, sqrt(3.0_wp), sqrt(3.0_wp), sqrt(3.0_wp)]
                trafo = np.array([0,0,0,0,0, # 5 empty spaces
                                  1,1,1,np.sqrt(3.),np.sqrt(3.),np.sqrt(3.)],dtype=np.float64)
#   integer :: info
#
#   call slaterToGauss(nprim, npq, l, zeta, alp, cont, .true., info)
                alp, cont, info = slaterToGauss(nprim, npq, l, zeta, True)
#   basis%minalp(ish) = minval(alp(:nprim))
                basis_minalp[ish] = np.min(alp[:nprim])
#
#   do j = 5, 10
                for j in range(5,11):
                    basis_primcount[ibf] = ipr
                    basis_valao[ibf] = valao
                    basis_aoat[ibf] = iat
                    basis_lao[ibf] = j
                    basis_nprim[ibf] = nprim
                    basis_hdiag[ibf] = level
                    ibf += 1

                    for p in range(0,nprim):
                        basis_alp[ipr]=alp[p]
                        basis_cont[ipr]=cont[p]*trafo[j]
                        ipr += 1
                    
                    if j == 5: 
                        continue

                    basis_valao2[iao] = valao
                    basis_aoat2 [iao] = iat
                    basis_lao2  [iao] = j-1
                    basis_hdiag2[iao] = level
                    basis_aoexp [iao] = zeta
                    basis_ao2sh [iao] = ish
                    iao += 1
#
#      ibf = ibf+1
#      basis%primcount(ibf) = ipr
#      basis%valao    (ibf) = valao
#      basis%aoat     (ibf) = iat
#      basis%lao      (ibf) = j
#      basis%nprim    (ibf) = nprim
#      basis%hdiag    (ibf) = level
#
#      do p=1,nprim
#         ipr = ipr+1
#         basis%alp (ipr)=alp (p)
#         basis%cont(ipr)=cont(p)*trafo(j)
#      enddo
#
#      if (j .eq. 5) cycle
#
#      iao = iao+1
#      basis%valao2(iao) = valao
#      basis%aoat2 (iao) = iat
#      basis%lao2  (iao) = j-1
#      basis%hdiag2(iao) = level
#      basis%aoexp (iao) = zeta
#      basis%ao2sh (iao) = ish
#
#         endif
#
#         ! f
#         if(l.eq.3)then
            if l == 3:
#            MODIFICATIION: inset this function directly to not pass ton of arrays.
#            call set_f_function(basis,iat,ish,iao,ibf,ipr, &
#               &                npq,l,nprim,zeta,level,1)
#         endif
#   real(wp) :: trafo(11:20) = &
#      & [1.0_wp, 1.0_wp, 1.0_wp, sqrt(5.0_wp), sqrt(5.0_wp), &
#      &  sqrt(5.0_wp), sqrt(5.0_wp), sqrt(5.0_wp), sqrt(5.0_wp), sqrt(15.0_wp)]
                valao = 1
                trafo = np.array([0,0,0,0,0,0,0,0,0,0,0, # 11 empty spaces
                                  1.0, 1.0, 1.0, np.sqrt(5.0), np.sqrt(5.0), np.sqrt(5.0), np.sqrt(5.0), np.sqrt(5.0), np.sqrt(5.0), np.sqrt(15.0)],dtype = np.float64)
#   integer :: info
#
#   call slaterToGauss(nprim, npq, l, zeta, alp, cont, .true., info)
                alp, cont, info = slaterToGauss(nprim,npq,l,zeta,True)
#   basis%minalp(ish) = minval(alp(:nprim))
                basis_minalp[ish] = np.min(alp[:nprim])
#
#   do j = 11, 20
                for j in range(11,21):
                    basis_primcount[ibf] = ipr
                    basis_valao[ibf] = valao
                    basis_aoat[ibf] = iat
                    basis_lao[ibf] = j
                    basis_nprim[ibf] = nprim
                    basis_hdiag[ibf] = level
                    ibf += 1

                    for p in range(0,nprim):
                        basis_alp[ipr]=alp[p]
                        basis_cont[ipr]=cont[p]*trafo[j]
                        ipr += 1

                    if (j >= 11 and j <= 13):
                        continue

                    basis_valao2[iao] = valao
                    basis_aoat2[iao] = iat
                    basis_lao2[iao] = j-3
                    basis_hdiag2[iao] = level
                    basis_aoexp[iao] = zeta
                    basis_ao2sh[iao] = ish
                    iao = iao+1
#
#      ibf = ibf+1
#      basis%primcount(ibf) = ipr
#      basis%valao    (ibf) = valao
#      basis%aoat     (ibf) = iat
#      basis%lao      (ibf) = j
#      basis%nprim    (ibf) = nprim
#      basis%hdiag    (ibf) = level
#
#      do p=1,nprim
#         ipr = ipr+1
#         basis%alp (ipr)=alp (p)
#         basis%cont(ipr)=cont(p)*trafo(j)
#      enddo
#
#      if (j.ge.11 .and. j.le.13) cycle
#
#      iao = iao+1
#      basis%valao2(iao) = valao
#      basis%aoat2 (iao) = iat
#      basis%lao2  (iao) = j-3
#      basis%hdiag2(iao) = level
#      basis%aoexp (iao) = zeta
#      basis%ao2sh (iao) = ish
#
#         basis%sh2bf(2,ish) = ibf-basis%sh2bf(1,ish)
#         basis%sh2ao(2,ish) = iao-basis%sh2ao(1,ish)
            basis_sh2bf[ish,1] = ibf-basis_sh2bf[ish,0]
            basis_sh2ao[ish,1] = iao-basis_sh2ao[ish,0]
            ish += 1
#      enddo shells
#      basis%shells(2,iat)=ish
#      basis%fila  (2,iat)=ibf
#      basis%fila2 (2,iat)=iao
        basis_shells[iat,1]=ish
        basis_fila[iat,1]=ibf
        basis_fila2[iat,1]=iao
#   enddo atoms
#
#   ok = all(basis%alp(:ipr) > 0.0_wp) .and. basis%nbf == ibf .and. basis%nao == iao
    ok = np.all(basis_alp[:ipr] > 0) and nbf == ibf and nao == iao
    return basis_shells, basis_sh2ao, basis_sh2bf, basis_minalp, basis_level, basis_zeta, basis_valsh,\
        basis_hdiag, basis_alp, basis_cont, basis_hdiag2, basis_aoexp, basis_ash, basis_lsh, basis_ao2sh, \
        basis_nprim, basis_primcount, basis_caoshell, basis_saoshell, basis_fila, basis_fila2, basis_lao, \
        basis_aoat, basis_valao, basis_lao2, basis_aoat2, basis_valao2, ok
#
#end subroutine newBasisset
#
#
#! ========================================================================
#!> determine basisset limits
#subroutine xbasis0(xtbData,n,at,basis)
def xbasis0(element_ids):
#   type(TxTBData), intent(in) :: xtbData
#   type(TBasisset),intent(inout) :: basis
#   integer,intent(in)  :: n
#   integer,intent(in)  :: at(n)
#   integer :: nbf
#   integer :: nao
#   integer :: nshell
#
#   integer i,j,k,l
#
#   call dim_basis(xtbData,n,at,nshell,nao,nbf)
    n = element_ids.shape[0]
    nshell, nao, nbf = dim_basis(element_ids)
#
#   call basis%allocate(n,nbf,nao,nshell)
    allocate_basisset(n, nbf, nao, nshell)
    
#
#end subroutine xbasis0
#subroutine allocate_basisset(self,n,nbf,nao,nshell)
def allocate_basisset(n, nbf, nao, nshell):
#   implicit none
#   class(TBasisset),intent(inout) :: self
#   integer,intent(in) :: n,nbf,nao,nshell
#   self%n=n
#   self%nbf=nbf
#   self%nao=nao
#   self%nshell=nshell
#   call self%deallocate
#   allocate( self%shells(2,n),    source = 0 )
    shells = np.zeros(nshell,2)
#   allocate( self%sh2ao(2,nshell),source = 0 )
    sh2ao = np.zeros(nshell,2)
#   allocate( self%sh2bf(2,nshell),source = 0 )
    sh2bf = np.zeros(nshell,2)
#   allocate( self%minalp(nshell), source = 0.0_wp )
    minalp = np.zeros(nshell)
#   allocate( self%level(nshell),  source = 0.0_wp )
    level = np.zeros(nshell)
#   allocate( self%zeta(nshell),   source = 0.0_wp )
    zeta = np.zeros(nshell)
#   allocate( self%valsh(nshell),  source = 0 )
    valsh = np.zeros(nshell)
#   allocate( self%hdiag(nbf),     source = 0.0_wp )
    hdiag = np.zeros(nbf)
#   allocate( self%alp(9*nbf),     source = 0.0_wp )
    alp = np.zeros(9*nbf)
#   allocate( self%cont(9*nbf),    source = 0.0_wp )
    cont = np.zeros(9*nbf)
#   allocate( self%hdiag2(nao),    source = 0.0_wp )
    hdiag2 = np.zeros(nao)
#   allocate( self%aoexp(nao),     source = 0.0_wp )
    aoexp = np.zeros(nao)
#   allocate( self%ash(nao),       source = 0 )
    ash = np.zeros(nao)
#   allocate( self%lsh(nao),       source = 0 )
    lsh = np.zeros(nao)
#   allocate( self%ao2sh(nao),     source = 0 )
    ao2sh = np.zeros(nao)
#   allocate( self%nprim(nbf),     source = 0 )
    nprim = np.zeros(nbf)
#   allocate( self%primcount(nbf), source = 0 )
    primcount = np.zeros(nbf)
#   allocate( self%caoshell(5,n),  source = 0 )
    caoshell = np.zeros(n,5)
#   allocate( self%saoshell(5,n),  source = 0 )
    saoshell = np.zeros(n,5)
#   allocate( self%fila(2,n),      source = 0 )
    fila = np.zeros(n,2)
#   allocate( self%fila2(2,n),     source = 0 )
    fila2 = np.zeros(n,2)
#   allocate( self%lao(nbf),       source = 0 )
    lao = np.zeros(nbf)
#   allocate( self%aoat(nbf),      source = 0 )
    aoat = np.zeros(nbf)
#   allocate( self%valao(nbf),     source = 0 )
    valao = np.zeros(nbf)
#   allocate( self%lao2(nao),      source = 0 )
    lao2 = np.zeros(nao)
#   allocate( self%aoat2(nao),     source = 0 )
    aoat2 = np.zeros(nao)
#   allocate( self%valao2(nbf),    source = 0 )
    valao2 = np.zeros(nao)
#end subroutine allocate_basisset
BAS = bool(1)
if BAS:
    t1 = time.time()
    x1 = new_basis_set(element_ids) 
    t2 = time.time()
    x2 = new_basis_set_simple(element_ids)
    t3 = time.time()
    allclose = True
    sum1 = 0
    sum2 = 0
    for idx, (output1, output2) in enumerate(zip(x1,x2)):
        close = np.allclose(output1,output2) or np.allclose(output1-1,output2)
        if not close:
            print(f"{idx+1:02} normal: {output1}\nsimple: {output2}")
        allclose &= close
        sum1 += np.sum(output1)
        sum2 += np.sum(output2)
    print(f"all close: {allclose}")
    print_res2(sum1,sum2,t1,t2,t3,"Sum of new basis set output")
