import math
import numpy as np
from blas import mchrg_dsymv, mchrg_dsytrs1
from dftd4_reference import hcount, alphaiw, ascale, refh, sscale, refsys, secaiw, refn

thopi = 3.0/math.pi
ootpi = 0.5/math.pi

Z_eff = np.array([
    1,                                                 2,           # H-He
    3, 4,                               5, 6, 7, 8, 9,10,           # Li-Ne
    11,12,                              13,14,15,16,17,18,          # Na-Ar
    19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,          # K-Kr
    9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,           # Rb-Xe
    9,10,11,30,31,32,33,34,35,36,37,38,39,40,41,42,43,              # Cs-Lu
    12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,                   # Hf-Rn
    9,10,11,30,31,32,33,34,35,36,37,38,39,40,41,42,43,              # Fr-Lr
    12,13,14,15,16,17,18,19,20,21,22,23,24,25,26                    # Rf-Og
], dtype=np.int32)

r4_over_r2 = np.array(
    [
        0.0,  # None
        8.0589,  # H
        3.4698,  # He
        29.0974,  # Li (2nd)
        14.8517,  # Be
        11.8799,  # B
        7.8715,  # C
        5.5588,  # N
        4.7566,  # O
        3.8025,  # F
        3.1036,  # Ne
        26.1552,  # Na (3rd)
        17.2304,  # Mg
        17.7210,  # Al
        12.7442,  # Si
        9.5361,  # P
        8.1652,  # S
        6.7463,  # Cl
        5.6004,  # Ar
        29.2012,  # K  (4th)
        22.3934,  # Ca
        19.0598,  # Sc
        16.8590,  # Ti
        15.4023,  # V
        12.5589,  # Cr
        13.4788,  # Mn
        12.2309,  # Fe
        11.2809,  # Co
        10.5569,  # Ni
        10.1428,  # Cu
        9.4907,  # Zn
        13.4606,  # Ga
        10.8544,  # Ge
        8.9386,  # As
        8.1350,  # Se
        7.1251,  # Br
        6.1971,  # Kr
        30.0162,  # Rb (5th)
        24.4103,  # Sr
        20.3537,  # Y
        17.4780,  # Zr
        13.5528,  # Nb
        11.8451,  # Mo
        11.0355,  # Tc
        10.1997,  # Ru
        9.5414,  # Rh
        9.0061,  # Pd
        8.6417,  # Ag
        8.9975,  # Cd
        14.0834,  # In
        11.8333,  # Sn
        10.0179,  # Sb
        9.3844,  # Te
        8.4110,  # I
        7.5152,  # Xe
        32.7622,  # Cs (6th)
        27.5708,  # Ba
        23.1671,  # La
        21.6003,  # Ce
        20.9615,  # Pr
        20.4562,  # Nd
        20.1010,  # Pm
        19.7475,  # Sm
        19.4828,  # Eu
        15.6013,  # Gd
        19.2362,  # Tb
        17.4717,  # Dy
        17.8321,  # Ho
        17.4237,  # Er
        17.1954,  # Tm
        17.1631,  # Yb
        14.5716,  # Lu
        15.8758,  # Hf
        13.8989,  # Ta
        12.4834,  # W
        11.4421,  # Re
        10.2671,  # Os
        8.3549,  # Ir
        7.8496,  # Pt
        7.3278,  # Au
        7.4820,  # Hg
        13.5124,  # Tl
        11.6554,  # Pb
        10.0959,  # Bi
        9.7340,  # Po
        8.8584,  # At
        8.0125,  # Rn
        29.8135,  # Fr (7th)
        26.3157,  # Ra
        19.1885,  # Ac
        15.8542,  # Th
        16.1305,  # Pa
        15.6161,  # U
        15.1226,  # Np
        16.1576,  # Pu
        14.6510,  # Am
        14.7178,  # Cm
        13.9108,  # Bk
        13.5623,  # Cf
        13.2326,  # Es
        12.9189,  # Fm
        12.6133,  # Md
        12.3142,  # No
        14.8326,  # Lr
        12.3771,  # Rf
        10.6378,  # Db
        9.3638,  # Sg
        8.2297,  # Bh
        7.5667,  # Hs
        6.9456,  # Mt
        6.3946,  # Ds
        5.9159,  # Rg
        5.4929,  # Cn
        6.7286,  # Nh
        6.5144,  # Fl
        10.9169,  # Lv
        10.3600,  # Mc
        9.4723,  # Ts
        8.6641,  # Og
    ]
)

sqrt_z_r4_over_r2 = np.sqrt(
    np.array([0.5 * math.sqrt(z) for z in range(0, 119)]) * r4_over_r2
)

# https://github.com/dftd4/dftd4/blob/502d7c59bf88beec7c90a71c4ecf80029794bd5e/src/dftd4/reference.f90#L285
# Set the reference polarizibility for an atomic number
# TODO: Add proper test
def set_refalpha_gfn2_num(alpha, ga, gc, atomic_number):
    if (atomic_number >= 0 and atomic_number < len(refn)):
        ref = int(refn[atomic_number-1])
        for ir in range(ref):
            _is = refsys[atomic_number-1, ir]
            if (abs(_is) < 1e-12):
                continue

            iz = get_effective_charge_num(_is-1)
            aiw = sscale[_is-1] * secaiw[_is-1, :] * zeta(ga, get_hardness_num(_is-1)*gc, iz, refh[atomic_number-1, ir]+iz)
            alpha[ir, :] = np.maximum(ascale[atomic_number-1, ir] * (alphaiw[atomic_number-1, ir, :] - hcount[atomic_number-1, ir] * aiw), 0.0)

def zeta(a, c, qref, qmod):
    if (qmod < 0.0):
        return math.exp(a)
    return math.exp(a * (1.0 - math.exp(c * (1.0 - qref/qmod))))

def dzeta(a, c, qref, qmod):
    if (qmod < 0.0):
        return 0.0
    return -a * c * math.exp(c * (1.0 - qref/qmod)) * zeta(a, c, qref, qmod) * qref/(qmod**2)

def get_effective_charge_num(atomic_number):
    if (atomic_number >= 0 and atomic_number < len(effective_nuclear_charge)):
        return effective_nuclear_charge[atomic_number]
    return 0.0

def get_hardness_num(atomic_number):
    if (atomic_number >= 0 and atomic_number < len(chemical_hardness)):
        return chemical_hardness[atomic_number]
    return 0.0


# Effective nuclear charges from the def2-ECPs used for calculating the reference
# polarizibilities for DFT-D4.
effective_nuclear_charge = np.array([
     1,                                                 2,  # H-He
     3, 4,                               5, 6, 7, 8, 9,10,  # Li-Ne
    11,12,                              13,14,15,16,17,18,  # Na-Ar
    19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,  # K-Kr
     9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,  # Rb-Xe
     9,10,11,30,31,32,33,34,35,36,37,38,39,40,41,42,43,  # Cs-Lu
    12,13,14,15,16,17,18,19,20,21,22,23,24,25,26, # Hf-Rn
    # just copy & paste from above
     9,10,11,30,31,32,33,34,35,36,37,38,39,40,41,42,43, # Fr-Lr
    12,13,14,15,16,17,18,19,20,21,22,23,24,25,26 # Rf-Og
], dtype=np.float64)


# Element-specific chemical hardnesses for the charge scaling function used
# to extrapolate the C6 coefficients in DFT-D4.
chemical_hardness = np.array([
    0.47259288, 0.92203391, 0.17452888, 0.25700733, 0.33949086,
    0.42195412, 0.50438193, 0.58691863, 0.66931351, 0.75191607,
    0.17964105, 0.22157276, 0.26348578, 0.30539645, 0.34734014,
    0.38924725, 0.43115670, 0.47308269, 0.17105469, 0.20276244,
    0.21007322, 0.21739647, 0.22471039, 0.23201501, 0.23933969,
    0.24665638, 0.25398255, 0.26128863, 0.26859476, 0.27592565,
    0.30762999, 0.33931580, 0.37235985, 0.40273549, 0.43445776,
    0.46611708, 0.15585079, 0.18649324, 0.19356210, 0.20063311,
    0.20770522, 0.21477254, 0.22184614, 0.22891872, 0.23598621,
    0.24305612, 0.25013018, 0.25719937, 0.28784780, 0.31848673,
    0.34912431, 0.37976593, 0.41040808, 0.44105777, 0.05019332,
    0.06762570, 0.08504445, 0.10247736, 0.11991105, 0.13732772,
    0.15476297, 0.17218265, 0.18961288, 0.20704760, 0.22446752,
    0.24189645, 0.25932503, 0.27676094, 0.29418231, 0.31159587,
    0.32902274, 0.34592298, 0.36388048, 0.38130586, 0.39877476,
    0.41614298, 0.43364510, 0.45104014, 0.46848986, 0.48584550,
    0.12526730, 0.14268677, 0.16011615, 0.17755889, 0.19497557, # Tl-At
    0.21240778, 0.07263525, 0.09422158, 0.09920295, 0.10418621, # Rn-Th
    0.14235633, 0.16394294, 0.18551941, 0.22370139, 0.25110000, # Pa-Am 
    0.25030000, 0.28840000, 0.31000000, 0.33160000, 0.35320000, # Cm-Fm
    0.36820000, 0.39630000, 0.40140000, 0.00000000, 0.00000000, # Md-Db
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, # Sg-Ds
    0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, # Rg-Mc
    0.00000000, 0.00000000, 0.00000000 # Lv,Ts,Og 
])


def new_structure(xyz, sym):
    ndim = min(sym.shape[0], xyz.shape[0])
    num = np.zeros(ndim, dtype=np.int64)
    for iat in range(ndim):
        num[iat] = symbol_to_number(sym[iat])

    ndim = min(num.shape[0], xyz.shape[0], sym.shape[0])

    _nat = ndim
    _id = np.zeros(ndim)
    _xyz = np.zeros((ndim, 3))
    _nid = get_identity_symbol(_id, sym)
    _map = np.zeros(_nid, dtype=np.int64)
    collect_identical(_id, _map)

    _num = np.zeros(_nid, dtype=np.int64)
    _sym = np.zeros(_nid, dtype='U1')
    for iid in range(_nid):
        _num[iid] = num[_map[iid]]
        _sym[iid] = sym[_map[iid]]

    _xyz[:, :] = xyz[:, :ndim]


    #if (present(periodic)) then
    #   self%periodic = periodic
    #else
    #   if (present(lattice)) then
    #      allocate(self%periodic(3))
    #      self%periodic(:) = .true.
    #   else
    #      allocate(self%periodic(1))
    #      self%periodic(:) = .false.
    #   end if
    #end if

    _lattice = np.zeros((1, 1))
    _periodic = np.full(1, False, dtype=bool)

    return _nat, _id, _xyz, _nid, _map, _num, _sym, _lattice, _periodic

# Get chemical identity from a list of element symbols
# Mutates identity and returns nid
def get_identity_symbol(identity, symbol):
    stmp = {}
    identity[:] = [stmp.setdefault(s, len(stmp)) for s in symbol]
    # return number of unique species (nid)
    return len(stmp)

def collect_identical(identity, mapping):
    for iid in range(len(mapping)):
        for iat in range(len(identity)):
            if (identity[iat] == iid):
                mapping[iid] = iat
                break


# Lower case version of the periodic system of elements
lcpse = np.array([
'h ','he', 
'li','be','b ','c ','n ','o ','f ','ne', 
'na','mg','al','si','p ','s ','cl','ar', 
'k ','ca', 
'sc','ti','v ','cr','mn','fe','co','ni','cu','zn', 
          'ga','ge','as','se','br','kr', 
'rb','sr', 
'y ','zr','nb','mo','tc','ru','rh','pd','ag','cd', 
          'in','sn','sb','te','i ','xe', 
'cs','ba','la', 
'ce','pr','nd','pm','sm','eu','gd','tb','dy','ho','er','tm','yb', 
'lu','hf','ta','w ','re','os','ir','pt','au','hg', 
          'tl','pb','bi','po','at','rn', 
'fr','ra','ac', 
'th','pa','u ','np','pu','am','cm','bk','cf','es','fm','md','no', 
'lr','rf','db','sg','bh','hs','mt','ds','rg','cn', 
          'nh','fl','mc','lv','ts','og'])

# ASCII offset between lowercase and uppercase letters
offset = ord('a') - ord('A')

def symbol_to_number(symbol):
    number = 0
    lcsymbol = [' '] * 2

    k = 0
    trimmed = symbol.strip()
    for j in range(len(trimmed)):
        if (k > 2):
            break

        l = ord(trimmed[j-1])
        if (k >= 1 and l == ord(' ') or k >= 1 and l == 9):
            break

        if (ord('A') <= l <= ord('Z')):
            l += offset

        if (ord('a') <= l <= ord('z')):
            k += 1
            if (k > 2):
                break
            lcsymbol[k-1] = chr(l)

    lcsymbol_joined = ''.join(lcsymbol)
    for i in range(len(lcpse)):
        if (lcsymbol_joined == lcpse[i]):
            number = i + 1
            break

    if (number == 0 and lcsymbol in ('d ', 't ')):
        number = 1

    return number

def get_nref_num(num):
    if (num >= 0 and num < len(refn)):
        return refn[num-1]
    return 0

def trapzd(pol):
   freq = np.array([
       0.000001, 0.050000, 0.100000, 
       0.200000, 0.300000, 0.400000, 
       0.500000, 0.600000, 0.700000, 
       0.800000, 0.900000, 1.000000, 
       1.200000, 1.400000, 1.600000, 
       1.800000, 2.000000, 2.500000, 
       3.000000, 4.000000, 5.000000, 
       7.500000, 10.00000
    ])

   weights = 0.5 * np.array([
        ( freq [1] - freq [0] ),  
        ( freq [1] - freq [0] ) + ( freq [2] - freq [1] ),  
        ( freq [2] - freq [1] ) + ( freq [3] - freq [2] ),  
        ( freq [3] - freq [2] ) + ( freq [4] - freq [3] ),  
        ( freq [4] - freq [3] ) + ( freq [5] - freq [4] ),  
        ( freq [5] - freq [4] ) + ( freq [6] - freq [5] ),  
        ( freq [6] - freq [5] ) + ( freq [7] - freq [6] ),  
        ( freq [7] - freq [6] ) + ( freq [8] - freq [7] ),  
        ( freq [8] - freq [7] ) + ( freq[9] - freq [8] ),  
        ( freq[9] - freq [8] ) + ( freq[10] - freq[9] ),  
        ( freq[10] - freq[9] ) + ( freq[11] - freq[10] ),  
        ( freq[11] - freq[10] ) + ( freq[12] - freq[11] ),  
        ( freq[12] - freq[11] ) + ( freq[13] - freq[12] ),  
        ( freq[13] - freq[12] ) + ( freq[14] - freq[13] ),  
        ( freq[14] - freq[13] ) + ( freq[15] - freq[14] ),  
        ( freq[15] - freq[14] ) + ( freq[16] - freq[15] ),  
        ( freq[16] - freq[15] ) + ( freq[17] - freq[16] ),  
        ( freq[17] - freq[16] ) + ( freq[18] - freq[17] ),  
        ( freq[18] - freq[17] ) + ( freq[19] - freq[18] ),  
        ( freq[19] - freq[18] ) + ( freq[20] - freq[19] ),  
        ( freq[20] - freq[19] ) + ( freq[21] - freq[20] ),  
        ( freq[21] - freq[20] ) + ( freq[22] - freq[21] ),  
        ( freq[22] - freq[21] )
    ])

   return np.sum(pol*weights)


thopi = 3.0/math.pi

#!> Create new D4 dispersion model from molecular structure input
#subroutine new_d4_model_with_checks(error, d4, mol, ga, gc, wf, ref)
def new_d4_model(nid, num, ga=3.0, gc=2.0, wf=6.0):
    ref = np.zeros(nid, dtype=np.int32)
    for isp in range(nid):
        izp = num[isp]
        ref[isp] = get_nref_num(izp)

    mref = np.max(ref)

    d4_aiw = np.zeros((nid, mref, 23))
    for isp in range(nid):
        izp = num[isp]
        set_refalpha_gfn2_num(d4_aiw[isp, :, :], ga, gc, izp)

    # NOTE: c6 coefficients
    aiw = np.zeros(23)
    d4_c6 = np.zeros((nid, nid, mref, mref))
    for isp in range(nid):
        izp = num[isp]
        for jsp in range(isp+1):
            for iref in range(ref[isp]):
                for jref in range(ref[jsp]):
                    aiw[:] = d4_aiw[isp, iref, :] * d4_aiw[jsp, jref, :]
                    c6 = thopi * trapzd(aiw) # NOTE: trapzd are the integration weights
                    d4_c6[isp, jsp, iref, jref] = c6
                    d4_c6[jsp, isp, jref, iref] = c6

    ncoup = 1

    return d4_c6, ref, ncoup



#!> Coordination number cutoff
#real(wp), parameter :: cn_default = 30.0_wp

#!> Two-body interaction cutoff
#real(wp), parameter :: disp2_default = 60.0_wp

#!> Three-body interaction cutoff
#real(wp), parameter :: disp3_default = 40.0_wp


#!> Collection of real space cutoffs
#type :: realspace_cutoff
#   sequence

#   !> Coordination number cutoff
#   real(wp) :: cn = cn_default

#   !> Two-body interaction cutoff
#   real(wp) :: disp2 = disp2_default

#   !> Three-body interaction cutoff
#   real(wp) :: disp3 = disp3_default

#end type realspace_cutoff

# Defaults
realspace_cutoff_cn = 30.0
realspace_cutoff_disp2 = 60.0
realspace_cutoff_disp3 = 40.0


# !> Wrapper to handle the evaluation of dispersion energy and derivatives
def get_dispersion(zeff, eta, ga, gc, wf, ngw, d4_cn, ref, rcov, kcn, norm_exp, ncoup, energy, lattice, periodic, num, gradient=None, sigma=None):
    mref = np.max(ref)
    grad = gradient is not None or sigma is not None

    lattr = get_lattice_points_cutoff(periodic, lattice, realspace_cutoff_cn)
    cn, _, _ = get_coordination_number(realspace_cutoff_cn, nat, id, xyz, lattr, rcov, kcn, norm_exp)

    print(f"coordination numbers: {cn}")

    dqdr = None
    dqdL = None
    if (grad):
        dqdr = np.zeros((nat, nat, 3))
        dqdL = np.zeros((nat, 3, 3))

    q = np.zeros(nat)
    get_charges(periodic, lattice, num, energy, gradient, sigma, q, dqdr, dqdL)

    gwvec = np.zeros((ncoup, nat, mref))
    gwdcn = None
    gwdq = None
    if (grad):
        # !> Number of atoms coupled to by pairwise parameters
        # integer :: ncoup
        gwdcn = np.zeros((mref, nat, ncoup))
        gwdq = np.zeros((mref, nat, ncoup))

    weight_references(zeff, eta, ga, gc, wf, ngw, d4_cn, cn, q, gwvec, gwdcn, gwdq)

    c6 = np.zeros((nat, nat))
    dc6dcn = None
    dc6dq = None
    if (grad):
       dc6dcn = np.zeros((nat, nat))
       dc6dq = np.zeros((nat, nat))

    get_atomic_c6(gwvec, gwdcn, gwdq, c6, dc6dcn, dc6dq)

    energies = np.zeros(nat)
    energies[:] = 0.0
    dEdcn = np.zeros((nat))
    dEdq = np.zeros((nat))
    if (grad):
        dEdcn[:] = 0.0
        dEdq[:] = 0.0
        assert gradient is not None
        gradient[:, :] = 0.0
        assert sigma is not None
        sigma[:, :] = 0.0

    trans = get_lattice_points_cutoff(periodic, lattice, realspace_cutoff_disp2)
    get_dispersion2(nat, id, xyz, trans, realspace_cutoff_disp2, r4_over_r2, c6, dc6dcn, dc6dq,energy, dEdcn, dEdq, gradient, sigma)

    if (grad):
        d4_gemv(gradient)
        d4_gemv(sigma)

    q[:] = 0.0
    weight_references()
    get_atomic_c6()

    get_lattice_points()
    get_dispersion3()

    if (grad):
        add_coordination_number_derivs()

    energy = np.sum(energies)



#!> Evaluation of the dispersion energy expression
def get_dispersion2(nat, id, xyz, trans, cutoff, r4r2, c6, dc6dcn, dc6dq,energy, dEdcn, dEdq, gradient, sigma):
   #!DEC$ ATTRIBUTES DLLEXPORT :: get_dispersion2

   #!> Damping parameters
   #class(rational_damping_param), intent(in) :: self

   #!> Molecular structure data
   #class(structure_type), intent(in) :: mol

   #!> Lattice points
   #real(wp), intent(in) :: trans(:, :)

   #!> Real space cutoff
   #real(wp), intent(in) :: cutoff

   #!> Expectation values for r4 over r2 operator
   #real(wp), intent(in) :: r4r2(:)

   #!> C6 coefficients for all atom pairs.
   #real(wp), intent(in) :: c6(:, :)

   #!> Derivative of the C6 w.r.t. the coordination number
   #real(wp), intent(in), optional :: dc6dcn(:, :)

   #!> Derivative of the C6 w.r.t. the partial charges
   #real(wp), intent(in), optional :: dc6dq(:, :)

   #!> Dispersion energy
   #real(wp), intent(inout) :: energy(:)

   #!> Derivative of the energy w.r.t. the coordination number
   #real(wp), intent(inout), optional :: dEdcn(:)

   #!> Derivative of the energy w.r.t. the partial charges
   #real(wp), intent(inout), optional :: dEdq(:)

   #!> Dispersion gradient
   #real(wp), intent(inout), optional :: gradient(:, :)

   #!> Dispersion virial
   #real(wp), intent(inout), optional :: sigma(:, :)

    if (abs(s6) < np.finfo(type(1.0)).eps and abs(s8) < np.finfo(type(1.0)).eps):
        return

    grad = dc6dcn is not None and dEdcn is not None and dc6dq is not None and dEdq is not None and gradient is not None and sigma is not None

    if (grad):
        get_dispersion_derivs(nat, id, xyz, trans, cutoff, r4r2, c6, dc6dcn, dc6dq, energy, dEdcn, dEdq, gradient, sigma)
    else:
        get_dispersion_energy(nat, id, xyz, trans, cutoff, r4r2, c6, energy)


# Scaling parameters for the dispersion model
# TODO: These values do not match any configuration in the dftd4 program. Change them?
a1 = 0.52
a2 = 5.0
s6 = 1.0
s8 = 2.7


#!> Evaluation of the dispersion energy expression
def get_dispersion_energy(nat, id, xyz, trans, cutoff, r4r2, c6, energy):
   #!> Damping parameters
   #class(rational_damping_param), intent(in) :: self

   #!> Molecular structure data
   #class(structure_type), intent(in) :: mol

   #!> Lattice points
   #real(wp), intent(in) :: trans(:, :)

   #!> Real space cutoff
   #real(wp), intent(in) :: cutoff

   #!> Expectation values for r4 over r2 operator
   #real(wp), intent(in) :: r4r2(:)

   #!> C6 coefficients for all atom pairs.
   #real(wp), intent(in) :: c6(:, :)

   #!> Dispersion energy
   #real(wp), intent(inout) :: energy(:)

    vec = np.zeros(3)
    cutoff2 = cutoff**2

    energy_local = np.zeros(energy.shape[0])
    for iat in range(nat):
        izp = id[iat]
        for jat in range(iat):
            jzp = id[iat]
            rrij = 3*r4r2[izp]*r4r2[jzp]
            r0ij = a1 * math.sqrt(rrij) + a2
            c6ij = c6[iat, jat]
            for jtr in range(trans.shape[1]):
                vec[:] = xyz[iat, :] - (xyz[jat, :] + trans[jtr, :])
                r2 = vec[0]**2 + vec[1]**2 + vec[2]**2
                if (r2 > cutoff2 or r2 < np.finfo(type(1.0)).eps):
                    continue

                t6 = 1.0/(r2**3 + r0ij**6)
                t8 = 1.0/(r2**4 + r0ij**8)

                edisp = s6*t6 + s8*rrij*t8

                dE = -c6ij*edisp * 0.5

                energy_local[iat] += dE
                if (iat != jat):
                    energy_local[jat] += dE

    energy[:] += energy_local[:]



#!> Evaluation of the dispersion energy expression
def get_dispersion_derivs(nat, id, xyz, trans, cutoff, r4r2, c6, dc6dcn, dc6dq, energy, dEdcn, dEdq, gradient, sigma):
   #!> Damping parameters
   #class(rational_damping_param), intent(in) :: self

   #!> Molecular structure data
   #class(structure_type), intent(in) :: mol

   #!> Lattice points
   #real(wp), intent(in) :: trans(:, :)

   #!> Real space cutoff
   #real(wp), intent(in) :: cutoff

   #!> Expectation values for r4 over r2 operator
   #real(wp), intent(in) :: r4r2(:)

   #!> C6 coefficients for all atom pairs.
   #real(wp), intent(in) :: c6(:, :)

   #!> Derivative of the C6 w.r.t. the coordination number
   #real(wp), intent(in) :: dc6dcn(:, :)

   #!> Derivative of the C6 w.r.t. the partial charges
   #real(wp), intent(in) :: dc6dq(:, :)

   #!> Dispersion energy
   #real(wp), intent(inout) :: energy(:)

   #!> Derivative of the energy w.r.t. the coordination number
   #real(wp), intent(inout) :: dEdcn(:)

   #!> Derivative of the energy w.r.t. the partial charges
   #real(wp), intent(inout) :: dEdq(:)

   #!> Dispersion gradient
   #real(wp), intent(inout) :: gradient(:, :)

   #!> Dispersion virial
   #real(wp), intent(inout) :: sigma(:, :)

    vec = np.zeros(3)
    dG = np.zeros(3)
    dS = np.zeros((3, 3))

    cutoff2 = cutoff**2

    energy_local = np.zeros(energy.shape[0])
    dEdcn_local = np.zeros(dEdcn.shape[0])
    dEdq_local = np.zeros(dEdq.shape[0])
    gradient_local = np.zeros((gradient.shape[0], gradient.shape[1]))
    sigma_local = np.zeros((sigma.shape[0], sigma.shape[1]))

    for iat in range(nat):
        izp = id[iat]
        for jat in range(iat):
            jzp = id[jat]
            rrij = 3*r4r2[izp] * r4r2[jzp]
            r0ij = a1 * math.sqrt(rrij) + a2
            c6ij = c6[iat, jat]
            for jtr in range(trans.shape[1]):
                vec[:] = xyz[iat, :] - (xyz[jat, :] + trans[jtr, :])
                r2 = vec[0]**2 + vec[1]**2 + vec[2]**2
                if (r2 > cutoff2 or r2 < np.finfo(type(1.0)).eps):
                    continue

                t6 = 1.0/(r2**3 + r0ij**6)
                t8 = 1.0/(r2**4 + r0ij**8)

                d6 = -6*r2**2*t6**2
                d8 = -8*r2**3*t8**2

                edisp = s6 * t6 + s8 * rrij * t8
                gdisp = s6 * d6 + s8 * rrij * d8

                dE = -c6ij * edisp * 0.5
                dG[:] = -c6ij * gdisp * vec
                dS[:, :] = 0.5 * dG[:, np.newaxis] * vec[np.newaxis, :] # NOTE: Is this correct?

                energy_local[iat] += dE
                dEdcn_local[iat] -= dc6dcn[jat, iat] * edisp
                dEdq_local[iat] -= dc6dq[jat, iat] * edisp
                sigma_local[:, :] += dS

                if (iat != jat):
                    energy_local[jat] += dE
                    dEdcn_local[jat] -= dc6dcn[iat, jat] * edisp
                    dEdq_local[jat] -= dc6dq[iat, jat] * edisp
                    gradient_local[iat, :] += dG
                    gradient_local[jat, :] -= dG
                    sigma_local[:, :] += dS

    energy[:] += energy_local[:]
    dEdcn[:] += dEdcn_local[:]
    dEdq[:] += dEdq_local[:]
    gradient[:, :] += gradient_local[:, :]
    sigma[:, :] += sigma_local[:, :]





#!> Calculate atomic dispersion coefficients and their derivatives w.r.t.
#!> the coordination numbers and atomic partial charges.
def get_atomic_c6(gwvec, gwdcn, gwdq, c6, dc6dcn, dc6dq):
   #!DEC$ ATTRIBUTES DLLEXPORT :: get_atomic_c6

   #!> Instance of the dispersion model
   #class(d4_model), intent(in) :: self

   #!> Molecular structure data
   #class(structure_type), intent(in) :: mol

   #!> Weighting function for the atomic reference systems
   #real(wp), intent(in) :: gwvec(:, :, :)

   #!> Derivative of the weighting function w.r.t. the coordination number
   #real(wp), intent(in), optional :: gwdcn(:, :, :)

   #!> Derivative of the weighting function w.r.t. the partial charge
   #real(wp), intent(in), optional :: gwdq(:, :, :)

   #!> C6 coefficients for all atom pairs.
   #real(wp), intent(out) :: c6(:, :)

   #!> Derivative of the C6 w.r.t. the coordination number
   #real(wp), intent(out), optional :: dc6dcn(:, :)

   #!> Derivative of the C6 w.r.t. the partial charge
   #real(wp), intent(out), optional :: dc6dq(:, :)

    if (gwdcn != None and dc6dcn != None and gwdq != None and dc6dq != None):
        c6[:, :] = 0.0
        dc6dcn[:, :] = 0.0
        dc6dq[:, :] = 0.0

        for iat in range(nat):
            izp = id[iat]
            for jat in range(iat):
                jzp = id[jat]
                dc6 = 0.0
                dc6dcni = 0.0
                dc6dcnj = 0.0
                dc6dqi = 0.0
                dc6dqj = 0.0
                for iref in range(ref[izp]):
                    for jref in range(ref[jzp]):
                        refc6 = c6[jzp, izp, jref, iref]
                        dc6 += gwvec[1, iat, iref] * gwvec[1, jat, jref] * refc6
                        dc6dcni += gwdcn[1, iat, iref] * gwvec[1, jat, iref] * refc6
                        dc6dcnj += gwvec[1, iat, iref] * gwdcn[1, jat, iref] * refc6
                        dc6dqi += gwdq[1, iat, iref] * gwvec[1, jat, jref] * refc6
                        dc6dqj += gwvec[1, iat, iref] * gwdq[1, jat, jref] * refc6

                c6[jat, iat] = dc6
                c6[iat, jat] = dc6
                dc6dcn[jat, iat] = dc6dcni
                dc6dcn[iat, jat] = dc6dcnj
                dc6dq[jat, iat] = dc6dqi
                dc6dq[iat, jat] = dc6dqj
    else:
        c6[:, :] = 0.0

        for iat in range(nat):
            izp = id[iat]
            for jat in range(iat):
                jzp = id[jat]
                dc6 = 0.0
                for iref in range(ref[izp]):
                    for jref in range(ref[jzp]):
                        refc6 = c6[jzp, izp, jref, iref]
                        dc6 += gwvec[1, iat, iref] * gwvec[1, jat, jref] * refc6

                c6[jat, iat] = dc6
                c6[iat, jat] = dc6


    


#!> Obtain charges from electronegativity equilibration model
def get_charges(periodic, lattice, num, energy, gradient, sigma, qvec, dqdr, dqdL):
   #!DEC$ ATTRIBUTES DLLEXPORT :: get_charges

   #!> Molecular structure data
   #type(structure_type), intent(in) :: mol

   #!> Atomic partial charges
   #real(wp), intent(out), contiguous :: qvec(:)

   #!> Derivative of the partial charges w.r.t. the Cartesian coordinates
   #real(wp), intent(out), contiguous, optional :: dqdr(:, :, :)

   #!> Derivative of the partial charges w.r.t. strain deformations
   #real(wp), intent(out), contiguous, optional :: dqdL(:, :, :)

    #cn_max = 8.0
    cutoff = 25.0

    rad, chi, eta, kcn, rcov, cutoff, cn_exp, cn_max = new_eeq2019_model(num)
    #if(allocated(error)) then
    #   write(error_unit, '("[Error]:", 1x, a)') error%message
    #   error stop
    #end if

    #if (grad):
    #   dcndr = np.zeros((nat, nat, 3))
    #   dcndL = np.zeros((nat, 3, 3))

    cn, dcndr, dcndL = get_cn(periodic, lattice, cutoff, rcov, kcn, cn_exp)
    solve(nat, periodic, lattice, cn, dcndr, dcndL, energy, gradient, sigma, qvec=qvec, dqdr=dqdr, dqdL=dqdL)

    #if(allocated(error)) then
    #  write(error_unit, '("[Error]:", 1x, a)') error%message
    #  error stop
    #end if


def get_cn(periodic, lattice, cutoff, rcov, kcn, norm_exp):
    # lattr
    trans = get_lattice_points_cutoff(periodic, lattice, cutoff)
    # NOTE: Should the cutoff of the two functions be the same?
    cn, dcndr, dcndL = get_coordination_number(cutoff, nat, id, xyz, trans, rcov, kcn, norm_exp)
    return cn, dcndr, dcndL


def solve(nat, periodic, lattice, cn, kcn, chi, charge, rad, eta, dcndr, dcndL, energy, gradient, sigma, qvec, dqdr, dqdL):
    ndim = nat + 1
    nimg = None
    tridx = None
    trans = None
    if (any(periodic)):
        nimg, tridx, trans = new_wignerseitz_cell(periodic, lattice)
        get_alpha(lattice, alpha)

    dcn = dcndr != None and dcndL != None
    grad = gradient != None and sigma != None and dcn
    cpq = dqdr != None and dqdL != None and dcn

    amat = np.zeros((ndim, ndim))
    xvec = np.zeros(ndim)
    dxdcn = None
    if (grad or cpq):
        dxdcn = np.zeros(ndim)

    get_vrhs(nat, id, kcn, chi, charge, cn, xvec, dxdcn)
    if (any(periodic)):
        get_amat_3d(lattice, nimg, trans, tridx, rad, eta, alpha, amat)
    else:
        get_amat_0d(rad, eta, amat)

    vrhs = xvec
    ainv = amat

    ipiv, info = mchrg_dsytrf(ainv, uplo='l')
    if (info != 0):
        print("Fatal Error: Bunch-Kaufman factorization failed")
        return

    if (cpq):
        #! Inverted matrix is needed for coupled-perturbed equations
        mchrg_dsytri(ainv, info)
        if (info != 0):
            print("Fatal Error: Inversion of factorized matrix failed")
            return

        #! Solve the linear system
        mchrg_dsymv(ainv, xvec, vrhs, uplo='l', alpha=None, beta=None)
        for ic in range(ndim):
            for jc in range(ic+1, ndim):
                ainv[jc, ic] = ainv[ic, jc]
    else:
        #! Solve the linear system
        mchrg_dsytrs1(ainv, vrhs, ipiv, uplo='l', info=info)
        if (info != 0):
            print("Fatal Error: Solution of linear system failed")
            return

    if (qvec != None):
        qvec[:] = vrhs[:nat]

    if (energy != None):
        mchrg_dsymv(amat[:nat, :], vrhs[:nat], xvec[:nat], uplo='l', alpha=0.5, beta=-1.0)
        energy[:] += vrhs[:nat] * xvec[:nat]

    if (grad or cpq):
        dadr = np.zeros((ndim, nat, 3))
        dadL = np.zeros((ndim, 3, 3))
        atrace = np.zeros((nat, 3))

        if (any(periodic)):
            get_damat_3d(lattice, rad, nimg, trans, tridx, alpha, vrhs, dadr, dadL, atrace)
        else:
            get_damat_0d(rad, vrhs, dadr, dadL, atrace)

        xvec[:] = -dxdcn * vrhs

    if (grad):
      gemv(dadr, vrhs, gradient, beta=1.0_wp)
      gemv(dcndr, xvec[beta=1.0, gradient, :nat])
      gemv(dadL, vrhs, sigma, beta=1.0, alpha=0.5)
      gemv(dcndL, xvec(:nat), sigma, beta=1.0)

    if (cpq):
        for iat in range(nat):
            dadr[iat, iat, :] = atrace[iat, :] + dadr[iat, iat, :]
            dadr[iat, :, :] = -dcndr[iat, :, :] * dxdcn[iat] + dadr[iat, :, :]
            dadL[iat, :, :] = -dcndr[iat, :, :] * dxdcn[iat] + dadL[iat, :, :]

        gemm(dadr, ainv[:nat, :], dqdr, alpha=-1.0)
        gemm(dadL, ainv[:nat, :], dqdL, alpha=-1.0)



def get_damat_0d(rad, qvec, dadr, dadL, atrace):
   #type(mchrg_model_type), intent(in) :: self
   #type(structure_type), intent(in) :: mol
   #real(wp), intent(in) :: qvec(:)
   #real(wp), intent(out) :: dadr(:, :, :)
   #real(wp), intent(out) :: dadL(:, :, :)
   #real(wp), intent(out) :: atrace(:, :)

    vec = np.zeros(3)

    atrace_local = np.array(atrace, copy=True)
    dadr_local = np.array(dadr, copy=True)
    dadL_local = np.array(dadL, copy=True)

    for iat in range(nat):
        izp = id[iat]
        for jat in range(iat-1):
            jzp = id[jat]
            vec = xyz[iat, :] - xyz[jat, :]
            r2 = vec[0]**2 + vec[1]**2 + vec[2]**2
            gam = 1.0 / math.sqrt(rad[izp]**2 + rad[jzp]**2)
            arg = gam*gam*r2
            dtmp = 2.0 * gam * math.exp(-arg) / (math.sqrt(math.pi)*r2) - math.erf(math.sqrt(arg))/(r2*math.sqrt(r2))
            dG = dtmp*vec
            dS = dG[:, None] * vec[None, :]
            atrace_local[iat, :] = +dG*qvec[jat] + atrace_local[iat, :]
            atrace_local[jat, :] = -dG*qvec[iat] + atrace_local[jat, :]
            dadr_local[jat, iat, :] = +dG*qvec[iat]
            dadr_local[iat, jat, :] = -dG*qvec[jat]
            dadL_local[jat, :, :] = +dS*qvec[iat] + dadL_local[jat, :, :]
            dadL_local[iat, :, :] = +dS*qvec[jat] + dadL_local[iat, :, :]

    atrace[:, :] += atrace_local[:, :]
    dadr[:, :, :] += dadr_local[:, :, :]
    dadL[:, :, :] += dadL_local[:, :, :]


def get_damat_3d(lattice, rad, nimg, trans, tridx, alpha, qvec, dadr, dadL, atrace):
   #type(mchrg_model_type), intent(in) :: self
   #type(structure_type), intent(in) :: mol
   #type(wignerseitz_cell_type), intent(in) :: wsc
   #real(wp), intent(in) :: alpha
   #real(wp), intent(in) :: qvec(:)
   #real(wp), intent(out) :: dadr(:, :, :)
   #real(wp), intent(out) :: dadL(:, :, :)
   #real(wp), intent(out) :: atrace(:, :)

    atrace[:, :] = 0.0
    dadr[:, :, :] = 0.0
    dadL[:, :, :] = 0.0

    vol = abs(np.linalg.det(lattice))
    dtrans = get_dir_trans(lattice)
    rtrans = get_rec_trans(lattice)

    dG = np.zeros(3)
    dS = np.zeros((3, 3))
    dSd = np.zeros((3, 3))
    dSr = np.zeros((3, 3))

    atrace_local = np.array(atrace, copy=True)
    dadr_local = np.array(dadr, copy=True)
    dadL_local = np.array(dadL, copy=True)

    for iat in range(nat):
        izp = id[iat]
        for jat in range(iat-1):
            jzp = id[jat]
            dG[:] = 0.0
            dS[:, :] = 0.0
            gam = 1.0 / math.sqrt(rad[izp]**2 + rad[jzp]**2)
            wsw = 1.0 / float(nimg[iat, jat])
            for img in range(nimg[iat, jat]):
                vec = xyz[iat, :] - xyz[jat, :] - trans[tridx[iat, jat, img], :]
                dGd, dSd = get_damat_dir_3d(vec, gam, alpha, dtrans)
                dGr, dSr = get_damat_rec_3d(vec, vol, alpha, rtrans)
                dG += (dGd + dGr) * wsw
                dS += (dSd + dSr) * wsw

            atrace_local[iat, :] = +dG*qvec[jat] + atrace_local[iat, :]
            atrace_local[jat, :] = -dG*qvec[jat] + atrace_local[jat, :]
            dadr_local[jat, iat, :] = +dG*qvec(iat) + dadr_local[jat, iat, :]
            dadr_local[iat, jat, :] = -dG*qvec(jat) + dadr_local[iat, jat, :]
            dadL_local[jat, :, :] = +dS*qvec(iat) + dadL_local[jat, :, :]
            dadL_local[iat, :, :] = +dS*qvec(jat) + dadL_local[iat, :, :]

        dS[:, :] = 0.0
        gam = 1.0 / math.sqrt(2.0 * rad[izp]**2)
        wsw = 1.0 / float(nimg[iat, iat])
        for img in range(nimg[iat, iat]):
            vec = trans[tridx[iat, iat, img], :]
            dGd, dSd = get_damat_dir_3d(vec, gam, alpha, dtrans)
            dGr, dSr = get_damat_rec_3d(vec, vol, alpha, rtrans)
            dS += (dSd + dSr) * wsw

        dadL_local[iat, :, :] = +dS*qvec[iat] + dadL_local[iat, :, :]

    atrace[:, :] += atrace_local[:, :]
    dadr[:, :, :] += dadr_local[:, :, :]
    dadL[:, :, :] += dadL_local[:, :, :]



def get_damat_dir_3d(rij, gam, alp, trans):
   #real(wp), intent(in) :: rij(3)
   #real(wp), intent(in) :: gam
   #real(wp), intent(in) :: alp
   #real(wp), intent(in) :: trans(:, :)
   #real(wp), intent(out) :: dg(3)
   #real(wp), intent(out) :: ds(3, 3)

    gam2 = gam**2
    alp2 = alp**2

    vec = np.zeros(3)
    dg = np.zeros(3)
    ds = np.zeros((3, 3))

    for itr in range(trans.shape[1]):
        vec[:] = rij + trans[itr, :]
        r1 = np.linalg.norm(vec)
        if (r1 < eps):
            continue
        r2 = r1**2
        gtmp = +2*gam*math.exp(-r2*gam2)/(math.sqrt(math.pi)*r2) - math.erf(r1*gam)/(r2*r1)
        atmp = -2*alp*math.exp(-r2*alp2)/(math.sqrt(math.pi)*r2) - math.erf(r1*alp)/(r2*r1)
        dg[:] += (gtmp + atmp) * vec
        ds[:, :] += (gtmp + atmp) * vec[:, None] * vec[None, :]

    return dg, ds


def get_damat_rec_3d(rij, vol, alp, trans):
   #real(wp), intent(in) :: rij(3)
   #real(wp), intent(in) :: vol
   #real(wp), intent(in) :: alp
   #real(wp), intent(in) :: trans(:, :)
   #real(wp), intent(out) :: dg(3)
   #real(wp), intent(out) :: ds(3, 3)

    unity = np.eye(3, dtype=np.float64)

    fac = 4*math.pi/vol
    alp2 = alp**2

    vec = np.zeros(3)
    dg = np.zeros(3)
    ds = np.zeros((3, 3))

    for itr in range(trans.shape[1]):
        vec[:] = trans[itr, :]
        g2 = np.dot(vec, vec)
        if (g2 < eps):
            continue
        gv = np.dot(rij, vec)
        etmp = fac * math.exp(-0.25 * g2/alp2)/g2
        dtmp = -math.sin(gv) * etmp
        dg[:] += dtmp * vec
        ds[:, :] += etmp * math.cos(gv) * ((2.0/g2 + 0.5/alp2) * vec[:, None] * vec[None, :] - unity)

    return dg, ds


def get_amat_3d(lattice, nimg, trans, tridx, rad, eta, alpha, amat):
   #type(mchrg_model_type), intent(in) :: self
   #type(structure_type), intent(in) :: mol
   #type(wignerseitz_cell_type), intent(in) :: wsc
   #real(wp), intent(in) :: alpha
   #real(wp), intent(out) :: amat(:, :)

    vec = np.zeros(3)
    amat[:, :] = 0.0

    vol = abs(np.linalg.det(lattice))
    dtrans = get_dir_trans(lattice)
    rtrans = get_rec_trans(lattice)

    amat_local = np.array(amat, copy=True)

    for iat in range(nat):
        izp = id[iat]
        for jat in range(iat-1):
            jzp = id[jat]
            gam = 1.0 / math.sqrt(rad[izp]**2 + rad[jzp]**2)
            wsw = 1.0 / float(nimg[iat, jat])
            for img in range(nimg[iat, jat]):
                vec = xyz[iat, :] - xyz[jat, :] - trans[tridx[iat, jat, img], :]
                dtmp = get_amat_dir_3d(vec, gam, alpha, dtrans)
                rtmp = get_amat_rec_3d(vec, vol, alpha, rtrans)
                amat_local[iat, jat] += (dtmp + rtmp) * wsw
                amat_local[jat, iat] += (dtmp + rtmp) * wsw

        gam = 1.0 / math.sqrt(2.0 * rad[izp]**2)
        wsw = 1.0 / float(nimg[iat, iat])
        for img in range(nimg[iat, iat]):
            vec = trans[tridx[iat, iat, img], :]
            dtmp = get_amat_dir_3d(vec, gam, alpha, dtrans)
            rtmp = get_amat_rec_3d(vec, vol, alpha, rtrans)
            amat_local[iat, iat] = amat_local[iat, iat] + (dtmp + rtmp) * wsw

        dtmp = eta[izp] + (math.sqrt(2.0/math.pi)) / rad[izp] - 2 * alpha / math.sqrt(math.pi)
        amat_local[iat, iat] += dtmp

    amat[:, :] += amat_local[:, :]

    amat[1:nat+1, nat+1] = 1.0
    amat[nat+1, 1:nat+1] = 1.0
    amat[nat+1, nat+1] = 0.0


def get_amat_0d(rad, eta, amat):
   #type(mchrg_model_type), intent(in) :: self
   #type(structure_type), intent(in) :: mol
   #real(wp), intent(out) :: amat(:, :)

    vec = np.zeros(3)
    amat[:, :] = 0.0

    amat_local = np.array(amat, copy=True)

    for iat in range(nat):
        izp = id[iat]
        for jat in range(iat-1):
            jzp = id[jat]
            vec = xyz[jat, :] - xyz[iat, :]
            r2 = vec[0]**2 + vec[1]**2 + vec[2]**2
            gam = 1.0 / (rad[izp]**2 + rad[jzp]**2)
            tmp = math.erf(math.sqrt(r2*gam)) / math.sqrt(r2)
            amat_local[iat, jat] += tmp
            amat_local[jat, iat] += tmp

        tmp = eta[izp] + math.sqrt(2*math.pi) / rad[izp]
        amat_local[iat, iat] += tmp

    amat[:, :] += amat_local[:, :]

    amat[1:nat+1, nat+1] = 1.0
    amat[nat+1, 1:nat+1] = 1.0
    amat[nat+1, nat+1] = 0.0



def get_dir_trans(lattice):
   #real(wp), intent(in) :: lattice(:, :)
   #real(wp), allocatable, intent(out) :: trans(:, :)

    rep = np.full(3, 2)
    trans = get_lattice_points_rep_3d(lattice, rep, True)
    return trans

def get_rec_trans(lattice):
   #real(wp), intent(in) :: lattice(:, :)
   #real(wp), allocatable, intent(out) :: trans(:, :)

    rep = np.full(3, 2)
    rec_lat = np.zeros((3, 3))

    rec_lat = (2*math.pi) * np.transpose(np.linalg.inv(lattice))
    trans = get_lattice_points_rep_3d(rec_lat, rep, False)
    return trans


def get_amat_dir_3d(rij, gam, alp, trans):
   #real(wp), intent(in) :: rij(3)
   #real(wp), intent(in) :: gam
   #real(wp), intent(in) :: alp
   #real(wp), intent(in) :: trans(:, :)
   #real(wp), intent(out) :: amat

    vec = np.zeros(3)
    amat = 0.0
    for itr in range(trans.shape[1]):
        vec[:] = rij + trans[itr, :]
        r1 = np.linalg.norm(vec)
        if (r1 < eps):
            continue
        tmp = math.erf(gam*r1)/r1 - math.erf(alp*r1)/r1
        amat += tmp

    return amat

def get_amat_rec_3d(rij, vol, alp, trans):
   #real(wp), intent(in) :: rij(3)
   #real(wp), intent(in) :: gam
   #real(wp), intent(in) :: alp
   #real(wp), intent(in) :: trans(:, :)
   #real(wp), intent(out) :: amat

    vec = np.zeros(3)
    amat = 0.0
    fac = 4*math.pi/vol

    for itr in range(trans.shape[1]):
        vec[:] = rij + trans[itr, :]
        g2 = np.dot(vec, vec)
        if (g2 < eps):
            continue
        tmp = math.cos(np.dot(rij, vec)) * fac * math.exp(-0.25 * g2 / (alp * alp)) / g2
        amat += tmp

    return amat


def get_vrhs(nat, id, kcn, chi, charge, cn, xvec, dxdcn):
   #type(mchrg_model_type), intent(in) :: self
   #type(structure_type), intent(in) :: mol
   #real(wp), intent(in) :: cn(:)
   #real(wp), intent(out) :: xvec(:)
   #real(wp), intent(out), optional :: dxdcn(:)

    reg = 1.0e-14

    if (dxdcn != None):
        for iat in range(nat):
            izp = id[iat]
            tmp = kcn[izp] / math.sqrt(cn[iat] + reg)
            xvec[iat] = -chi[izp] + tmp * cn[iat]
            dxdcn[iat] = 0.5 * tmp
        dxdcn[nat+1] = 0.0
    else:
        for iat in range(nat):
            izp = id[iat]
            tmp = kcn[izp] / math.sqrt(cn[iat] + reg)
            xvec[iat] = -chi[izp] + tmp * cn[iat]

    xvec[nat+1] = charge





eps = math.sqrt(np.finfo(np.float64).eps)

def get_alpha(lattice, alpha):
   #real(wp), intent(in) :: lattice(:, :)
   #real(wp), intent(out) :: alpha

    rec_lat = np.zeros((3, 3))

    vol = abs(np.linalg.det(lattice))
    rec_lat = 2 * math.pi * np.transpose(np.linalg.inv(lattice))

    search_alpha(lattice, rec_lat, vol, eps, alpha)


#!> Get optimal alpha-parameter for the Ewald summation by finding alpha, where
#!> decline of real and reciprocal part of Ewald are equal.
def search_alpha(lattice, rec_lat, volume, tolerance, alpha):
   #!> Lattice vectors
   #real(wp), intent(in) :: lattice(:,:)
   #!> Reciprocal vectors
   #real(wp), intent(in) :: rec_lat(:,:)
   #!> Volume of the unit cell
   #real(wp), intent(in) :: volume
   #!> Tolerance for difference in real and rec. part
   #real(wp), intent(in) :: tolerance
   #!> Optimal alpha
   #real(wp), intent(out) :: alpha

    alpha0 = 1.0e-8
    niter = 30

    rlen = math.sqrt(np.min(np.sum(rec_lat[:, :]**2, axis=0)))
    dlen = math.sqrt(np.min(np.sum(lattice[:, :]**2, axis=0)))

    stat = 0
    alpha = alpha0
    diff = rec_dir_diff(alpha, get_rec_term_3d, rlen, dlen, volume)
    
    while (diff < -tolerance and alpha <= np.finfo(np.float64).max):
        alpha = 2.0 * alpha
        diff = rec_dir_diff(alpha, get_rec_term_3d, rlen, dlen, volume)

    if (alpha > np.finfo(np.float64).max):
        stat = 1
    elif (alpha == alpha0):
        stat = 2

    alpl = 0.0
    if (stat == 0):
        alpl = 0.5 * alpha
        while (diff < tolerance and alpha <= np.finfo(np.float64).max):
            alpha = 2.0 * alpha
            diff = rec_dir_diff(alpha, get_rec_term_3d, rlen, dlen, volume)

        if (alpha > np.finfo(np.float64).max):
            stat = 3

    if (stat == 0):
        alpr = alpha
        alpha = (alpl + alpr) * 0.5
        ibs = 0
        diff = rec_dir_diff(alpha, get_rec_term_3d, rlen, dlen, volume)
        while (abs(diff) > tolerance and ibs <= niter):
            if (diff < 0):
                alpl = alpha
            else:
                alpr = alpha

            alpha = (alpl + alpr) * 0.5
            diff = rec_dir_diff(alpha, get_rec_term_3d, rlen, dlen, volume)
            ibs += 1

        if (ibs > niter):
            stat = 4

    if (stat != 0):
        alpha = 0.25



#!> Returns the max. value of a term in the reciprocal space part of the Ewald
#!> summation for a given vector length.
def get_rec_term_3d(gg, alpha, vol): #returns rval
   #!> Length of the reciprocal space vector
   #real(wp), intent(in) :: gg

   #!> Parameter of the Ewald summation
   #real(wp), intent(in) :: alpha

   #!> Volume of the real space unit cell
   #real(wp), intent(in) :: vol

   #!> Reciprocal term
   #real(wp) :: rval

    return 4.0 * math.pi * (np.exp(-0.25*gg*gg/(alpha**2))/(vol*gg*gg))



#!> Returns the difference in the decrease of the real and reciprocal parts of the
#!> Ewald sum. In order to make the real space part shorter than the reciprocal
#!> space part, the values are taken at different distances for the real and the
#!> reciprocal space parts.
def rec_dir_diff(alpha, get_rec_term, rlen, dlen, volume): # Returns diff
   #!> Parameter for the Ewald summation
   #real(wp), intent(in) :: alpha

   #!> Procedure pointer to reciprocal routine
   #procedure(get_rec_term_gen) :: get_rec_term

   #!> Length of the shortest reciprocal space vector in the sum
   #real(wp), intent(in) :: rlen

   #!> Length of the shortest real space vector in the sum
   #real(wp), intent(in) :: dlen

   #!> Volume of the real space unit cell
   #real(wp), intent(in) :: volume

   #!> Difference between changes in the two terms
   #real(wp) :: diff

    return (get_rec_term(4*rlen, alpha, volume) - get_rec_term(5*rlen, alpha, volume)) - (get_dir_term(2*dlen, alpha) - get_dir_term(3*dlen, alpha))



#!> Returns the max. value of a term in the real space part of the Ewald summation
#!> for a given vector length.
def get_dir_term(rr, alpha): #returns dval
   #!> Length of the real space vector
   #real(wp), intent(in) :: rr

   #!> Parameter of the Ewald summation
   #real(wp), intent(in) :: alpha

   #!> Real space term
   #real(wp) :: dvaerfc(alpha*rr)/rr

    return math.erfc(alpha*rr)/rr




#!> Small cutoff threshold to create only closest cells
thr = math.sqrt(np.finfo(float).eps)

def new_wignerseitz_cell(periodic, lattice):
   #!> Wigner-Seitz cell instance
   #type(wignerseitz_cell_type), intent(out) :: self

   #!> Molecular structure data
   #type(structure_type), intent(in) :: mol

    trans = get_lattice_points_cutoff(periodic, lattice, thr)
    ntr = trans.shape[1]

    nimg = np.zeros((nat, nat))
    tridx = np.zeros((nat, nat, ntr))
    _tridx = np.zeros(ntr)

    vec = np.zeros(3)

    for iat in range(nat):
       for jat in range(nat):
           vec[:] = xyz[iat, :] - xyz[jat, :]
           get_pairs(nimg, trans, vec, _tridx)
           nimg[iat, jat] = nimg
           tridx[iat, jat, :] = _tridx

    return nimg, tridx, trans


#!> Tolerance to consider equivalent images
tol = 0.01

def get_pairs(iws, trans, rij, list1):
   #integer, intent(out) :: iws
   #real(wp), intent(in) :: rij(3)
   #real(wp), intent(in) :: trans(:, :)
   #integer, intent(out) :: list(:)

    mask = np.zeros(len(list1), dtype=np.bool)

    iws = 0
    img = 0
    list1[:] = 0
    mask[:] = True

    vec = np.zeros(3)
    dist = np.zeros(len(list1))

    for itr in range(trans.shape[1]):
        vec[:] = rij - trans[itr, :]
        r2 = vec[0]**2 + vec[1]**2 + vec[2]**2
        if (r2 < thr):
            continue
        img += 1
        dist[img] = r2

    if (img == 0):
        return

    pos = np.argmin(dist[:img], axis=0)

    r2 = dist[pos]
    mask[pos] = False

    iws = 1
    list1[iws] = pos
    if (img <= iws):
        return

    mask_slice = mask[:img]
    dist_slice = dist[:img]
    masked_dist = np.where(mask_slice, dist_slice, np.inf)
    while True:
        pos = np.argmin(masked_dist, axis=0)
        if (abs(dist[pos] - r2) > tol):
            break
        mask[pos] = False
        iws += 1
        list1[iws] = pos


def new_eeq2019_model(num):
    cutoff = 25.0
    cn_exp = 7.5
    cn_max = 8.0

    chi = get_eeq_chi_num(num)
    eta = get_eeq_eta_num(num)
    kcn = get_eeq_kcn_num(num)
    rad = get_eeq_rad_num(num)
    rcov = get_covalent_rad_num(num)

    # This just wraps the values in a class.
    #new_mchrg_model(model, mol, error, chi, rad, eta, kcn, cutoff, cn_exp, rcov, cn_max)

    return rad, chi, eta, kcn, rcov, cutoff, cn_exp, cn_max



#!> Element-specific electronegativity for the electronegativity equilibration charges.
eeq_chi = np.array([
    1.23695041, 1.26590957, 0.54341808, 0.99666991, 1.26691604,
    1.40028282, 1.55819364, 1.56866440, 1.57540015, 1.15056627,
    0.55936220, 0.72373742, 1.12910844, 1.12306840, 1.52672442,
    1.40768172, 1.48154584, 1.31062963, 0.40374140, 0.75442607,
    0.76482096, 0.98457281, 0.96702598, 1.05266584, 0.93274875,
    1.04025281, 0.92738624, 1.07419210, 1.07900668, 1.04712861,
    1.15018618, 1.15388455, 1.36313743, 1.36485106, 1.39801837,
    1.18695346, 0.36273870, 0.58797255, 0.71961946, 0.96158233,
    0.89585296, 0.81360499, 1.00794665, 0.92613682, 1.09152285,
    1.14907070, 1.13508911, 1.08853785, 1.11005982, 1.12452195,
    1.21642129, 1.36507125, 1.40340000, 1.16653482, 0.34125098,
    0.58884173, 0.68441115, 0.56999999, 0.56999999, 0.56999999,
    0.56999999, 0.56999999, 0.56999999, 0.56999999, 0.56999999,
    0.56999999, 0.56999999, 0.56999999, 0.56999999, 0.56999999,
    0.56999999, 0.87936784, 1.02761808, 0.93297476, 1.10172128,
    0.97350071, 1.16695666, 1.23997927, 1.18464453, 1.14191734,
    1.12334192, 1.01485321, 1.12950808, 1.30804834, 1.33689961,
    1.27465977, 1.06598299, 0.68184178, 1.04581665, 1.09888688, 
    1.07206461, 1.09821942, 1.10900303, 1.01039812, 1.00095966,
    1.11003303, 1.16831853, 1.00887482, 1.05928842, 1.07672363,
    1.11308426, 1.14340090, 1.13714110
])


def get_eeq_chi_num(number):
    if (number >= 0 and number < eeq_chi.shape[0]):
        return eeq_chi[number]
    return -1.0



#!> Element-specific chemical hardnesses for the electronegativity equilibration charges.
eeq_eta = np.array([
    -0.35015861, 1.04121227, 0.09281243, 0.09412380, 0.26629137,
     0.19408787, 0.05317918, 0.03151644, 0.32275132, 1.30996037,
     0.24206510, 0.04147733, 0.11634126, 0.13155266, 0.15350650,
     0.15250997, 0.17523529, 0.28774450, 0.42937314, 0.01896455,
     0.07179178,-0.01121381,-0.03093370, 0.02716319,-0.01843812,
    -0.15270393,-0.09192645,-0.13418723,-0.09861139, 0.18338109,
     0.08299615, 0.11370033, 0.19005278, 0.10980677, 0.12327841,
     0.25345554, 0.58615231, 0.16093861, 0.04548530,-0.02478645,
     0.01909943, 0.01402541,-0.03595279, 0.01137752,-0.03697213,
     0.08009416, 0.02274892, 0.12801822,-0.02078702, 0.05284319,
     0.07581190, 0.09663758, 0.09547417, 0.07803344, 0.64913257,
     0.15348654, 0.05054344, 0.11000000, 0.11000000, 0.11000000,
     0.11000000, 0.11000000, 0.11000000, 0.11000000, 0.11000000,
     0.11000000, 0.11000000, 0.11000000, 0.11000000, 0.11000000,
     0.11000000,-0.02786741, 0.01057858,-0.03892226,-0.04574364,
    -0.03874080,-0.03782372,-0.07046855, 0.09546597, 0.21953269,
     0.02522348, 0.15263050, 0.08042611, 0.01878626, 0.08715453,
     0.10500484, 0.10034731, 0.15801991,-0.00071039,-0.00170887, 
    -0.00133327,-0.00104386,-0.00094936,-0.00111390,-0.00125257,
    -0.00095936,-0.00102814,-0.00104450,-0.00112666,-0.00101529, 
    -0.00059592,-0.00012585,-0.00140896
])

def get_eeq_eta_num(number):
    if (number >= 0 and number < eeq_eta.shape[0]):
        return eeq_eta[number]
    return -1.0


#!> Element-specific CN scaling constant for the electronegativity equilibration charges.
eeq_kcn = np.array([
    0.04916110, 0.10937243,-0.12349591,-0.02665108,-0.02631658,
    0.06005196, 0.09279548, 0.11689703, 0.15704746, 0.07987901,
    0.10002962,-0.07712863,-0.02170561,-0.04964052, 0.14250599,
    0.07126660, 0.13682750, 0.14877121,-0.10219289,-0.08979338,
    0.08273597,-0.01754829,-0.02765460,-0.02558926,-0.08010286,
    0.04163215,-0.09369631,-0.03774117,-0.05759708, 0.02431998,
    0.01056270,-0.02692862, 0.07657769, 0.06561608, 0.08006749,
    0.14139200,-0.05351029,-0.06701705,-0.07377246,-0.02927768,
    0.03867291,-0.06929825,-0.04485293,-0.04800824,-0.01484022,
    0.07917502, 0.06619243, 0.02434095,-0.01505548,-0.03030768,
    0.01418235, 0.08953411, 0.08967527, 0.07277771,-0.02129476,
    0.06188828,-0.06568203,-0.11000000,-0.11000000,-0.11000000,
    0.11000000,-0.11000000,-0.11000000,-0.11000000,-0.11000000,
    0.11000000,-0.11000000,-0.11000000,-0.11000000,-0.11000000,
    0.11000000,-0.03585873,-0.03132400,-0.05902379,-0.02827592,
    0.07606260,-0.02123839, 0.03814822, 0.02146834, 0.01580538,
    0.00894298,-0.05864876,-0.01817842, 0.07721851, 0.07936083,
    0.05849285, 0.00013506,-0.00020631, 0.00473118, 0.01590519,
    0.00369763, 0.00417543, 0.00706682, 0.00488679, 0.00505103,
    0.00710682, 0.00463050, 0.00387799, 0.00296795, 0.00400648, 
    0.00548481, 0.01350400, 0.00675380
])

def get_eeq_kcn_num(number):
    if (number >= 0 and number < eeq_kcn.shape[0]):
        return eeq_kcn[number]
    return -1.0


#!> Element-specific charge widths for the electronegativity equilibration charges.
eeq_rad = np.array([
    0.55159092, 0.66205886, 0.90529132, 1.51710827, 2.86070364,
    1.88862966, 1.32250290, 1.23166285, 1.77503721, 1.11955204,
    1.28263182, 1.22344336, 1.70936266, 1.54075036, 1.38200579,
    2.18849322, 1.36779065, 1.27039703, 1.64466502, 1.58859404,
    1.65357953, 1.50021521, 1.30104175, 1.46301827, 1.32928147,
    1.02766713, 1.02291377, 0.94343886, 1.14881311, 1.47080755,
    1.76901636, 1.98724061, 2.41244711, 2.26739524, 2.95378999,
    1.20807752, 1.65941046, 1.62733880, 1.61344972, 1.63220728,
    1.60899928, 1.43501286, 1.54559205, 1.32663678, 1.37644152,
    1.36051851, 1.23395526, 1.65734544, 1.53895240, 1.97542736,
    1.97636542, 2.05432381, 3.80138135, 1.43893803, 1.75505957,
    1.59815118, 1.76401732, 1.63999999, 1.63999999, 1.63999999,
    1.63999999, 1.63999999, 1.63999999, 1.63999999, 1.63999999,
    1.63999999, 1.63999999, 1.63999999, 1.63999999, 1.63999999,
    1.63999999, 1.47055223, 1.81127084, 1.40189963, 1.54015481,
    1.33721475, 1.57165422, 1.04815857, 1.78342098, 2.79106396,
    1.78160840, 2.47588882, 2.37670734, 1.76613217, 2.66172302,
    2.82773085, 1.04059593, 0.60550051, 1.22262145, 1.28736399,
    1.44431317, 1.29032833, 1.41009404, 1.25501213, 1.15181468,
    1.42010424, 1.43955530, 1.28565237, 1.35017463, 1.33011749, 
    1.30745135, 1.26526071, 1.34071499
])

def get_eeq_rad_num(number):
    if (number >= 0 and number < eeq_rad.shape[0]):
        return eeq_rad[number]
    return -1.0

## Values from 2018 CODATA NIST file ##

#!> Planck's constant
planck_constant = 6.62607015e-34
h = planck_constant

#!> Speed of light in vacuum
speed_of_light_in_vacuum = 299792458e0
c = speed_of_light_in_vacuum

#!> electron rest mass
electron_mass = 9.1093837015e-31
me = electron_mass

#!> fine structure constant (CODATA2018)
fine_structure_constant = 7.2973525693e-3
alpha = fine_structure_constant

#######################################


#!> Reduced Planck's constant
hbar = h/(2.0*math.pi) # J·s = kg·m²·s⁻¹

#!> Bohr radius
bohr = hbar/(me*c*alpha) # m

#!> Conversion factor from bohr to Ångström
autoaa = bohr * 1e10

#!> Conversion factor from Ångström to bohr
aatoau = 1.0/autoaa

#!> Covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009,
#!> 188-197), values for metals decreased by 10 %
covalent_rad_2009 = aatoau * np.array([
0.32,0.46, # H,He
1.20,0.94,0.77,0.75,0.71,0.63,0.64,0.67, # Li-Ne
1.40,1.25,1.13,1.04,1.10,1.02,0.99,0.96, # Na-Ar
1.76,1.54, # K,Ca
                1.33,1.22,1.21,1.10,1.07, # Sc-
                1.04,1.00,0.99,1.01,1.09, # -Zn
                1.12,1.09,1.15,1.10,1.14,1.17, # Ga-Kr
1.89,1.67, # Rb,Sr
                1.47,1.39,1.32,1.24,1.15, # Y-
                1.13,1.13,1.08,1.15,1.23, # -Cd
                1.28,1.26,1.26,1.23,1.32,1.31, # In-Xe
2.09,1.76, # Cs,Ba
        1.62,1.47,1.58,1.57,1.56,1.55,1.51, # La-Eu
        1.52,1.51,1.50,1.49,1.49,1.48,1.53, # Gd-Yb
                1.46,1.37,1.31,1.23,1.18, # Lu-
                1.16,1.11,1.12,1.13,1.32, # -Hg
                1.30,1.30,1.36,1.31,1.38,1.42, # Tl-Rn
2.01,1.81, # Fr,Ra
     1.67,1.58,1.52,1.53,1.54,1.55,1.49, # Ac-Am
     1.49,1.51,1.51,1.48,1.50,1.56,1.58, # Cm-No
                1.45,1.41,1.34,1.29,1.27, # Lr-
                1.21,1.16,1.15,1.09,1.22, # -Cn
                1.36,1.43,1.46,1.58,1.48,1.57  # Nh-Og
])

#!> D3 covalent radii used to construct the coordination number
covalent_rad_d3 = np.array([
    4.0 / 3.0 * covalent_rad_2009
])

#!> Get covalent radius for a given atomic number
def get_covalent_rad_num(num):
   #!> Atomic number
   #integer, intent(in) :: num

   #!> Covalent radius
   #real(wp) :: rad

    if (num >= 0 and num < len(covalent_rad_d3)):
        return covalent_rad_d3[num]
    return  0.0



#!> Calculate the weights of the reference system and the derivatives w.r.t.
#!> coordination number for later use.
def weight_references(zeff, eta, ga, gc, wf, ngw, cn, d4_cn, q, gwvec, gwdcn, gwdq):
    #!> Instance of the dispersion model
    #class(d4_model), intent(in) :: self

    #!> Molecular structure data
    #class(structure_type), intent(in) :: mol

    #!> Coordination number of every atom
    #real(wp), intent(in) :: cn(:)

    #!> Partial charge of every atom
    #real(wp), intent(in) :: q(:)

    #!> weighting for the atomic reference systems
    #real(wp), intent(out) :: gwvec(:, :, :)

    #!> derivative of the weighting function w.r.t. the coordination number
    #real(wp), intent(out), optional :: gwdcn(:, :, :)

    #!> derivative of the weighting function w.r.t. the charge scaling
    #real(wp), intent(out), optional :: gwdq(:, :, :)

    if (gwdcn != None and gwdq != None):
        gwvec[:, :, :] = 0.0
        gwdcn[:, :, :] = 0.0
        gwdq[:, :, :] = 0.0

        for iat in range(nat):
            izp = id[iat]
            zi = zeff[izp]
            gi = eta[izp] * gc
            norm = 0.0
            dnorm = 0.0

            for iref in range(ref[izp]):
                for igw in range(ngw[izp, iref]):
                    wf = igw * wf
                    gw = weight_cn(wf, cn[iat], d4_cn[izp, iref])
                    norm += gw
                    dnorm += 2*wf * (d4_cn[izp, iref] - cn[iat]) * gw

            norm = 1.0 / norm

            for iref in range(ref[izp]):
                expw = 0.0
                expd = 0.0
                for igw in range(ngw[izp, iref]):
                    wf = igw * wf
                    gw = weight_cn(wf, cn[iat], d4_cn[izp, iref])
                    expw += gw
                    expd += 2*wf * (d4_cn[izp, iref] - cn[iat]) * gw

                gwk = expw * norm
                if (is_exceptional(gwk)):
                    maxcn = np.max(d4_cn[izp, :ref[izp]])
                    if (abs(maxcn - d4_cn[izp, iref]) < 1e-12):
                        gwk = 1.0
                    else:
                        gwk = 0.0

                gwvec[1, iat, iref] = gwk * zeta(ga, gi, q[izp, iref]+zi, q[iat]+zi)
                gwdq[1, iat, iref] = gwk * dzeta(ga, gi, q[izp, iref]+zi, q[iat]+zi)

                dgwk = norm * (expd - expw * dnorm * norm)
                if (is_exceptional(dgwk)):
                    dgwk = 0.0

                gwdcn[1, iat, iref] = dgwk * zeta(ga, gi, q[izp, iref]+zi, q[iat]+zi)

    else:
        gwvec[:, :, :] = 0.0

        for iat in range(nat):
            izp = id[iat]
            zi = zeff[izp]
            gi = eta[izp] * gc
            norm = 0.0
            for iref in range(ref[izp]):
                for igw in range(ngw[izp, iref]):
                    wf = igw * wf
                    norm += weight_cn(wf, cn[iat], d4_cn[izp, iref])

            norm = 1.0 / norm
            for iref in range(ref[izp]):
                expw = 0.0
                for igw in range(ngw[izp, iref]):
                    wf = igw * wf
                    expw += weight_cn(wf, cn[iat], d4_cn[izp, iref])

                gwk = expw * norm
                if (is_exceptional(gwk)):
                    maxcn = np.max(d4_cn[izp, :ref[izp]])
                    if (abs(maxcn - d4_cn[izp, iref]) < 1e-12):
                        gwk = 1.0
                    else:
                        gwk = 0.0

                gwvec[1, iat, iref] = gwk * zeta(ga, gi, q[izp, iref]+zi, q[iat]+zi)




def weight_cn(wf, cn, cnref):
    return math.exp(-wf * (cn - cnref)**2)

def is_exceptional(val):
    import sys
    return math.isnan(val) or abs(val) > sys.float_info.max


#!> Create lattice points within a given cutoff
def get_lattice_points_cutoff(periodic, lat, rthr):
    #!> Periodic dimensions
    #logical, intent(in) :: periodic(:)

    #!> Real space cutoff
    #real(wp), intent(in) :: rthr

    #!> Lattice parameters
    #real(wp), intent(in) :: lat(:, :)

    #!> Generated lattice points
    #real(wp), allocatable, intent(out) :: trans(:, :)

    if (not np.any(periodic)):
        trans = np.zeros((1, 3))
        trans[:, :] = 0.0
    else:
        rep = get_translations(lat, rthr)
        trans = get_lattice_points_rep_3d(lat, rep, True)

    return trans


#!> Generate lattice points from repeatitions
def get_lattice_points_rep_3d(lat, rep, origin):
    #!> Lattice vectors
    #real(wp), intent(in) :: lat(:, :)

    #!> Repeatitions of lattice points to generate
    #integer, intent(in) :: rep(:)

    #!> Include the origin in the generated lattice points
    #logical, intent(in) :: origin

    #!> Generated lattice points
    #real(wp), allocatable, intent(out) :: trans(:, :)

    itr = 0
    if (origin):
        trans = np.zeros((np.prod(2*rep+1), 3))
        for ix in range(rep[0]+1):
            for iy in range(rep[1]+1):
                for iz in range(rep[2]+1):
                    for jx in range(1, int(np.where(ix > 0, -1, 1)) - 1, -2):
                        for jy in range(1, int(np.where(iy > 0, -1, 1)) - 1, -2):
                            for jz in range(1, int(np.where(iz > 0, -1, 1)) - 1, -2):
                                itr += 1
                                trans[itr, :] = lat[1, :] * ix * jx + lat[2, :] * iy * jy + lat[3, :] * iz * jz
    else:
        trans = np.zeros((np.prod(2*rep+1)-1, 3))
        for ix in range(rep[0]+1):
            for iy in range(rep[1]+1):
                for iz in range(rep[2]+1):
                    if (ix == 0 and iy == 0 and iz == 0):
                        continue
                    for jx in range(1, int(np.where(ix > 0, -1, 1)) - 1, -2):
                        for jy in range(1, int(np.where(iy > 0, -1, 1)) - 1, -2):
                            for jz in range(1, int(np.where(iz > 0, -1, 1)) - 1, -2):
                                itr += 1
                                trans[itr, :] = lat[1, :] * ix * jx + lat[2, :] * iy * jy + lat[3, :] * iz * jz

    return trans


#!> Generate a supercell based on a realspace cutoff, this subroutine
#!> doesn't know anything about the convergence behaviour of the
#!> associated property.
def get_translations(lat , rthr):
   #! find normal to the plane...
   normx = crossproduct(lat[2, :], lat[3, :])
   normy = crossproduct(lat[3, :], lat[1, :])
   normz = crossproduct(lat[1, :], lat[2, :])
   #! ...normalize it...
   normx = normx/np.linalg.norm(normx)
   normy = normy/np.linalg.norm(normy)
   normz = normz/np.linalg.norm(normz)
   #! cos angles between normals and lattice vectors
   cos10 = sum(normx*lat[1, :])
   cos21 = sum(normy*lat[2, :])
   cos32 = sum(normz*lat[3, :])

   rep = np.zeros(3)
   rep[0] = np.ceil(abs(rthr/cos10))
   rep[1] = np.ceil(abs(rthr/cos21))
   rep[2] = np.ceil(abs(rthr/cos32))
   return rep


def crossproduct(a, b):
    c = np.zeros(3)
    c[0] = a[1]*b[2]-b[1]*a[2]
    c[1] = a[2]*b[0]-b[2]*a[0]
    c[2] = a[0]*b[1]-b[0]*a[1]
    return c


#subroutine get_coordination_number(mol, trans, cutoff, rcov, en, cn, dcndr, dcndL)
#!> Geometric fractional coordination number, supports error function counting.
#def get_coordination_number(mol, trans, cutoff, rcov, en, cn, dcndr, dcndL):
#    new_ncoord(ncoord, mol, cn_count%dftd4, kcn=default_kcn, cutoff, rcow, en, error):
#    if error:
#        print error
#        error stop (exit?)
#
#    ncoord%get_coordination_number(cut, mol, trans, cn, dcndr, dcndL)



#!> Geometric fractional coordination number
def get_coordination_number(cut, nat, id, xyz, trans, rcov, kcn, norm_exp):
    #!> Coordination number container
    #class(ncoord_type), intent(in) :: self
    #
    #!> Molecular structure data
    #type(structure_type), intent(in) :: mol
    #
    #!> Lattice points
    #real(wp), intent(in) :: trans(:, :)
    #
    #!> Error function coordination number.
    #real(wp), intent(out) :: cn(:)
    #
    #!> Derivative of the CN with respect to the Cartesian coordinates.
    #real(wp), intent(out), optional :: dcndr(:, :, :)
    #
    #!> Derivative of the CN with respect to strain deformations.
    #real(wp), intent(out), optional :: dcndL(:, :, :)

    #if (dcndr != None and dcndL != None):
    #    cn, dcndr, dcdnL = ncoord_d(rcov, kcn, norm_exp, cut, nat, id, xyz, trans)
    #else:
    #    cn = ncoord(rcov, kcn, norm_exp, trans)

    cn, dcndr, dcndL = ncoord_d(rcov, kcn, norm_exp, cut, nat, id, xyz, trans)

    if (cut > 0.0):
        cn = cut_coordination_number(cut, cn, dcndr, dcndL)

    return cn, dcndr, dcndL





def ncoord(rcov, kcn, norm_exp, trans):
    #!> Coordination number container
    #class(ncoord_type), intent(in) :: self
    #!> Molecular structure data
    #type(structure_type), intent(in) :: mol
    #!> Lattice points
    #real(wp), intent(in) :: trans(:, :)
    #!> Error function coordination number.
    #real(wp), intent(out) :: cn(:)

    cutoff2 = realspace_cutoff_cn**2

    cn_local = np.zeros(nat)
    for iat in range(nat):
        izp = id[iat]
        for jat in range(iat):
            jzp = id[jat]
            den = get_en_factor(izp, jzp)

            for itr in range(trans.shape[1]):
                rij = xyz[iat, :] - (xyz[jat, :] + trans[itr, :])
                r2 = np.sum(rij**2)
                if (r2 > cutoff2 or r2 < 1e-12):
                    continue
                r1 = math.sqrt(r2)

                countf = den * ncoord_count(rcov, kcn, norm_exp, izp, jzp, r1)

                cn_local[iat] += countf
                if (iat != jat):
                    cn_local[jat] += countf * directed_factor

    return cn_local



# TODO: There are 3 different ncoord_count. I don't know which is used. Might need to try the other ones.
#!> Error counting function for coordination number contributions.
def ncoord_count(rcov, kcn, norm_exp, izp, jzp, r):
    #!> Coordination number container
    #class(erf_ncoord_type), intent(in) :: self
    #!> Atom i index
    #integer, intent(in)  :: izp
    #!> Atom j index
    #integer, intent(in)  :: jzp
    #!> Current distance.
    #real(wp), intent(in) :: r

    rc = (rcov[izp] + rcov[jzp])
    count = 0.5 * (1.0 + math.erf(-kcn * (r-rc)/rc)**norm_exp)
    return count

# TODO: Probably 3 different versions here as well
def ncoord_dcount(rcov, kcn, norm_exp, izp, jzp, r):
    rc = rcov[izp] + rcov[jzp]

    exponent = kcn * (r-rc)/rc**norm_exp
    expterm = math.exp(-exponent**2.0)
    count = -(kcn * expterm)/(math.sqrt(math.pi)*rc**norm_exp)
    return count


#!> Factor determining whether the CN is evaluated with direction
#!> if +1 the CN contribution is added equally to both partners
#!> if -1 (i.e. with the EN-dep.) it is added to one and subtracted from the other
#real(wp)  :: directed_factor
directed_factor = 1.0


#subroutine ncoord_d(self, mol, trans, cn, dcndr, dcndL)
#   !> Coordination number container
#   class(ncoord_type), intent(in) :: self
#   !> Molecular structure data
#   type(structure_type), intent(in) :: mol
#   !> Lattice points
#   real(wp), intent(in) :: trans(:, :)
#   !> Error function coordination number.
#   real(wp), intent(out) :: cn(:)
#   !> Derivative of the CN with respect to the Cartesian coordinates.
#   real(wp), intent(out) :: dcndr(:, :, :)
#   !> Derivative of the CN with respect to strain deformations.
#   real(wp), intent(out) :: dcndL(:, :, :)

#   integer :: iat, jat, izp, jzp, itr
#   real(wp) :: r2, r1, rij(3), countf, countd(3), sigma(3, 3), cutoff2, den

#   ! Thread-private arrays for reduction
#   real(wp), allocatable :: cn_local(:)
#   real(wp), allocatable :: dcndr_local(:, :, :), dcndL_local(:, :, :)
def ncoord_d(rcov, kcn, norm_exp, cutoff, nat, id, xyz, trans):
    cn = np.zeros(xyz.shape[0]) # I think we need 1 coordination number for each atom?
    #dcndr[:, :, :] = 0.0    # TODO: Find out how large dcndr and dcndL should be
    #dcndL[:, :, :] = 0.0
    dcndr = np.zeros((xyz.shape[0], 3, 1))
    dcndL = np.zeros((xyz.shape[0], 3, 1))
    cutoff2 = cutoff**2

    # NOTE: I think all the local arrays are just used for openmp reduction
    cn_local = np.array(cn, copy=True)
    dcndr_local = np.array(dcndr, copy=True)
    dcndL_local = np.array(dcndL, copy=True)

    for iat in range(1, nat+1):
        izp = id[iat]
        for jat in range(1, iat+1):
            jzp = id[jat]
            den = get_en_factor(jzp, izp)

            for itr in range(1, trans.shape[1]+1):
                rij = xyz[iat, :] - (xyz[jat, :] + trans[itr, :])
                r2 = np.sum(rij**2)
                if (r2 > cutoff2 or r2 < 1e-12):
                    continue
                r1 = math.sqrt(r2)

                countf = den * ncoord_count(rcov, kcn, norm_exp, izp, jzp, r1)
                countd = den * ncoord_dcount(rcov, kcn, norm_exp, izp, jzp, r1) * rij/r1

                cn_local[iat] += countf
                if (iat != jat):
                    cn_local[jat] += countf * directed_factor

                dcndr_local[iat, iat, :] += countd
                dcndr_local[jat, jat, :] -= countd * directed_factor
                dcndr_local[jat, iat, :] += countd * directed_factor
                dcndr_local[iat, jat, :] -= countd

                sigma = np.outer(rij, countd)
                dcndL_local[iat, :, :] += sigma
                if (iat != jat):
                    dcndL_local[jat, :, :] += sigma * directed_factor

    cn[:] += cn_local[:]
    dcndr[:, :, :] += dcndr_local[:, :, :]
    dcndL[:, :, :] += dcndL_local[:, :, :]

    return cn, dcndr, dcndL


# NOTE: Bro fr just returns 1. wtf
#!> Evaluates pairwise electronegativity factor if non applies
def get_en_factor(izp, jzp):
    return 1.0


# !> Cutoff function for large coordination numbers
def cut_coordination_number(cn_max, cn, dcndr, dcndL):
    #!> Maximum CN (not strictly obeyed)
    #real(wp), intent(in) :: cn_max

    #!> On input coordination number, on output modified CN
    #real(wp), intent(inout) :: cn(:)

    #!> On input derivative of CN w.r.t. cartesian coordinates,
    #!> on output derivative of modified CN
    #real(wp), intent(inout), optional :: dcndr(:, :, :)

    #!> On input derivative of CN w.r.t. strain deformation,
    #!> on output derivative of modified CN
    #real(wp), intent(inout), optional :: dcndL(:, :, :)

    # NOTE: This function has some checks for whether args are set.
    # For now we only have 1 call with all args, so the checks are removed
    # This is also true for some of the other functions, but forgot to document it

    for iat in range(len(cn)):
        dcnpdcn = math.exp(cn_max) / (math.exp(cn_max) + math.exp(cn[iat]))
        dcndL[iat, :, :] = dcnpdcn * dcndL[iat, :, :]

    for iat in range(len(cn)):
        dcnpdcn = math.exp(cn_max) / (math.exp(cn_max) + math.exp(cn[iat]))
        dcndL[iat, :, :] = dcnpdcn * dcndr[iat, :, :]

    for iat in range(len(cn)):
        cn[iat] = math.log(1.0 + math.exp(cn_max)) - math.log(1.0 + math.exp(cn_max - cn))


#!> Create a new dftd4 coordination number container
#subroutine new_erf_dftd4_ncoord(self, mol, kcn, cutoff, rcov, en, cut, norm_exp)
#   !> Coordination number container
#   type(erf_dftd4_ncoord_type), intent(out) :: self
#   !> Molecular structure data
#   type(structure_type), intent(in) :: mol
#   !> Steepness of counting function
#   real(wp), optional :: kcn
#   !> Real space cutoff
#   real(wp), intent(in), optional :: cutoff
#   !> Covalent radii
#   real(wp), intent(in), optional :: rcov(:)
#   !> Electronegativity
#   real(wp), intent(in), optional :: en(:)
#   !> Cutoff for the maximum coordination number
#   real(wp), intent(in), optional :: cut
#   !> Exponent of the distance normalization
#   real(wp), intent(in), optional :: norm_exp
def new_erf_dftd4_ncoord(mol, rcov, en, cut, kcn=7.5, cutoff=25.0, norm_exp=1.0):
    # This function just collects all the args in a class object

    return NotImplemented



from lapack import mchrg_dsytri, mchrg_dsytrf
from xyz_reader import parse_xyz_with_symbols
symbols, positions = parse_xyz_with_symbols("./caffeine.xyz")
nat, id, xyz, nid, map, num, sym, _lattice, _periodic = new_structure(positions, symbols)
c6, ref = new_d4_model(nid, num)

#from energy import GFN2_coordination_numbers_np
#from xyz_reader import parse_xyz
#element_ids, positions = parse_xyz("./caffeine.xyz")
#cns = GFN2_coordination_numbers_np(element_ids, positions)
#print(f"coordination numbers: {cns}")
