import math
import numpy as np
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

def C6_AB():
    a = 3 / math.pi
    for j in range(2,23):
        return NotImplementedError

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

# Get chemical identity from a list of element symbols
# Mutates identity and returns nid
#def get_identity_symbol(identity, symbol):
#    nat = len(identity)
#    stmp = []
#    for iat in range(nat):
#        if symbol[iat] in stmp:
#            identity[iat] = stmp.index(symbol[iat])
#            continue
#        stmp.append(symbol[iat])
#        identity[iat] = len(stmp)-1
#    # Number of unique species
#    nid = len(stmp)
#    return nid


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

    return _nat, _id, _xyz, _nid, _map, _num, _sym

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
def new_d4_model(symbols, positions, ga=3.0, gc=2.0, wf=6.0):
    nat, id, xyz, nid, map, num, sym = new_structure(positions, symbols)

    ref = np.zeros(nid, dtype=np.int32)
    for isp in range(nid):
        izp = num[isp]
        ref[isp] = get_nref_num(izp)

    mref = np.max(ref)

    d4_aiw = np.zeros((nid, mref, 23))
    for isp in range(nid):
        izp = num[isp]
        set_refalpha_gfn2_num(d4_aiw[isp, :, :], ga, gc, izp)

    aiw = np.zeros(23)
    d4_c6 = np.zeros((nid, nid, mref, mref))
    for isp in range(nid):
        izp = num[isp]
        for jsp in range(isp+1):
            for iref in range(ref[isp]):
                for jref in range(ref[jsp]):
                    aiw[:] = d4_aiw[isp, iref, :] * d4_aiw[jsp, jref, :]
                    c6 = thopi * trapzd(aiw)
                    d4_c6[isp, jsp, iref, jref] = c6
                    d4_c6[jsp, isp, jref, iref] = c6

    return d4_c6
