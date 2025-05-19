import math
import numpy as np

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

max_elem = 118
refn = np.zeros(max_elem)
refh = np.zeros((7, max_elem), dtype=np.float64)
refsys = np.zeros((7, max_elem))
hcount = np.zeros((7, max_elem), dtype=np.float64)
ascale = np.zeros((7, max_elem), dtype=np.float64)
sscale = np.zeros(17, dtype=np.float64)
secaiw = np.zeros((23, 17), dtype=np.float64)
alphaiw = np.zeros((23, 17, max_elem), dtype=np.float64)
# https://github.com/dftd4/dftd4/blob/502d7c59bf88beec7c90a71c4ecf80029794bd5e/src/dftd4/reference.f90#L285
# Set the reference polarizibility for an atomic number
def set_refalpha_eeq_num(alpha, ga, gc, atomic_number):
    aiw = np.zeros(23)

    if (atomic_number > 0 and atomic_number <= len(refn)):
        ref = int(refn[atomic_number])
        for ir in range(ref):
            _is = refsys[atomic_number, ir] # NOTE: Bro where is refsys populated frfr?
            if (abs(_is) < 1e-12):
                continue

            iz = get_effective_charge_num(atomic_number)
            aiw = sscale[_is] * secaiw[_is, :] * zeta(ga, get_hardness_num(_is)*gc, iz, refh[atomic_number, ir]+iz)  # NOTE: Bruh where is sscale and refh populated :sob:
            alpha[ir, :] = max(ascale[atomic_number, ir] * (alphaiw[atomic_number, ir, :] - hcount[atomic_number, ir] * aiw), 0.0) # NOTE: Where is ascale, alphaiw and hcount populated?

def zeta(a, c, qref, qmod):
    if (qmod < 0.0):
        return math.exp(a)
    return math.exp(a * (1.0 - math.exp(c * (1.0 - qref/qmod))))

def get_effective_charge_num(atomic_number):
    if (atomic_number > 0 and atomic_number <= len(effective_nuclear_charge)):
        return effective_nuclear_charge[atomic_number]
    return 0.0

def get_hardness_num(atomic_number):
    if (atomic_number > 0 and atomic_number <= len(chemical_hardness)):
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
])


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
