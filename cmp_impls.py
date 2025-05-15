import numpy as np
from energy import build_SDQH0, dtrf2, form_product, get_multiints, h0scal, horizontal_shift, multipole_3d, olapp
import glob
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compare original xtb results with our own python implementation.")
parser.add_argument("directory", help="Path to the directory containing the binary files from xtb containing the arguments and result.")
args = parser.parse_args()

directory = args.directory
directory = os.path.abspath(directory)


def compare(fo, py, label, force_equal=False):
    if py.shape != fo.shape:
        print(f"{label}:")
        print(f"\tShape missmatch\nPython: {py.shape}\nFortran: {fo.shape}")
        return
    equal = np.array_equal(py,fo)
    if equal:
        return
    close = np.allclose(py,fo)
    if close and not force_equal:
        return
    print("\033[0;31m", end='')
    if close:
        print(f"{label}: Is close but not equal!")
    else:
        print(f"{label}: Is not close!")
    print("\033[0;0m", end='')
    print(f"\tPython: \n{py}")
    print(f"\tFortran: \n{fo}")
    diff = ""
    diff_arr = py - fo
    threshold = 0 if force_equal else 1e-08 
    max_before_newline = 5
    shape = py.shape
    last_index = ()
    count = 0
    closing = py.ndim
    for idx in np.ndindex(shape):
        if py.ndim > 1:
            if last_index:
                for dim in range(0,py.ndim-1):
                    closing += idx[dim] != last_index[dim]
                if closing > 0:
                    diff += f"{']'*closing}{' '*(py.ndim-closing)}"
                    count = 0
                print(idx, closing)
            last_index = idx
        diff += f'{"\n"*(closing>0)}{" "*(py.ndim-closing)}{"["*closing}'
        if count == max_before_newline:
            diff += "\n"
            count = 0
        closing = 0
        diff_val = diff_arr[idx]
        color = 0
        if np.abs(diff_val) >= threshold:
            color = 32
            if diff_val < 0:
                color = 31
        pos_space = " " if py[idx] >= 0 else ""
        diff += f" \033[0;{color}m{f'{pos_space}{py[idx]}': <32}\033[0;0m"

        # Add line breaks when a major axis changes (e.g., new row in 2D, new matrix in 3D)
        count += 1
    diff += "]"*py.ndim
    print(f"\tDiff: \n{diff}")
    return


def test_get_multiints():
    for file_path in glob.glob(f'{directory}/get_multiints/*.bin'):  # Matches all .bin files in the directory
        with open(file_path, 'rb') as f:
            def read_ints(n=1):
                return np.fromfile(f, dtype=np.int32, count=n)

            def read_reals(n=1):
                return np.fromfile(f, dtype=np.float64, count=n)  # match `real(wp)`

            icao, jcao, naoi, naoj, ishtyp, jshtyp = read_ints(6)
            ri = read_reals(3)
            rj = read_reals(3)
            point = read_reals(3)
            intcut = read_reals(1)[0]
            n_nprim = read_ints(1)[0]
            nprim = read_ints(n_nprim)
            n_primcount = read_ints(1)[0]
            primcount = read_ints(n_primcount)
            n_alp = read_ints(1)[0]
            alp = read_reals(n_alp)
            n_cont = read_ints(1)[0]
            cont = read_reals(n_cont)

            # Read ss dimensions
            m, n = read_ints(2)
            fss = np.fromfile(f, dtype=np.float64, count=m * n).reshape((n, m))

            # Read dd dimensions
            d1, d2, d3 = read_ints(3)
            fdd = np.fromfile(f, dtype=np.float64, count=d1 * d2 * d3).reshape((d3, d2, d1))

            # Read qq dimensions
            q1, q2, q3 = read_ints(3)
            fqq = np.fromfile(f, dtype=np.float64, count=q1 * q2 * q3).reshape((q3, q2, q1))

            ss,dd,qq = get_multiints(icao, jcao, naoi, naoj, ishtyp, jshtyp, ri, rj, point, intcut, nprim, primcount, alp, cont)


            ss_equal = np.allclose(ss, fss)
            if (not ss_equal):
                print("Fortran:")
                print("ss: ", fss)

                print("Python:")
                print("ss: ", ss)

                print("\033[1;31m")
                assert ss_equal, "ss matrices do not match"

            dd_equal = np.allclose(dd, fdd)
            if (not dd_equal):
                print("Fortran:")
                print("dd: ", fdd)

                print("Python:")
                print("dd: ", dd)

                print("\033[1;31m")
                assert dd_equal, "dd matrices do not match"

            qq_equal = np.allclose(qq, fqq)
            if (not qq_equal):
                print("Fortran:")
                print("qq: ", fqq)

                print("Python:")
                print("qq: ", qq)

                print("\033[1;31m")
                assert qq_equal, "[get_multiints] qq matrices do not match"

    print("\033[0;32m", end='')
    print("matches! [get_multiints]")
    print("\033[0;0m", end='')



def test_dtrf2():
    for file_path in glob.glob(f'{directory}/dtrf2/*.bin'):  # Matches all .bin files in the directory
        with open(file_path, 'rb') as f:
            def read_ints(n=1):
                return np.fromfile(f, dtype=np.int32, count=n)

            s1, s2 = read_ints(2)
            s = np.fromfile(f, dtype=np.float64, count=s1 * s2).reshape((s2, s1))

            li = read_ints(1)[0]
            lj = read_ints(1)[0]

            s1_res, s2_res = read_ints(2)
            s_res = np.fromfile(f, dtype=np.float64, count=s1_res * s2_res).reshape((s2_res, s1_res))

            dtrf2(s, li, lj)

            s_equal = np.array_equal(s, s_res)
            if (not s_equal):
                print("Fortran:")
                print("s: ", s_res)

                print("Python:")
                print("s: ", s)

                print("\033[1;31m")
                assert s_equal, "[dtrf2] s matrices do not match"

    print("\033[0;32m", end='')
    print("matches! [dtrf2]")
    print("\033[0;0m", end='')


def test_form_product():
    for file_path in glob.glob(f'{directory}/form_product/*.bin'):  # Matches all .bin files in the directory
        with open(file_path, 'rb') as f:
            def read_ints(n=1):
                return np.fromfile(f, dtype=np.int32, count=n)

            la, lb = read_ints(2)

            a1 = read_ints(1)[0]
            a = np.fromfile(f, dtype=np.float64, count=a1)

            b1 = read_ints(1)[0]
            b = np.fromfile(f, dtype=np.float64, count=b1)

            d1 = read_ints(1)[0]
            d = np.fromfile(f, dtype=np.float64, count=d1)

            d1_res = read_ints(1)[0]
            d_res = np.fromfile(f, dtype=np.float64, count=d1_res)

            form_product(a, b, la, lb, d)

            d_equal = np.array_equal(d, d_res)
            if (not d_equal):
                print("Fortran:")
                print("d: ", d_res)

                print("Python:")
                print("d: ", d)

                print("\033[1;31m")
                assert d_equal, "[form_product] d matrices do not match"

    print("\033[0;32m", end='')
    print("matches! [form_product]")
    print("\033[0;0m", end='')


def test_horizontal_shift():
    for file_path in glob.glob(f'{directory}/horizontal_shift/*.bin'):  # Matches all .bin files in the directory
        with open(file_path, 'rb') as f:
            def read_ints(n=1):
                return np.fromfile(f, dtype=np.int32, count=n)

            def read_reals(n=1):
                return np.fromfile(f, dtype=np.float64, count=n)  # match `real(wp)`

            ae = read_reals(1)[0]
            l = read_ints(1)[0]

            cfs1 = read_ints(1)[0]
            cfs = np.fromfile(f, dtype=np.float64, count=cfs1)

            cfs_res1 = read_ints(1)[0]
            cfs_res = np.fromfile(f, dtype=np.float64, count=cfs_res1)

            horizontal_shift(ae, l, cfs)

            cfs_equal = np.array_equal(cfs, cfs_res)
            if (not cfs_equal):
                print("Fortran:")
                print("cfs: ", cfs_res)

                print("Python:")
                print("cfs: ", cfs)

                print("\033[1;31m")
                assert cfs_equal, "[horizontal_shift] cfs matrices do not match"

    print("\033[0;32m", end='')
    print("matches! [horizontal_shift]")
    print("\033[0;0m", end='')


def test_multipole_3d():
    for file_path in glob.glob(f'{directory}/multipole_3d/*.bin'):
        with open(file_path, 'rb') as f:
            def read_ints(n=1):
                return np.fromfile(f, dtype=np.int32, count=n)

            ri1 = read_ints(1)[0]
            ri = np.fromfile(f, dtype=np.float64, count=ri1)

            rj1 = read_ints(1)[0]
            rj = np.fromfile(f, dtype=np.float64, count=rj1)

            rc1 = read_ints(1)[0]
            rc = np.fromfile(f, dtype=np.float64, count=rc1)

            rp1 = read_ints(1)[0]
            rp = np.fromfile(f, dtype=np.float64, count=rp1)

            li1 = read_ints(1)[0]
            li = np.fromfile(f, dtype=np.int32, count=li1)

            lj1 = read_ints(1)[0]
            lj = np.fromfile(f, dtype=np.int32, count=lj1)

            s1d1 = read_ints(1)[0]
            s1d = np.fromfile(f, dtype=np.float64, count=s1d1)

            s3d1 = read_ints(1)[0]
            s3d = np.fromfile(f, dtype=np.float64, count=s3d1)

            s3d_res1 = read_ints(1)[0]
            s3d_res = np.fromfile(f, dtype=np.float64, count=s3d_res1)


            multipole_3d(ri, rj, rc, rp, li, lj, s1d, s3d)

            s3d_equal = np.array_equal(s3d, s3d_res)
            if (not s3d_equal):
                print("Fortran:")
                print("s3d: ", s3d_res)

                print("Python:")
                print("s3d: ", s3d)

                print("\033[1;31m")
                assert s3d_equal, "[multipole_3d] s3d matrices do not match"

    print("\033[0;32m", end='')
    print("matches! [multipole_3d]")
    print("\033[0;0m", end='')



def test_olapp():
    for file_path in glob.glob(f'{directory}/olapp/*.bin'):
        with open(file_path, 'rb') as f:
            def read_ints(n=1):
                return np.fromfile(f, dtype=np.int32, count=n)

            def read_reals(n=1):
                return np.fromfile(f, dtype=np.float64, count=n)

            l1 = read_ints(1)[0]
            l = np.fromfile(f, dtype=np.int32, count=l1)

            gama = read_reals(1)[0]

            s_res1 = read_ints(1)[0]
            s_res = np.fromfile(f, dtype=np.float64, count=s_res1)

            s = olapp(l, gama)

            s_equal = np.array_equal(s, s_res)
            if (not s_equal):
                print("Fortran:")
                print("s: ", s_res)

                print("Python:")
                print("s: ", s)

                print("\033[1;31m")
                assert s_equal, "[olapp] s reals do not match"

    print("\033[0;32m", end='')
    print("matches! [olapp]")
    print("\033[0;0m", end='')



def test_h0scal():
    for file_path in glob.glob(f'{directory}/h0scal/*.bin'):
        with open(file_path, 'rb') as f:
            def read_ints(n=1):
                return np.fromfile(f, dtype=np.int32, count=n)

            def read_reals(n=1):
                return np.fromfile(f, dtype=np.float64, count=n)

            def read_logicals(n=1):
                return np.fromfile(f, dtype=np.int32, count=n).astype(bool)

            il, jl = read_ints(2)
            izp, jzp = read_ints(2)
            valaoi, valaoj = read_logicals(2)

            km_res = read_reals(1)[0]

            km = h0scal(il, jl, izp, jzp, valaoi, valaoj)

            km_equal = np.array_equal(km, km_res)
            if (not km_equal):
                print("Fortran:")
                print("km: ", km_res)

                print("Python:")
                print("km: ", km)

                print("\033[1;31m")
                assert km_equal, "[h0scal] km reals do not match"

    print("\033[0;32m", end='')
    print("matches! [h0scal]")
    print("\033[0;0m", end='')


        

# compare_args_i: compare fortran args to python for iteration i
def test_build_SDQH0(compare_args_i = -1):
    for i, file_path in enumerate(glob.glob(f'{directory}/build_SDQH0/*.bin')):
        with open(file_path, 'rb') as f:
            def read_ints(n=1):
                return np.fromfile(f, dtype=np.int32, count=n)

            def read_reals(n=1):
                return np.fromfile(f, dtype=np.float64, count=n)

            nat = read_ints(1)[0]
            at1 = read_ints(1)[0]
            at = np.fromfile(f, dtype=np.int32, count=at1)
            nbf = read_ints(1)[0]
            nao = read_ints(1)[0]
            xyz1, xyz2 = read_ints(2)
            xyz = np.fromfile(f, dtype=np.float64, count=xyz1 * xyz2).reshape((xyz2, xyz1))
            trans1, trans2 = read_ints(2)
            trans = np.fromfile(f, dtype=np.float64, count=trans1 * trans2).reshape((trans2, trans1))
            selfEnergy1, selfEnergy2 = read_ints(2)
            selfEnergy = np.fromfile(f, dtype=np.float64, count=selfEnergy1 * selfEnergy2).reshape((selfEnergy2, selfEnergy1))
            intcut = read_reals(1)[0]
            caoshell1, caoshell2 = read_ints(2)
            caoshell = np.fromfile(f, dtype=np.int32, count=caoshell1 * caoshell2).reshape((caoshell2, caoshell1))
            saoshell1, saoshell2 = read_ints(2)
            saoshell = np.fromfile(f, dtype=np.int32, count=saoshell1 * saoshell2).reshape((saoshell2, saoshell1))
            nprim1 = read_ints(1)[0]
            nprim = np.fromfile(f, dtype=np.int32, count=nprim1)
            primcount1 = read_ints(1)[0]
            primcount = np.fromfile(f, dtype=np.int32, count=primcount1)
            alp1 = read_ints(1)[0]
            alp = np.fromfile(f, dtype=np.float64, count=alp1)
            cont1 = read_ints(1)[0]
            cont = np.fromfile(f, dtype=np.float64, count=cont1)

            sint_res1, sint_res2 = read_ints(2)
            sint_res = np.fromfile(f, dtype=np.float64, count=sint_res1 * sint_res2).reshape((sint_res2, sint_res1))
            dpint_res1, dpint_res2, dpint_res3 = read_ints(3)
            dpint_res = np.fromfile(f, dtype=np.float64, count=dpint_res1 * dpint_res2 * dpint_res3).reshape((dpint_res3, dpint_res2, dpint_res1))
            qpint_res1, qpint_res2, qpint_res3 = read_ints(3)
            qpint_res = np.fromfile(f, dtype=np.float64, count=qpint_res1 * qpint_res2 * qpint_res3).reshape((qpint_res3, qpint_res2, qpint_res1))
            H0_res1 = read_ints(1)[0]
            H0_res = np.fromfile(f, dtype=np.float64, count=H0_res1)
            H0_noovlp_res1 = read_ints(1)[0]
            H0_noovlp_res = np.fromfile(f, dtype=np.float64, count=H0_noovlp_res1)

            sint, dpint, qpint, H0, H0_noovlp = build_SDQH0(nat, at, nbf, nao, xyz, trans, selfEnergy, intcut, caoshell, saoshell, nprim, primcount, alp, cont)

            ### Print argument comparison ###
            if (compare_args_i == i):
                from energy import trans as ptrans, getSelfEnergy, GFN2_coordination_numbers_np, intcut as pintcut
                from basisset import dim_basis_np, new_basis_set_simple
                element_ids = at-1
                element_cnt = element_ids.shape[0]
                positions = xyz
                _, basis_nao, basis_nbf = dim_basis_np(element_ids)
                cn = GFN2_coordination_numbers_np(element_ids, positions)
                selfEnergy_H_kappa_kappa = getSelfEnergy(element_ids, cn)


                basis_shells, basis_sh2ao, basis_sh2bf, basis_minalp, basis_level, basis_zeta, basis_valsh, basis_hdiag, basis_alp, basis_cont, basis_hdiag2, basis_aoexp, basis_ash, basis_lsh, basis_ao2sh, basis_nprim, basis_primcount, basis_caoshell, basis_saoshell, basis_fila, basis_fila2, basis_lao, basis_aoat, basis_valao, basis_lao2, basis_aoat2, basis_valao2, ok = new_basis_set_simple(element_ids)

                compare(np.array(nat),np.array(element_cnt), "nat")
                compare(at,element_ids+1, "at")
                compare(nbf,basis_nbf, "nbf")
                compare(nao, basis_nao, "nao")
                compare(xyz, positions, "xyz")
                compare(trans, ptrans, "trans")
                compare(selfEnergy, selfEnergy_H_kappa_kappa, "selfEnergy")
                compare(intcut, pintcut, "intcut")
                compare(caoshell, basis_caoshell, "caoshell")
                compare(saoshell, basis_saoshell, "saoshell")
                compare(nprim, basis_nprim, "nprim")
                compare(primcount, basis_primcount, "primcount")
                compare(alp, basis_alp, "alp")
                compare(cont, basis_cont, "cont")


            #compare(sint_res, sint, "[build_SDQH0] sint")
            #compare(dpint_res, dpint, "[build_SDQH0] dpint")
            #compare(qpint_res, qpint, "[build_SDQH0] qpint")
            #compare(H0_res, H0, "[build_SDQH0] H0")
            #compare(H0_noovlp_res, H0_noovlp, "[build_SDQH0] H0_noovlp")

    print("\033[0;32m", end='')
    print("matches! [build_SDQH0]")
    print("\033[0;0m", end='')


test_olapp()
test_multipole_3d()
test_horizontal_shift()
test_form_product()
test_dtrf2()
test_get_multiints()
#test_h0scal()
test_build_SDQH0(compare_args_i=0)
