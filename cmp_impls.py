import numpy as np
from energy import dtrf2, form_product, get_multiints, h0scal, horizontal_shift, multipole_3d, olapp
import glob
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compare original xtb results with our own python implementation.")
parser.add_argument("directory", help="Path to the directory containing the binary files from xtb containing the arguments and result.")
args = parser.parse_args()

directory = args.directory
directory = os.path.abspath(directory)


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
            fss = np.fromfile(f, dtype=np.float64, count=m * n).reshape((m, n))

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
                assert ss_equal, "ss matrices do not match"

            dd_equal = np.allclose(dd, fdd)
            if (not dd_equal):
                print("Fortran:")
                print("dd: ", fdd)

                print("Python:")
                print("dd: ", dd)
                assert dd_equal, "dd matrices do not match"

            qq_equal = np.allclose(qq, fqq)
            if (not qq_equal):
                print("Fortran:")
                print("qq: ", fqq)

                print("Python:")
                print("qq: ", qq)
                assert qq_equal, "qq matrices do not match"



def test_dtrf2():
    for file_path in glob.glob(f'{directory}/dtrf2/*.bin'):  # Matches all .bin files in the directory
        with open(file_path, 'rb') as f:
            def read_ints(n=1):
                return np.fromfile(f, dtype=np.int32, count=n)

            s1, s2 = read_ints(2)
            s = np.fromfile(f, dtype=np.float64, count=s1 * s2).reshape((s1, s2))

            li = read_ints(1)[0]
            lj = read_ints(1)[0]

            s1_res, s2_res = read_ints(2)
            s_res = np.fromfile(f, dtype=np.float64, count=s1_res * s2_res).reshape((s1_res, s2_res))

            dtrf2(s, li, lj)

            s_equal = np.array_equal(s, s_res)
            if (not s_equal):
                print("Fortran:")
                print("s: ", s_res)

                print("Python:")
                print("s: ", s)
                assert s_equal, "s matrices do not match"


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
                assert d_equal, "d matrices do not match"


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
                assert cfs_equal, "cfs matrices do not match"


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
                assert s3d_equal, "s3d matrices do not match"



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
                assert s_equal, "s reals do not match"



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
                assert km_equal, "km reals do not match"


#test_olapp()
#test_multipole_3d()
#test_horizontal_shift()
#test_form_product()
#test_dtrf2()
test_get_multiints()
#test_h0scal()
