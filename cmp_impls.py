import numpy as np
from energy import dtrf2, get_multiints
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
            fdd = np.fromfile(f, dtype=np.float64, count=d1 * d2 * d3).reshape((d1, d2, d3))

            # Read qq dimensions
            q1, q2, q3 = read_ints(3)
            fqq = np.fromfile(f, dtype=np.float64, count=q1 * q2 * q3).reshape((q1, q2, q3))

            ss,dd,qq = get_multiints(icao, jcao, naoi, naoj, ishtyp, jshtyp, ri, rj, point, intcut, nprim, primcount, alp, cont)

            ss_equal = np.array_equal(ss, fss)
            if (not ss_equal):
                print("Fortran:")
                print("ss: ", fss)

                print("Python:")
                print("ss: ", ss)
                assert ss_equal, "ss matrices do not match"

            dd_equal = np.array_equal(dd, fdd)
            if (not dd_equal):
                print("Fortran:")
                print("dd: ", fdd)

                print("Python:")
                print("dd: ", dd)
                assert dd_equal, "dd matrices do not match"

            qq_equal = np.array_equal(qq, fqq)
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

            if (not np.array_equal(s, s_res)):
                print("Fortran:")
                print("s: ", s_res)

                print("Python:")
                print("s: ", s)
                assert np.array_equal(s, s_res), "s matrices do not match"



test_get_multiints()
test_dtrf2()
