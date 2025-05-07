import numpy as np
from energy import get_multiints
import glob
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compare original xtb results with our own python implementation.")
parser.add_argument("directory", help="Path to the directory containing the binary files from xtb containing the arguments and result.")
args = parser.parse_args()

directory = args.directory
directory = os.path.abspath(directory)

for file_path in glob.glob(f'{directory}/*.bin'):  # Matches all .bin files in the directory
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

        print("Fortran:")
        print("ss: ", fss)
        print("dd: ", fdd)
        print("qq: ", fqq)

        ss,dd,qq = get_multiints(icao, jcao, naoi, naoj, ishtyp, jshtyp, ri, rj, point, intcut, nprim, primcount, alp, cont)

        print("Python:")
        print("ss: ", ss)
        print("dd: ", dd)
        print("qq: ", qq)

        assert np.array_equal(ss, fss)
        assert np.array_equal(dd, fdd)
        assert np.array_equal(qq, fqq)

        #print("icao:", icao)
        #print("jcao:", jcao)
        #print("naoi:", naoi)
        #print("naoj:", naoj)
        #print("ishtyp:", ishtyp)
        #print("jshtyp:", jshtyp)
        #print("ri:", ri)
        #print("rj:", rj)
        #print("point:", point)
        #print("intcut:", intcut)
        #print("nprim:", nprim)
        #print("primcount:", primcount)
        #print("alp:", alp)
        #print("cont:", cont)
