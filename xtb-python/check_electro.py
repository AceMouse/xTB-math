import numpy as np
import argparse
import glob
import os
from scc import electro

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Compare original xtb results with our own python implementation.")
parser.add_argument("directory", help="Path to the directory containing the binary files from xtb containing the arguments and result.")
args = parser.parse_args()

directory = args.directory
directory = os.path.abspath(directory)


def test_electro():
    fn_name = "electro"
    for i, file_path in enumerate(glob.glob(f'{directory}/calls/{fn_name}/*.bin')):
        with open(file_path, 'rb') as f:
            def read_ints(n=1):
                return np.fromfile(f, dtype=np.int32, count=n)

            def read_reals(n=1):
                return np.fromfile(f, dtype=np.float64, count=n)

            nbf = read_ints(1)[0]

            H01 = read_ints(1)[0]
            H0 = np.fromfile(f, dtype=np.float64, count=H01)

            m, n = read_ints(2)
            P = np.fromfile(f, dtype=np.float64, count=m * n).reshape((n, m))

            dq1 = read_ints(1)[0]
            dq = np.fromfile(f, dtype=np.float64, count=dq1)

            dqsh1 = read_ints(1)[0]
            dqsh = np.fromfile(f, dtype=np.float64, count=dqsh1)

            atomicGam1 = read_ints(1)[0]
            atomicGam = None if atomicGam1 == 0 else np.fromfile(f, dtype=np.float64, count=atomicGam1)

            shellGam1 = read_ints(1)[0]
            shellGam = None if shellGam1 == 0 else np.fromfile(f, dtype=np.float64, count=shellGam1)

            m, n = read_ints(2)
            jmat = np.fromfile(f, dtype=np.float64, count=m * n).reshape((n, m))

            shift1 = read_ints(1)[0]
            shift = np.fromfile(f, dtype=np.float64, count=shift1)
            

            es_res, scc_res = read_reals(2)

            es, scc = electro(nbf, H0, P, dq, dqsh, atomicGam, shellGam, jmat, shift)

            print(f"{es} {es_res} {scc} {scc_res}")

test_electro()
