import numpy as np

element2id = {
    "H": 0,
    "C": 5,
    "N": 6,
    "O": 7
}

angstrom_to_bohr_conversion_constant = 1.8897259886

def parse_xyz(file, input_in_angstrom=True):
    element_ids = []
    positions = []
    with open(file, "r") as f:
        f.readline()
        f.readline()

        for line in f:
            e = line.split()
            element_ids.append(element2id[e[0]])
            if input_in_angstrom:
                positions.append([float(e)*angstrom_to_bohr_conversion_constant for e in e[1:4]])
            else:
                positions.append([float(e) for e in e[1:4]])
    element_ids = np.array(element_ids)
    positions = np.array(positions, dtype=float)
    return element_ids, positions


def parse_xyz_with_symbols(file, input_in_angstrom=True):
    symbols = []
    positions = []
    with open(file, "r") as f:
        f.readline()
        f.readline()

        for line in f:
            e = line.split()
            symbols.append(e[0])
            if input_in_angstrom:
                positions.append([float(e)*angstrom_to_bohr_conversion_constant for e in e[1:4]])
            else:
                positions.append([float(e) for e in e[1:4]])
    symbols = np.array(symbols)
    positions = np.array(positions, dtype=float)
    return symbols, positions

#element_ids, positions = parse_xyz("./caffeine.xyz")
#print(f"element_ids: {element_ids}")
#print(f"positions: {positions}")
