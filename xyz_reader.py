def parse_xyz(file):
    with open(file, "r") as f:
        f.readline()
        f.readline()
        atoms = [
            (e[0], [float(e[1]), float(e[2]), float(e[3])])
            for e in (line.split() for line in f)
        ]
    return atoms
