#!/usr/bin/env python

import argparse
from Bio.PDB import PDBParser, PDBIO


def select_altloc_by_occupancy(input_pdb, output_pdb):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", input_pdb)

    for model in structure:
        for chain in model:
            for residue in chain:
                altloc_groups = {}
                for atom in residue:
                    altloc = atom.get_altloc()
                    altloc_groups.setdefault(altloc, []).append(atom)

                # altLoc が複数ある場合のみ処理
                if len(altloc_groups) > 1:
                    best_altloc = max(
                        altloc_groups.items(),
                        key=lambda x: sum(
                            (a.get_occupancy() or 0.0) for a in x[1]
                        )
                    )[0]

                    for alt, atoms in altloc_groups.items():
                        if alt != best_altloc:
                            for atom in atoms:
                                residue.detach_child(atom.id)

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)


def main():
    parser = argparse.ArgumentParser(
        description="Select altLoc by maximum summed occupancy"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input PDB file"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output PDB file"
    )

    args = parser.parse_args()
    select_altloc_by_occupancy(args.input, args.output)


if __name__ == "__main__":
    main()
