import torch
from pymatgen.core import Structure, Lattice
from pymatgen.io.cif import CifWriter
import sys
import argparse

def convert_process(args):
    x= torch.load(sys.path[0] + args.model_path, map_location=torch.device('cpu'))
    def get_crystals_list(
            frac_coords, atom_types, lengths, angles, num_atoms):
        """
        args:
            frac_coords: (num_atoms, 3)
            atom_types: (num_atoms)
            lengths: (num_crystals)
            angles: (num_crystals)
            num_atoms: (num_crystals)
        """
        assert frac_coords.size(0) == atom_types.size(0) == num_atoms.sum()
        assert lengths.size(0) == angles.size(0) == num_atoms.size(0)

        start_idx = 0
        crystal_array_list = []
        for batch_idx, num_atom in enumerate(num_atoms.tolist()):
            cur_frac_coords = frac_coords.narrow(0, start_idx, num_atom)
            cur_atom_types = atom_types.narrow(0, start_idx, num_atom)
            cur_lengths = lengths[batch_idx]
            cur_angles = angles[batch_idx]

            crystal_array_list.append({
                'frac_coords': cur_frac_coords.detach().cpu().numpy(),
                'atom_types': cur_atom_types.detach().cpu().numpy(),
                'lengths': cur_lengths.detach().cpu().numpy(),
                'angles': cur_angles.detach().cpu().numpy(),
            })
            start_idx = start_idx + num_atom
        return crystal_array_list

    def get_crystal_array_list(batch_idx=0):
        data = x
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx])

        if 'input_data_batch' in data:
            batch = data['input_data_batch']
            if isinstance(batch, dict):
                true_crystal_array_list = get_crystals_list(
                    batch['frac_coords'], batch['atom_types'], batch['lengths'],
                    batch['angles'], batch['num_atoms'])
            else:
                true_crystal_array_list = get_crystals_list(
                    batch.frac_coords, batch.atom_types, batch.lengths,
                    batch.angles, batch.num_atoms)
        else:
            true_crystal_array_list = None

        return crys_array_list, true_crystal_array_list

    res = get_crystal_array_list()

    for idx, data in enumerate(res[0]):
        frac_coords = data['frac_coords']
        atom_types = data['atom_types']
        lengths = data['lengths']
        angles = data['angles']
        lattice = Lattice.from_parameters(a=lengths[0], b=lengths[1], c=lengths[2], alpha=angles[0], beta=angles[1], gamma=angles[2])
        structure = Structure(lattice, atom_types, frac_coords)
        cif_writer = CifWriter(structure)
        cif_writer.write_file(sys.path[0] + "/../structures/" + str(idx) + '.cif')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    args = parser.parse_args()
    convert_process(args=args)