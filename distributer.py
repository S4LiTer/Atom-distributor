import distribution_checker as dc
import numpy as np

atoms = 50
atoms_placed_randomly = 20
steps_in_row = 20

checker = dc.DistributionChecker(target_atom_count=atoms)
size = checker.fixed_size
dist = np.random.uniform(0, size, (atoms_placed_randomly, 3))
step = size/steps_in_row

checker.run(dist, plot = False)

for _ in range(atoms-atoms_placed_randomly):
    error_grid = np.zeros((steps_in_row, steps_in_row, steps_in_row))


    for _x in range(steps_in_row):
        x = step*_x

        for _y in range(steps_in_row):
            y = step*_y
            
            for _z in range(steps_in_row):
                z = step*_z

                new_error = checker.add_one_atom(np.array([x, y, z]), check_only=True)
                error_grid[_x, _y, _z] = new_error

    min_coords = np.array(np.unravel_index(np.argmin(error_grid), error_grid.shape))
    min_coords = min_coords * step
    print(checker.add_one_atom(min_coords, check_only=False))

checker.run(checker.distribution, plot=True)