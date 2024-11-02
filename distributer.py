import distribution_checker as dc
import time
import numpy as np


total_atom_count = 80
already_placed_atoms = 1
steps_in_one_row = 8
depth_steps = 4

starting_file = ""#"blueprint0-55.txt"

output_file_name = "dist.txt"

use_gpu = True


def find_best_location(checker, step, start, steps_in_row, gpu):
    grid_indicies = np.indices((steps_in_row, steps_in_row, steps_in_row)).reshape(3, -1).T
    positions = (start) + (step * grid_indicies)
    
    #errors = np.array([checker.add_one_atom(pos, check_only=True) for pos in positions])
    errors = None
    if gpu:
        errors = checker.add_one_atom_gpu(positions)
    else:
        errors = checker.add_one_atom_vectorized(positions)

    error_grid = errors.reshape(steps_in_row, steps_in_row, steps_in_row)

    min_coords = np.array(np.unravel_index(np.argmin(error_grid), error_grid.shape))
    min_coords = (start) + (min_coords * step)

    return min_coords



def start_brute_force(atoms=80, placed_atoms=1, steps_in_row=4, depth=4, file="", output_name="dist.txt", gpu=False):
    print("STARTING BRUTE FORCE")
    print("Steps in row:", steps_in_row)
    print("Depth:", depth)
    print("Total atom count:", atoms)
    if file:
        print(f"{placed_atoms} atoms loaded from {file}")
    else:
        print(f"Placing {placed_atoms} randomly")
    print("-"*30)

    start_time = time.time()


    checker = dc.DistributionChecker(target_atom_count=atoms)
    size = checker.fixed_size

    dist = np.random.uniform(0, size, (placed_atoms, 3))
    if file:
        dist = np.loadtxt(file)
        dist = dist[:placed_atoms]


    step = size/(steps_in_row)
    checker.run(dist, plot = False)

    for _ in range(atoms-placed_atoms):

        step = size/(steps_in_row)
        start = (step/2)

        final_pos = None

        for i in range(depth):
            final_pos = find_best_location(checker, step, start, steps_in_row, gpu)

            step = size/(steps_in_row**(i+2))
            start = final_pos - (float(steps_in_row-1)/2)*step
        

        
        error = checker.add_one_atom(final_pos, check_only=False)
        if not _%10:
            print(f"{str(_).zfill(len(str(atoms)))}/{atoms} | Error:", round(error,4))

    final_error = checker.run(checker.distribution, plot=False)
    np.savetxt(output_name, checker.distribution)
    
    print(f"{atoms}/{atoms} | Error:", round(final_error, 5))
    print(f"Finished in {round(time.time() - start_time)}s")

    return final_error


if __name__ == "__main__":
    start_brute_force(total_atom_count, already_placed_atoms, steps_in_one_row, 
                        depth_steps, starting_file, output_file_name, use_gpu)