import distribution_checker as dc
import numpy as np

checker = dc.DistributionChecker()
checker.run(np.loadtxt("dist0-49.txt"), plot=True)
exit()
atoms = 500
placed_atoms = 425
steps_in_row = 13
depth = 7

load_from_file = True
file = "blueprint0-50.txt"

output_name = "dist.txt"

print("Steps in row:", steps_in_row)
print("Depth:", depth)
print("Total atom count:", atoms)
if file and load_from_file:
    print(f"{placed_atoms} atoms loaded from {file}")
else:
    print(f"Placing {placed_atoms} randomly")
print("-"*30)



def find_best_location(checker, step, start):
    grid_indicies = np.indices((steps_in_row, steps_in_row, steps_in_row)).reshape(3, -1).T
    positions = (start) + (step * grid_indicies)
    
    errors = np.array([checker.add_one_atom(pos, check_only=True) for pos in positions])
    error_grid = errors.reshape(steps_in_row, steps_in_row, steps_in_row)

    min_coords = np.array(np.unravel_index(np.argmin(error_grid), error_grid.shape))
    min_coords = (start) + (min_coords * step)

    return min_coords




checker = dc.DistributionChecker(target_atom_count=atoms)
size = checker.fixed_size

dist = np.random.uniform(0, size, (placed_atoms, 3))
if file and load_from_file:
    dist = np.loadtxt(file)
    dist = dist[:placed_atoms]


step = size/(steps_in_row)
checker.run(dist, plot = False)

for _ in range(atoms-placed_atoms):

    step = size/(steps_in_row)
    start = (step/2)

    final_pos = None

    for i in range(depth):
        final_pos = find_best_location(checker, step, start)

        step = size/(steps_in_row**(i+2))
        start = final_pos - (float(steps_in_row-1)/2)*step
    

    
    error = checker.add_one_atom(final_pos, check_only=False)
    if not _%5: 
        print(f"Atoms placed: {_}, error score:", error)

print("Final erorr:", checker.run(checker.distribution, plot=False))
np.savetxt(output_name, checker.distribution)

