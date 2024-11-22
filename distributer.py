import distribution_checker as dc
import time
import numpy as np


atoms = 80
placed_atoms = 80
steps_in_row = 21
depth = 4

prediction_depth = 0
prediction_position_count = 7


file = "unique-80-0-26.txt"#"dist80-0-19.txt"

output_file = "dist.txt"

gpu = False
all_unique = True


# Funkce pokládá atomy v závislosti na nejnižším erroru po několika dalších položených atomech
# Sice funguje, ale nárok na výkon je mnohem vyšší než hodnota výsledku
# Je schopna tvořit relativně konstantně distribuce < 0.3
def find_best_location_predictive(org_checker, size, depth, prediction_depth, steps_in_row, gpu):
    step = size/(steps_in_row)
    start = (step/2)

    
    if not prediction_depth:
        er = None
        final_pos = None
        for i in range(depth):
            final_pos, er = find_best_location(org_checker, step, start, steps_in_row, gpu)

            step = size/(steps_in_row**(i+2))
            start = final_pos - (float(steps_in_row-1)/2)*step
        
        #print(final_pos, er)
        return final_pos, er
    


    errs = np.array([])
    poss = np.array([])


    grid_indicies = np.indices((steps_in_row, steps_in_row, steps_in_row)).reshape(3, -1).T
    positions = (start) + (step * grid_indicies)

    if depth == 1:
        if gpu: errs = org_checker.add_one_atom_gpu(positions)
        else: errs = org_checker.add_one_atom_vectorized(positions)

        poss = positions

    else:
        for pos in positions:
            er = None

            # do hloubky hledá pouze na základní úrovni. 
            # Proto se pro každou pozici na zíkladní úrovni najde nejoptimílnější místo "pod ní"
            for i in range(depth-1):
                step = size/(steps_in_row**(i+2))
                start = pos - (float(steps_in_row-1)/2)*step

                pos, er = find_best_location(org_checker, step, start, steps_in_row, gpu)

            # pos -> nejoptimálnější uložení pro každou základní polohu
            errs = np.append(errs, er)
            poss = np.append(poss, pos)


    poss = poss.reshape(-1, 3)
    sort_indicies = np.argsort(errs)
    errs = errs[sort_indicies]
    poss = poss[sort_indicies]
    
    lowest_error = 99999.0
    lowest_error_pos = None

    for i in range(prediction_position_count):
        if prediction_depth == 3: print(f"{i}/{prediction_position_count-1}")
        checker = org_checker.copy()

        pos = poss[i]
        er = checker.add_one_atom(pos, check_only=False)

        step = size/(steps_in_row)
        start = (step/2)


        _, er = find_best_location_predictive(checker, size, depth, prediction_depth-1, steps_in_row, gpu)


        if er < lowest_error: 
            lowest_error = er
            lowest_error_pos = pos

    return lowest_error_pos, lowest_error



def find_best_location(checker, step, start, steps_in_row, gpu, unique):
    grid_indicies = np.indices((steps_in_row, steps_in_row, steps_in_row)).reshape(3, -1).T
    positions = (start) + (step * grid_indicies)
    
    errors = None
    if gpu:
        errors = checker.add_one_atom_gpu(positions)
    else:
        #errors = np.array([checker.add_one_atom(pos, check_only=True) for pos in positions])
        errors = checker.add_one_atom_vectorized(positions)

    error_grid = errors.reshape(steps_in_row, steps_in_row, steps_in_row)
    
    
    if not unique:
        error = np.min(error_grid)
        min_coords = np.array(np.unravel_index(np.argmin(error_grid), error_grid.shape))
        min_coords = (start) + (min_coords * step)

        return min_coords, error
    
    
    sorted_errors = np.sort(errors)
    
    for err in sorted_errors:
        pos = np.where(error_grid == err)
        min_pos = np.array([pos[0][0], pos[1][0], pos[2][0]])
        min_coords = (start) + (min_pos * step)

        if not np.any(np.all(checker.distribution == min_coords, axis=1)):
            return min_coords, err
        
        error_grid[tuple(min_pos)] = 99

    raise Exception("je konec tohle vubec neni doreseny. proste zvis rozliseni lol")
    



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


err_before = checker.run(dist, plot = False)

"""
print(err_before)

for i in range(10):
    missplaced_atom_index = checker.pref()
    checker.distribution = np.delete(checker.distribution, missplaced_atom_index, axis=0)

    step = size/(steps_in_row)
    start = (step/2)

    final_pos = None
    unique = False
    for i in range(depth):
        if i == depth-1: 
            unique = all_unique

        final_pos, er = find_best_location(checker, step, start, steps_in_row, gpu, unique)

        step = size/(steps_in_row**(i+2))
        start = final_pos - (float(steps_in_row-1)/2)*step



    error = checker.add_one_atom(final_pos, check_only=False)

    print(error, checker.pref())

exit()
"""

for _ in range(atoms-placed_atoms):

    step = size/(steps_in_row)
    start = (step/2)

    final_pos = None
    unique = False
    for i in range(depth):
        if i == depth-1: 
            unique = all_unique

        final_pos, er = find_best_location(checker, step, start, steps_in_row, gpu, unique)

        step = size/(steps_in_row**(i+2))
        start = final_pos - (float(steps_in_row-1)/2)*step

    """
    pd = min(prediction_depth, atoms-_-placed_atoms)
    final_pos, er = find_best_location_predictive(checker, size, depth, pd, steps_in_row, gpu)
    """
    error = checker.add_one_atom(final_pos, check_only=False)



    if not (_+1)%1: print(f"{str(_+1+placed_atoms).zfill(len(str(atoms)))}/{atoms} | Error:", round(error,4))

final_error = checker.run(checker.distribution, plot=False)
np.savetxt(output_file, checker.distribution)

print(f"{atoms}/{atoms} | Error:", round(final_error, 5))
print(f"Finished in {round(time.time() - start_time, 3)}s")

