import numpy as np
import matplotlib.pyplot as plt

def find_whole(array, values):
    ixs = np.round(find_fractional(array, values)).astype(int)
    return ixs

def find_fractional(array, values):
    step = (array[-1] - array[0]) / (len(array) - 1)
    ixs = (values - array[0]) / step
    return ixs

def create_margins(size_of_the_cube, atomic_positions):
    atomic_positions_margins = np.zeros((1, 3))
    shifts = [-size_of_the_cube, 0, size_of_the_cube]
    for value_x in shifts:
        for value_y in shifts:
            for value_z in shifts:
                if not (value_x == 0 and value_y == 0 and value_z == 0):
                    shifted_positions = atomic_positions + np.array([value_x, value_y, value_z])
                    atomic_positions_margins = np.concatenate((atomic_positions_margins, shifted_positions))
    atomic_positions_margins = atomic_positions_margins[1:]
    return atomic_positions_margins

def average_distance_calculator(distribution, distribution_margins):
    experiment = np.loadtxt('xenon_distribution_data_linear.txt')[:, 0]
    nbins = len(experiment)
    average_distance_random = np.zeros(nbins)
    rs = experiment
    atoms = np.concatenate((distribution, distribution_margins))
    for atom in distribution:
        distances = np.linalg.norm(atoms - atom, axis=1)
        idxs = find_whole(rs, distances)
        for idx in idxs:
            if 0 <= idx < nbins:
                average_distance_random[int(idx)] += 1.
    return average_distance_random, rs

def plot_results(distances, rs, no_of_atoms):
    dft_distances = [4.2657, 6.0326, 7.3884, 8.5314, 9.5384]
    fig, ax = plt.subplots()
    for distance in dft_distances:
        ax.axvline(distance, color='gray', linestyle='--', linewidth=1)
    experiment = np.loadtxt('xenon_distribution_data_linear.txt')
    number_density = 0.01315
    bin_width = rs[1] - rs[0]
    distances = distances / (no_of_atoms * bin_width)
    average = number_density * 4. * np.pi * rs ** 2
    plot_values = np.array(distances / average)
    
    print(distances)

    error = np.sqrt(np.sum((plot_values - experiment[:, 1])**2))
    print(error)

    ax.plot(rs, plot_values, label='Input distribution')
    ax.plot(experiment[:, 0], experiment[:, 1], 'o', color='green', label='Experimental data')
    ax.legend(loc='upper right')
    ax.set_title('Error score: ' + str(round(error, 2)))
    ax.set_ylim(0., 4.)
    ax.set_xlim(3.7, 10.)
    ax.set_xlabel('Distance from atom [$\\AA$]')
    ax.set_ylabel('Radial distribution function [-]')
    #plt.show()
    plt.close(fig)

def plot_distribution(distribution):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(distribution[:, 0], distribution[:, 1], distribution[:, 2], s=1)
    ax.set_title('Visualization of the Input Distribution of Atoms')
    ax.set_xlabel('x [$\\AA$]')
    ax.set_ylabel('y [$\\AA$]')
    ax.set_zlabel('z [$\\AA$]')
    #plt.show()
    plt.close(fig)

def run(student_distribution):
    number_density = 0.01315
    size_of_the_cube = (len(student_distribution) / number_density) ** (1./3.)
    margins_of_a_student_distribution = create_margins(size_of_the_cube, student_distribution)
    distances, rs = average_distance_calculator(student_distribution, margins_of_a_student_distribution)
    # Plot results
    plot_results(distances, rs, len(student_distribution))
    plot_distribution(student_distribution)


dist = np.loadtxt("dist0-49.txt")
run(dist)