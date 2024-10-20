import numpy as np
import matplotlib.pyplot as plt

class DistributionChecker:
    def __init__(self, target_atom_count=None):
        self.number_density = 0.01315
        self.experiment = np.loadtxt('xenon_distribution_data_linear.txt')
        self.rs = self.experiment[:, 0]

        self.fixed_size = None
        if target_atom_count:
            self.fixed_size = (target_atom_count/self.number_density) ** (1./3.)
            print("fixed size:", self.fixed_size)

        self.distribution = None
        self.margins = None
        self.distances = None


    def find_whole(self, array, values):
        ixs = np.round(self.find_fractional(array, values)).astype(int)
        return ixs

    def find_fractional(self, array, values):
        step = (array[-1] - array[0]) / (len(array) - 1)
        ixs = (values - array[0]) / step
        return ixs

    def create_margins(self, size_of_the_cube, atomic_positions):
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

    def average_distance_calculator(self, distribution, distribution_margins):
        nbins = len(self.rs)
        average_distance_random = np.zeros(nbins)
        
        atoms = np.concatenate((distribution, distribution_margins))
        for atom in distribution:
            distances = np.linalg.norm(atoms - atom, axis=1)
            idxs = self.find_whole(self.rs, distances)
            
            for idx in idxs:
                if 0 <= idx < nbins:
                    average_distance_random[int(idx)] += 1.
        return average_distance_random

    def plot_results(self, distances, rs, no_of_atoms, error):
        dft_distances = [4.2657, 6.0326, 7.3884, 8.5314, 9.5384]
        fig, ax = plt.subplots()
        for distance in dft_distances:
            ax.axvline(distance, color='gray', linestyle='--', linewidth=1)

        number_density = 0.01315
        bin_width = rs[1] - rs[0]
        distances = distances / (no_of_atoms * bin_width)
        average = number_density * 4. * np.pi * rs ** 2
        plot_values = distances / average

        ax.plot(rs, plot_values, label='Input distribution')
        experiment = np.loadtxt('xenon_distribution_data_linear.txt')
        ax.plot(experiment[:, 0], experiment[:, 1], 'o', color='green', label='Experimental data')
        ax.legend(loc='upper right')
        ax.set_title('Error score: ' + str(round(error, 2)))
        ax.set_ylim(0., 4.)
        ax.set_xlim(3.7, 10.)
        ax.set_xlabel('Distance from atom [$\\AA$]')
        ax.set_ylabel('Radial distribution function [-]')
        plt.show()
        plt.close(fig)

    def plot_distribution(self, distribution):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(distribution[:, 0], distribution[:, 1], distribution[:, 2], s=1)
        ax.set_title('Visualization of the Input Distribution of Atoms')
        ax.set_xlabel('x [$\\AA$]')
        ax.set_ylabel('y [$\\AA$]')
        ax.set_zlabel('z [$\\AA$]')
        plt.show()
        plt.close(fig)

    def add_one_atom(self, position: np.ndarray, check_only=True):

        if not self.fixed_size:
            raise Exception("This function can be used only if fixed_size is defined => you have to define target_atom_count when creating this object.")
        
        margins = np.copy(self.margins)
        distribution = np.copy(self.distribution)
        shifts = [-self.fixed_size, 0, self.fixed_size]


        new_distances = np.array([])
        for value_x in shifts:
            for value_y in shifts:
                for value_z in shifts:
                    pos = position+np.array([value_x, value_y, value_z])
                    if not (value_x == 0 and value_y == 0 and value_z == 0):
                        margins = np.concatenate((margins, [pos]))
                        new_distances = np.concatenate((new_distances, np.linalg.norm(distribution-pos, axis=1)))
        

        base_box_distances = np.linalg.norm(distribution-position, axis=1)
        base_box_distances = np.append(base_box_distances, base_box_distances)
        margin_distances = np.linalg.norm(margins-position, axis=1)
        
        distribution = np.concatenate((distribution, [position]))

        all_distances = np.concatenate((new_distances, base_box_distances, margin_distances))
        distances = np.copy(self.distances)

        nbins = len(self.rs)  
        idxs = self.find_whole(self.rs, all_distances)
        idxs = idxs[(idxs >= 0) & (idxs < nbins)].astype(int)
        distances += np.bincount(idxs, minlength=nbins)

        if not check_only:
            self.distribution = distribution
            self.margins = margins
            self.distances = distances

        return self.calculate_error(distances, distribution)


    def run(self, student_distribution, plot = False):
        size_of_the_cube = self.fixed_size
        if not self.fixed_size:
            size_of_the_cube = (len(student_distribution) / self.number_density) ** (1./3.)

        margins_of_a_student_distribution = self.create_margins(size_of_the_cube, student_distribution)
        
        distances = self.average_distance_calculator(student_distribution, margins_of_a_student_distribution)
        self.distribution = student_distribution
        self.margins = margins_of_a_student_distribution
        self.distances = distances

        error = self.calculate_error(distances, student_distribution)
        # Plot results

        if plot: 
            self.plot_values(distances, error)

        return error

    def plot_values(self, distances, error):
        self.plot_results(distances, self.rs, len(self.distribution), error)
        self.plot_distribution(self.distribution)

    def calculate_error(self, distances, distribution):
        average = self.number_density * 4. * np.pi * self.rs ** 2
        error = np.sum(np.abs((self.experiment[:, 1] - distances / average / len(distribution)) / self.experiment[:, 1]))
        return error


