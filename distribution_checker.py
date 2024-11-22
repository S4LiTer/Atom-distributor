import numpy as np
import cupy
import matplotlib.pyplot as plt

class DistributionChecker:
    def __init__(self, target_atom_count=None):
        self.number_density = 0.01315
        self.experiment = np.loadtxt('xenon_distribution_data_linear.txt')
        self.rs = self.experiment[:, 0]

        self.atom_count = None
        self.fixed_size = None
        if target_atom_count:
            self.atom_count = target_atom_count
            self.fixed_size = (target_atom_count/self.number_density) ** (1./3.)


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

    def add_one_atom_gpu(self, positions:np.ndarray):
        if not self.fixed_size:
            raise Exception("This function can be used only if fixed_size is defined => you have to define target_atom_count when creating this object.")
        
        positions = cupy.asarray(positions)

        margins = cupy.copy(self.margins)
        distribution = cupy.copy(self.distribution)

        shifts = cupy.array([-self.fixed_size, 0, self.fixed_size])
        shift_combinations = cupy.array(cupy.meshgrid(shifts, shifts, shifts)).T.reshape(-1, 3)
        shift_combinations = cupy.delete(shift_combinations, 13, axis=0)

        positions = positions.reshape((-1, 1, 3))
        new_margins = positions + shift_combinations
        margins = cupy.tile(margins, (positions.shape[0], 1, 1))
        margins = cupy.concatenate((new_margins, margins), axis=1)

        distribution = cupy.tile(distribution, (positions.shape[0], 1, 1))

        new_distances = distribution[:, :, cupy.newaxis, :] - new_margins[:, cupy.newaxis, :, :] # Hnusne, ale nevim jak predelat
        new_distances = new_distances.reshape((positions.shape[0], -1, 3))
        new_distances = cupy.linalg.norm(new_distances, axis=2)
        # new_distances = distances between already placed atoms and new atoms in margin boxes
        
        base_box_distances = distribution[:, :, cupy.newaxis, :] - positions[:, cupy.newaxis, :, :]
        base_box_distances = base_box_distances.reshape((positions.shape[0], -1, 3))
        base_box_distances = cupy.linalg.norm(base_box_distances, axis=2)
        base_box_distances = cupy.concatenate((base_box_distances, base_box_distances), axis=1)
        # base_box_distances = distances between newly placed atoms in main box and already placed atoms

        margin_distances = margins[:, :, cupy.newaxis, :] - positions[:, cupy.newaxis, :, :]
        margin_distances = margin_distances.reshape((positions.shape[0], -1, 3))
        margin_distances = cupy.linalg.norm(margin_distances, axis=2)
        #margin_distances = distances between newly placed atoms and margins (including new atoms in margins)
        
        all_distances = cupy.concatenate((new_distances, base_box_distances, margin_distances), axis=1)
        distances = cupy.copy(self.distances)

        nbins = len(self.rs)  
        idxs = self.find_whole(self.rs, all_distances)

        idxs[(idxs < 0) | (idxs >= nbins)] = positions.shape[0] * nbins+5
        idxs = idxs + cupy.arange(positions.shape[0])[:, None] * nbins
        idxs = idxs.reshape((-1,))
        
        idxs = idxs[(idxs >= 0) & (idxs < positions.shape[0] * nbins)].astype(int)
        counts = cupy.bincount(idxs, minlength=nbins*positions.shape[0])
        counts = counts.reshape((positions.shape[0], nbins))

        distances = cupy.tile(distances, (positions.shape[0], 1))
        distances = distances + counts

        return self.calculate_error_gpu(distances, distribution)

    def calculate_error_gpu(self, distances, distribution):
        cupy_rs = cupy.asarray(self.rs)
        no_atoms = len(distribution)
        if self.atom_count:
            no_atoms = self.atom_count

        average = self.number_density * 4. * cupy.pi * cupy_rs ** 2
        bin_width = cupy_rs[1] - cupy_rs[0]
        distances = distances / (bin_width*no_atoms)
        plot_values = cupy.array(distances / average)
        
        axis = len(plot_values.shape)-1

        error = cupy.sqrt(cupy.sum((plot_values - cupy.asarray(self.experiment[:, 1]))**2, axis=axis))
        return cupy.asnumpy(error)


    def add_one_atom_vectorized(self, positions: np.ndarray):
        if not self.fixed_size:
            raise Exception("This function can be used only if fixed_size is defined => you have to define target_atom_count when creating this object.")
        
        margins = np.copy(self.margins)
        distribution = np.copy(self.distribution)

        shifts = np.array([-self.fixed_size, 0, self.fixed_size])
        shift_combinations = np.array(np.meshgrid(shifts, shifts, shifts)).T.reshape(-1, 3)
        shift_combinations = np.delete(shift_combinations, 13, axis=0)

        positions = positions.reshape((-1, 1, 3))
        new_margins = positions + shift_combinations
        margins = np.tile(margins, (positions.shape[0], 1, 1))
        margins = np.concatenate((new_margins, margins), axis=1)

        distribution = np.tile(distribution, (positions.shape[0], 1, 1))

        new_distances = distribution[:, :, np.newaxis, :] - new_margins[:, np.newaxis, :, :] # Hnusne, ale nevim jak predelat
        new_distances = new_distances.reshape((positions.shape[0], -1, 3))
        new_distances = np.linalg.norm(new_distances, axis=2)
        # new_distances = distances between already placed atoms and new atoms in margin boxes
        
        base_box_distances = distribution[:, :, np.newaxis, :] - positions[:, np.newaxis, :, :]
        base_box_distances = base_box_distances.reshape((positions.shape[0], -1, 3))
        base_box_distances = np.linalg.norm(base_box_distances, axis=2)
        base_box_distances = np.concatenate((base_box_distances, base_box_distances), axis=1)
        # base_box_distances = distances between newly placed atoms in main box and already placed atoms

        margin_distances = margins[:, :, np.newaxis, :] - positions[:, np.newaxis, :, :]
        margin_distances = margin_distances.reshape((positions.shape[0], -1, 3))
        margin_distances = np.linalg.norm(margin_distances, axis=2)
        #margin_distances = distances between newly placed atoms and margins (including new atoms in margins)
        
        all_distances = np.concatenate((new_distances, base_box_distances, margin_distances), axis=1)
        distances = np.copy(self.distances)

        nbins = len(self.rs)  
        idxs = self.find_whole(self.rs, all_distances)

        idxs[(idxs < 0) | (idxs >= nbins)] = positions.shape[0] * nbins+5
        idxs = idxs + np.arange(positions.shape[0])[:, None] * nbins
        idxs = idxs.reshape((-1,))
        
        idxs = idxs[(idxs >= 0) & (idxs < positions.shape[0] * nbins)].astype(int)
        counts = np.bincount(idxs, minlength=nbins*positions.shape[0])
        counts = counts.reshape((positions.shape[0], nbins))

        distances = np.tile(distances, (positions.shape[0], 1))
        distances = distances + counts

        return self.calculate_error(distances, distribution)


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
        
        no_atoms = len(distribution)
        if self.atom_count:
            no_atoms = self.atom_count

        average = self.number_density * 4. * np.pi * self.rs ** 2
        bin_width = self.rs[1] - self.rs[0]
        
        distances = distances / (bin_width*no_atoms*average)


        axis = len(distances.shape)-1
        error = np.sqrt(np.sum((distances - self.experiment[:, 1])**2, axis=axis))
        return error

    def pref(self):
        if len(np.unique(self.distribution, axis=0)) != len(self.distribution):
            #raise Exception("There are duplications in this distribution si this function wont work")
            pass

        positions = self.distribution


        shifts = np.array([-self.fixed_size, 0, self.fixed_size])
        shift_combinations = np.array(np.meshgrid(shifts, shifts, shifts)).T.reshape(-1, 3)
        shift_combinations = np.delete(shift_combinations, 13, axis=0)

        positions = positions.reshape((-1, 1, 3))
        new_margins = (positions + shift_combinations).reshape(positions.shape[0], 26, 1, 3)
        margins = self.margins.reshape(1, 1, -1, 3)
        margins = np.repeat((margins), positions.shape[0], axis=0)

        matches = ~np.any(np.all(margins == new_margins, axis=-1), axis=1)
        margins = margins.reshape(positions.shape[0], -1, 3)[matches].reshape(positions.shape[0], -1, 3)



        adjusted_dist = np.repeat((self.distribution[np.newaxis, :, :]), positions.shape[0], axis=0)[:, np.newaxis, :, :]
        positions = positions[:, :, np.newaxis, :]
        matches = ~np.any(np.all(adjusted_dist == positions, axis=-1), axis=1)
        adjusted_dist = adjusted_dist.reshape(positions.shape[0], -1, 3)[matches].reshape(positions.shape[0], 1, -1, 3)


        old_box_to_new_margins = adjusted_dist - new_margins 
        old_box_to_new_margins = old_box_to_new_margins.reshape((positions.shape[0], -1, 3))
        old_box_to_new_margins = np.linalg.norm(old_box_to_new_margins, axis=2)


        new_to_old_in_box = adjusted_dist - positions
        new_to_old_in_box = new_to_old_in_box.reshape((positions.shape[0], -1, 3))
        new_to_old_in_box = np.linalg.norm(new_to_old_in_box, axis=2)
        #new_to_old_in_box = np.concatenate((new_to_old_in_box, new_to_old_in_box), axis=1)

        
        new_to_old_margins = margins[:, np.newaxis, :, :] - positions
        new_to_old_margins = new_to_old_margins.reshape((positions.shape[0], -1, 3))
        new_to_old_margins = np.linalg.norm(new_to_old_margins, axis=2)

        atom_distances = np.concatenate((old_box_to_new_margins, new_to_old_in_box, new_to_old_margins), axis=-1)

        nbins = len(self.rs)  
        idxs = self.find_whole(self.rs, atom_distances)

        idxs[(idxs < 0) | (idxs >= nbins)] = positions.shape[0] * nbins+5
        idxs = idxs + np.arange(positions.shape[0])[:, None] * nbins
        idxs = idxs.reshape((-1,))
        
        idxs = idxs[(idxs >= 0) & (idxs < positions.shape[0] * nbins)].astype(int)
        counts = np.bincount(idxs, minlength=nbins*positions.shape[0])
        counts = counts.reshape((positions.shape[0], nbins))




        average = self.number_density * 4. * np.pi * self.rs ** 2
        bin_width = self.rs[1] - self.rs[0]
        
        err = self.distances / (bin_width*self.atom_count*average)
        err = (err - self.experiment[:, 1])*(bin_width*self.atom_count*average)
        err = np.repeat(err[np.newaxis, :], positions.shape[0], axis=0)

        missplacement_score = np.sum((err - counts), axis=1)
        top_missplacement_index = np.argmax(missplacement_score)
        
        return top_missplacement_index



    def copy(self):
        c = DistributionChecker(self.atom_count)
        c.margins = self.margins
        c.distribution = self.distribution
        c.distances = self.distances

        return c


"""()
positions = np.array([[1, 1, 1], [2, 2, 2], [3, 3 , 3]])



c = DistributionChecker(target_atom_count=4)
dist = np.array([[0, 0, 0], [0, 0, 0]])
c.run(dist, plot = False)


d = c.add_one_atom_vectorized(positions)
d0 = c.add_one_atom(positions[0], check_only=True)


print(d[0])
print(d0)

"""
