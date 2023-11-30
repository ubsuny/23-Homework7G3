import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import scipy.optimize

# Constants
ke2 = 197 / 137  # eV-nm   Coulomb force charge
alpha = 1.09e3  # eV      parameter of the model
rho = 0.0321    # nm      parameter of the model
b = 1.0         # eV      regular
c = 0.01        # nm

# Helpful solution to convert itertools combinations to numpy arrays here:
# https://stackoverflow.com/questions/33282369/convert-itertools-array-into-numpy-array
def cp(l):
    """
    Convert itertools combinations to a numpy array.

    Parameters:
    - l: Iterable

    Returns:
    - np.array
    """
    return np.fromiter(itertools.chain(*itertools.combinations(l, 2)), dtype=int).reshape(-1, 2)

class Cluster:
    def __init__(self, r_na, r_cl):
        '''
        Initialize the Cluster object.

        Parameters:
        - r_na: numpy array, positions of Na atoms
        - r_cl: numpy array, positions of Cl atoms
        '''
        self.positions = np.concatenate((r_na, r_cl))
        self.charges = np.concatenate([np.ones(r_na.shape[0]), np.full(r_cl.shape[0], -1)])
        self.combs = cp(np.arange(self.charges.size))
        self.chargeprods = self.charges[self.combs][:, 0] * self.charges[self.combs][:, 1]
        self.rij = np.linalg.norm(self.positions[self.combs][:, 0] - self.positions[self.combs][:, 1], axis=1)

    def Vij(self):
        '''
        Calculate a numpy vector of all the potentials of the combinations.

        Returns:
        - numpy array
        '''
        self.Vij_ = np.zeros_like(self.rij)
        pos = self.chargeprods > 0
        neg = ~pos
        self.Vij_[pos] = ke2 / self.rij[pos] + b * (c / self.rij[pos]) ** 12
        self.Vij_[neg] = -ke2 / self.rij[neg] + alpha * np.exp(-self.rij[neg] / rho) + b * (
                    c / self.rij[neg]) ** 12
        return self.Vij_

    def V(self):
        '''
        Calculate the total potential, which is a sum of the Vij vector.

        Returns:
        - float
        '''
        return np.sum(self.Vij())

    def get_vals(self):
        '''
        Return positions interpreted as a flat shape.

        Returns:
        - numpy array
        '''
        return np.reshape(self.positions, -1)

    def set_vals(self, vals):
        '''
        Set positions using a flat shape of positions.

        Parameters:
        - vals: numpy array
        '''
        self.positions = vals.reshape(self.positions.shape)
        self.rij = np.linalg.norm(self.positions[self.combs][:, 0] - self.positions[self.combs][:, 1], axis=1)

    def __call__(self, vals):
        '''
        Function that scipy.optimize.minimize will call.

        Parameters:
        - vals: numpy array

        Returns:
        - float
        '''
        self.set_vals(vals)
        return self.V()

# Initial configuration
a = 2.5
r_na = np.array([[0, 0, 0], [a/2, a/2, 0], [a/2, 0, a/2], [0, a/2, a/2]])
r_cl = np.array([[a/2, 0, 0], [0, a/2, 0], [a/2, a/2, a/2], [0, 0, a/2]])

# Create Cluster object
cluster = Cluster(r_na, r_cl)
vals_init = cluster.get_vals()

# Display initial information
print('initial Na positions:\n', r_na)
print('initial Cl positions:\n', r_cl)
print('initial positions flattened shape:\n', vals_init)
print('initial V  :', cluster.V())

# Optimize the configuration
res = scipy.optimize.minimize(fun=cluster, x0=vals_init, tol=1e-3, method="BFGS")
cluster.set_vals(res.x)  # Update the class instance with the final optimized positions

# Display final information
print("Final optimized cluster positions")
print(cluster.positions)
print("Final potential:", res.fun)

# 3D Plot
%matplotlib notebook
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

charges = cluster.charges
x, y, z = cluster.positions[:, 0], cluster.positions[:, 1], cluster.positions[:, 2]
ax.scatter(x, y, z, c=charges, cmap='coolwarm')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
