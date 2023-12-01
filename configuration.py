"""
Module docstring: Provide a brief overview of the module here.
"""
!pip install numpy scipy
import itertools
import numpy as np
import scipy.optimize

KE2 = 197 / 137  # eV-nm Coulomb force charge
ALPHA = 1.09e3   # eV parameter of model
RHO = 0.0321     # nm parameter of model
B = 1.0          # eV regular
C = 0.01         # nm

# Helpful solution to convert itertools combinations to numpy arrays here:
# https://stackoverflow.com/questions/33282369/convert-itertools-array-into-numpy-array
def cp(l):
    """
    Convert itertools combinations to numpy array.
    """
    return np.fromiter(itertools.chain(*itertools.combinations(l, 2)), dtype=int).reshape(-1, 2)

class Cluster:
    """
    Class representing a cluster.
    """

    def __init__(self, r_na, r_cl):
        """
        Initialize the cluster with Na and Cl positions.
        """
        self.positions = np.concatenate((r_na, r_cl))
        self.charges = np.concatenate([np.ones(r_na.shape[0]), np.full(r_cl.shape[0], -1)])
        self.combs = cp(np.arange(self.charges.size))
        self.chargeprods = self.charges[self.combs][:, 0] * self.charges[self.combs][:, 1]
        self.rij = np.linalg.norm(self.positions[self.combs][:, 0] - self.positions[self.combs][:, 1], axis=1)

    def vij(self):
        """
        Calculate a numpy vector of potentials for all combinations.
        """
        vij_ = np.zeros_like(self.rij)
        pos = self.chargeprods > 0
        neg = ~pos
        vij_[pos] = KE2 / self.rij[pos] + B * (C / self.rij[pos]) ** 12
        vij_[neg] = -KE2 / self.rij[neg] + ALPHA * np.exp(-self.rij[neg] / RHO) + B * (C / self.rij[neg]) ** 12
        return vij_

    def v(self):
        """
        Calculate the total potential.
        """
        return np.sum(self.vij())

    def get_vals(self):
        """
        Return positions interpreted as a flat shape.
        """
        return np.reshape(self.positions, -1)

    def set_vals(self, vals):
        """
        Set positions using a flat shape.
        """
        self.positions = vals.reshape(self.positions.shape)
        self.rij = np.linalg.norm(self.positions[self.combs][:, 0] - self.positions[self.combs][:, 1], axis=1)

    def __call__(self, vals):
        """
        Function that scipy.optimize.minimize will call.
        """
        self.set_vals(vals)
        return self.v()

# Initial "ideal" configuration for Na4Cl4 tetramer
A = 0.2
R_NA_IDEAL = np.array([[0, 0, 0], [A, A, 0], [A, 0, A], [0, A, A]])
R_CL_IDEAL = np.array([[A, 0, 0], [0, A, 0], [A, A, A], [0, 0, A]])

# Create Cluster instance using ideal positions
CLUSTER = Cluster(R_NA_IDEAL, R_CL_IDEAL)
VALS_INIT = CLUSTER.get_vals()

print('Initial Na positions:\n', R_NA_IDEAL)
print('Initial Cl positions:\n', R_CL_IDEAL)
print('Initial positions flattened shape:\n', VALS_INIT)
print('Initial V:', CLUSTER.v())

RES = scipy.optimize.minimize(fun=CLUSTER, x0=VALS_INIT, tol=1e-3, method="BFGS")
CLUSTER.set_vals(RES.x)
print("Final optimized cluster positions")
print(CLUSTER.positions)
print("Final potential:", RES.fun)
