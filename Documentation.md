# Configuration of Nacl Tetramers.

Sodium chloride, table salt, is an ionic crystal with a simple interleaved face-centered cubic structure of the Na and Cl ions. # Binding Energy of a Macroscopic Ionic Crystal
The binding energy of a macroscopic ionic crystal is often represented using the following formulas:

1. Coulomb Energy Form:
   E_B = (N * F * e^2) / (4 * pi * ε₀ * r₀) * 2A * exp(-2r₀ / r) / r!G ~ 1
2. Alternative Form:
   E_B = (N * F * e^2) / (4 * pi * ε₀ * r₀) * 2A * (8r₀) / (nG) ~ 2
Where:
- E_B is the binding energy.
- N is the number of ion pairs.
- r₀ is the nearest-neighbor equilibrium distance.
- e is the fundamental unit of charge.
- ε₀ is the permittivity of free space.
- A, F, and G are constants.
- r is a constant with a value of 0.317 Å for NaCl.
- n usually takes the value of 9.1 for NaCl.
The first term represents the Coulomb energy, and the second term represents the repulsive Pauli energy.

#### Coulomb Potential for NaCl Ions
The ions Na¹ and Cl² exhibit noble characteristics with closed electron shells, resulting in spherically symmetric shapes. Consequently, the Coulomb potential for ions at large separations simplifies to a two-point charge interaction. As the ions approach each other, electron charge distributions start to overlap, inducing dipole moments.
In the rigid sphere approximation, the two-body potential includes the sum of the attractive or repulsive point charge Coulomb energy and the repulsive Pauli energy. For the repulsive interaction, the Born–Mayer form (Eq. 1) is employed:


U_ij = (e^2 / (4 * π * ε₀ * r_ij)) * A_ij * exp(-2 * r_ij / r)


Here, \( r_{ij} \) is the separation between ions i and j. The coefficients A and r depend on the specific ion interaction:

- For Na–Na interactions: A_{ij} = 23.80 eV
- For Cl–Cl interactions: A_{ij} = 3485.23 eV
- For Na–Cl interactions: A_{ij} = 1254.53 eV
- The separation constant: r = 0.317 Å

These coefficients were derived from crystal data analysis by Tosi and Fumi, as compiled in Ref. 6. The potential function effectively describes ion–ion interactions in NaCl, accounting for the distinct electron cloud radii of alkali and halide ions.

# Homework Objectives

1. **Configuration Selection:**
   - Among the seven configurations of tetramers of NaCl, I have chosen these two configurations.
     <p align="center">
       <img src="https://github.com/poojashresthacode/23-Homework7G3/assets/143756553/f1ca6f72-aa41-4b76-9919-a761d9769652" alt="Configuration 1" width="400"/>
        <p align="center">
       <img src="https://github.com/poojashresthacode/23-Homework7G3/assets/143756553/1b7b2d7d-d6f3-40e9-a0bf-a66a8f8fc19e" alt="Configuration 2" width="400"/>
     </p>
2. **Plot the Equilibrium Configuration :**
   To plot the equilibrium configurations and determine the final configuration.


3. **Code Linting:**
   - To perform linting for code cleanliness.
  
  ## Cluster Optimization Code for Cubic Configuration with Docstrings

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import scipy.optimize

# Constants
ke2 = 197 / 137  # eV-nm   Coulomb force charge
alpha = 1.09e3   # eV      parameter of model
rho = 0.0321     # nm      parameter of model
b = 1.0          # eV      regular
c = 0.01         # nm

# Helpful solution to convert itertools combinations to numpy arrays:
# https://stackoverflow.com/questions/33282369/convert-itertools-array-into-numpy-array
def cp(l):
    return np.fromiter(itertools.chain(*itertools.combinations(l, 2)), dtype=int).reshape(-1, 2)

class Cluster:
    def __init__(self, r_na, r_cl):
        '''
        Inputs the list of Na and Cl positions. Na has charge +1, Cl has -1.
        The array of ions itself does not change throughout the calculation, and
        neither do the charges. As such, we can just compute the combinations one time
        and refer to it throughout the calculation.
        '''
        self.positions = np.concatenate((r_na, r_cl))
        self.charges = np.concatenate([np.ones(r_na.shape[0]), np.full(r_cl.shape[0], -1)])
        self.combs = cp(np.arange(self.charges.size))
        self.chargeprods = self.charges[self.combs][:, 0] * self.charges[self.combs][:, 1]
        self.rij = np.linalg.norm(self.positions[self.combs][:, 0] - self.positions[self.combs][:, 1], axis=1)

    def Vij(self):
        '''Calculate a numpy vector of all of the potentials of the combinations'''
        self.Vij_ = np.zeros_like(self.rij)
        pos = self.chargeprods > 0
        neg = ~pos
        self.Vij_[pos] = ke2 / self.rij[pos] + b * (c / self.rij[pos]) ** 12
        self.Vij_[neg] = -ke2 / self.rij[neg] + alpha * np.exp(-self.rij[neg] / rho) + b * (c / self.rij[neg]) ** 12
        return self.Vij_

    def V(self):
        '''Total potential, which is a sum of the Vij vector'''
        return np.sum(self.Vij())

    def get_vals(self):
        '''Positions interpreted as a flat shape'''
        return np.reshape(self.positions, -1)

    def set_vals(self, vals):
        '''Inputs flat shape of positions, used by __call__'''
        self.positions = vals.reshape(self.positions.shape)
        self.rij = np.linalg.norm(self.positions[self.combs][:, 0] - self.positions[self.combs][:, 1], axis=1)

    def __call__(self, vals):
        '''Function that  scipy.optimize.minimize will call'''
        self.set_vals(vals)
        return self.V()

# Initial "ideal" configuration for Na4Cl4 tetramer
a = 0.2
r_na_ideal = np.array([[0, 0, 0], [a, a, 0], [a, 0, a], [0, a, a]])
r_cl_ideal = np.array([[a, 0, 0], [0, a, 0], [a, a, a], [0, 0, a]])

# Create Cluster instance using ideal positions
cluster = Cluster(r_na_ideal, r_cl_ideal)
vals_init = cluster.get_vals()

print('initial Na positions:\n', r_na_ideal)
print('initial Cl positions:\n', r_cl_ideal)
print('initial positions flattened shape:\n', vals_init)
print('initial V  :', cluster.V())

res = scipy.optimize.minimize(fun=cluster, x0=vals_init, tol=1e-3, method="BFGS")
cluster.set_vals(res.x)  # For some reason, "minimize" is not updating the class at the last iteration
print("Final optimized cluster positions")
print(cluster.positions)
print("Final potential:", res.fun)

%matplotlib inline

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

charges = cluster.charges
x, y, z = cluster.positions[:, 0], cluster.positions[:, 1], cluster.positions[:, 2]

# Scatter plot with shading (adjust alpha for transparency)
scatter = ax.scatter(x, y, z, c=charges, cmap='coolwarm', s=20, alpha=0.6)

# Add a surface plot for background shading
ax.plot_trisurf(x, y, z, color='gray', alpha=0.1, linewidth=0)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Scatter Plot of Nacl Cubic Configuration')

# Adjust grid and axes visibility
ax.grid(True)
ax.set_axis_on()

# Set background color
ax.set_facecolor('lightgrey')

# Set specific limits for each axis
ax.set_xlim([min(x), max(x)])
ax.set_ylim([min(y), max(y)])
ax.set_zlim([min(z), max(z)])

# Add legend if needed
# ax.legend()

# Add annotations or arrows if needed
# ax.text(x_coord, y_coord, z_coord, 'Annotation')

plt.show()
```

## Steps Description:
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import scipy.optimize
```
## A. The necessary libraries are imported.# Python Libraries Import Statements

#### 1. NumPy (`import numpy as np`):
   - NumPy is a powerful library for numerical and mathematical operations in Python.
   - It supports large, multi-dimensional arrays and matrices.
   - The alias `as np` is commonly used to simplify referencing the library in the code.

#### 2. Matplotlib (`import matplotlib.pyplot as plt`):
   - Matplotlib is a comprehensive library for creating visualizations in Python.
   - The `pyplot` module provides functions for creating plots.
   - The alias `as plt` is commonly used to simplify referencing the module in the code.

#### 3. mpl_toolkits.mplot3d (`from mpl_toolkits.mplot3d import Axes3D`):
   - This module is an extension of Matplotlib, specifically designed for creating 3D plots.
   - The `Axes3D` class allows the creation of 3D axes for plotting.

#### 4. itertools (`import itertools`):
   - The itertools module provides tools for working with iterators in a memory-efficient manner.
   - It is commonly used for combinations and permutations of elements.

#### 5. SciPy Optimize (`import scipy.optimize`):
   - SciPy is an open-source library for mathematics, science, and engineering.
   - The `scipy.optimize` module offers optimization algorithms for finding function minima.

## B  Constants Description
```python
ke2 = 197 / 137 # eV-nm   Coulomb force charge
alpha = 1.09e3  # eV      parameter of model
rho = 0.0321    # nm      parameter of model
b = 1.0         # eV      regular
c = 0.01        # nm
```
#### 1. Coulomb Force Charge (`ke2 = 197 / 137`):
   - `ke2` represents the Coulomb force charge, calculated as the ratio of 197 to 137.
   - Unit: eV-nm
   - Coulomb force charge is a measure of the electrostatic force between charged particles.

#### 2. Parameter of Model (`alpha = 1.09e3`):
   - `alpha` is a parameter in the model.
   - Unit: eV
   - This parameter is part of a mathematical model and is likely used to adjust the behavior of the model based on empirical or theoretical considerations.

#### 3. Parameter of Model (`rho = 0.0321`):
   - `rho` is another parameter in the model.
   - Unit: nm
   - Similar to `alpha`, it is likely used to fine-tune the behavior of the model.

#### 4. Regular Parameter (`b = 1.0`):
   - `b` is a regular parameter in the model.
   - Unit: eV
   - It represents a regular term, possibly used to model a constant or linear component in the system.

#### 5. Regular Parameter (`c = 0.01`):
   - `c` is another regular parameter in the model.
   - Unit: nm
   - Similar to `b`, it represents a regular term, possibly used to model a constant or linear component in the system.


```python
def cp(l):
    return np.fromiter(itertools.chain(*itertools.combinations(l,2)),dtype=int).reshape(-1,2)
```
This function is particularly useful when one needs to efficiently generate pairs of combinations for further analysis or calculations. The `cp` function is designed to generate all unique pairs of combinations from a given list `l` using the `itertools.combinations` function. The function returns a NumPy array containing these combinations.

```python
class Cluster:
    def __init__(self, r_na, r_cl):
        self.positions = np.concatenate( (r_na,r_cl))
        self.charges = np.concatenate( [np.ones(r_na.shape[0]), np.full(r_cl.shape[0], -1)] )
        self.combs = cp(np.arange(self.charges.size))
        self.chargeprods = self.charges[self.combs][:,0] * self.charges[self.combs][:,1]
        self.rij = np.linalg.norm(self.positions[self.combs][:,0] - self.positions[self.combs][:,1], axis=1)
```


#### Cluster Class Explanation
The `Cluster` class is designed to represent a cluster of ions with positive and negative charges. This class is particularly useful for initializing and managing the state of the ion cluster, preparing it for subsequent calculations related to ion interactions. In summary, the Cluster class is an efficient way to organize and initialize the state of an ion cluster. It sets up crucial properties such as ion positions, charges, combinations, charge products, and distances, providing a foundation for subsequent calculations involving interactions within the ion cluster.

#### Initialization Method
The initialization method of the `Cluster` class performs the following tasks:

It concatenates the positions of sodium (Na) and chlorine (Cl) ions into a single array (self.positions) and assigns charges to the ions, where sodium (Na) ions have a charge of +1 (represented by an array of ones), and chlorine (Cl) ions have a charge of -1 (represented by an array of -1s). It then, computes combinations of indices for the ions, creating pairs for further calculations. It computes the product of charges for each ion pair, extracting charges using indices from self.combs and calculates the Euclidean distance (norm) between the positions of Na and Cl ions for each pair, considering the pairs defined by self.combs.

``` python
self.Vij_ = np.zeros_like(self.rij)
```

It initializes an array (self.Vij_) to store the potentials of ion pairs. It has the same shape as the distances array (self.rij).

```python
pos = self.chargeprods > 0
neg = ~pos
```

It separates ion pairs into two categories - pos for pairs with positive charge product and neg for pairs with negative charge product.

```python
self.Vij_[pos] = ke2 / self.rij[pos] + b * (c / self.rij[pos])**12
```
It calculates the potential for ion pairs with a positive charge product using a Coulombic term .

```python
self.Vij_[neg] = -ke2 / self.rij[neg] + alpha * np.exp(-self.rij[neg] / rho) + b * (c / self.rij[neg])**12
```
It calculates the potential for ion pairs with a negative charge product using an exponentially decaying term and a Lennard-Jones potential term.

```python
return self.Vij_
```
Returns the calculated potentials for all ion pairs.

The Vij method efficiently calculates potentials for all combinations of ions in the cluster, taking into account different potential functions based on the charge interactions between ions.

```python
def V(self):
        '''Total potential, which is a sum of the Vij vector'''
        return np.sum(self.Vij())
```
The V method provides a convenient way to calculate the total potential energy of the ion cluster. It leverages the Vij method to obtain the potentials for ion pairs and sums them up to yield the overall potential energy of the entire ion cluster.
     
```python
def get_vals(self):
        '''Positions interpreted as a flat shape'''
        return np.reshape(self.positions, -1)
```
The get_vals method simplifies the retrieval of ion positions by providing them in a flat shape. This can be useful for scenarios where a one-dimensional representation of positions is required or when interacting with functions that expect flattened arrays.

```python
def set_vals(self, vals ):
        '''Inputs flat shape of positions, used by __call__'''
        self.positions = vals.reshape(self.positions.shape)
        self.rij = np.linalg.norm(self.positions[self.combs][:,0] - self.positions[self.combs][:,1], axis=1)
```

The set_vals method provides a mechanism for updating the ion positions within the Cluster class. It ensures that the positions are reshaped appropriately and recomputes ion distances, maintaining the consistency of the ion cluster's state. This method is designed to be utilized by the __call__ function, allowing for dynamic updates to the ion cluster.

```python
def __call__(self, vals):
        '''Function that  scipy.optimize.minimize will call'''
        self.set_vals(vals)
        return self.V()
```
The __call__ method serves as the objective function that the optimization algorithm, specifically scipy.optimize.minimize, will use during minimization. It encapsulates the process of updating ion positions and computing the potential energy. This design allows for seamless integration with optimization routines, making it easy to find the minimum energy state of the ion cluster.

```python
# Initial "ideal" configuration for Na4Cl4 tetramer
a = 0.2
r_na_ideal = np.array([[0, 0, 0], [a, a, 0], [a, 0, a], [0, a, a]])
r_cl_ideal = np.array([[a, 0, 0], [0, a, 0], [a, a, a], [0, 0, a]])
```
The code defines the initial "ideal" configuration for a Na₄Cl₄ tetramer. The configuration is specified by setting the positions of sodium (Na) and chlorine (Cl) ions in 3D space. The variable a represents a distance parameter, presumably determining the separation between ions. The positions of sodium ions (r_na_ideal) are defined as a 4x3 NumPy array. Each row represents the (x, y, z) coordinates of a sodium ion. The positions of chlorine ions (r_cl_ideal) are similarly defined as a 4x3 NumPy array, representing the (x, y, z) coordinates of chlorine ions.

```python
# Create Cluster instance using ideal positions
cluster = Cluster(r_na_ideal, r_cl_ideal)
vals_init = cluster.get_vals()
print('initial Na positions:\n', r_na_ideal)
print('initial Cl positions:\n', r_cl_ideal)
print('initial positions flattened shape:\n', vals_init)
print('initial V  :', cluster.V())
```

This code is useful for initializing a Cluster instance with a specific configuration, and it provides a convenient way to retrieve the flattened positions for further use, such as initializing optimization algorithms or simulations and print information about initial configuration.

```python
res = scipy.optimize.minimize( fun=cluster, x0=vals_init, tol=1e-3, method="BFGS")
cluster.set_vals(res.x)  # For some reason, "minimize" is not updating the class at the last iteration
print ("Final optimized cluster positions")
print(cluster.positions)
print("Final potential:", res.fun)
```
This code uses the scipy.optimize.minimize function to perform an optimization on the Cluster instance, starting from the initial flattened positions (vals_init). It then updates the Cluster instance with the optimized positions and prints the final optimized cluster positions along with the corresponding potential energy. 

```python
%matplotlib inline

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract coordinates and charges from the Cluster instance
charges = cluster.charges
x, y, z = cluster.positions[:, 0], cluster.positions[:, 1], cluster.positions[:, 2]

# Scatter plot with shading (adjust alpha for transparency)
scatter = ax.scatter(x, y, z, c=charges, cmap='coolwarm', s=20, alpha=0.6)

# Add a surface plot for background shading
ax.plot_trisurf(x, y, z, color='gray', alpha=0.1, linewidth=0)

# Set labels for each axis
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Set the title of the plot
ax.set_title('3D Scatter Plot of NaCl Cubic Configuration')

# Adjust grid and axes visibility
ax.grid(True)
ax.set_axis_on()

# Set background color
ax.set_facecolor('lightgrey')

# Set specific limits for each axis
ax.set_xlim([min(x), max(x)])
ax.set_ylim([min(y), max(y)])
ax.set_zlim([min(z), max(z)])

# Show the plot
plt.show()
```

%matplotlib inline is a Jupyter Notebook magic command that ensures the plots are displayed within the notebook itself. Import matplotlib.pyplot as plt imports the Matplotlib library for plotting. from mpl_toolkits.mplot3d import Axes3D helps to imports the 3D plotting toolkit from Matplotlib.
The code creates a 3D scatter plot (scatter) of ion positions with color-coded charges using the 'coolwarm' colormap. It also adds a surface plot for background shading to give a sense of the overall configuration. Labels, title, and other formatting elements are added to enhance the visualization. The plt.show() command displays the plot.
This code snippet is useful for visually inspecting the spatial arrangement of ions in the NaCl cluster in a 3D space, with color-coded charges for additional information.

## Results of Cubic Configuration:

```python
initial Na positions:
 [[0.  0.  0. ]
 [0.2 0.2 0. ]
 [0.2 0.  0.2]
 [0.  0.2 0.2]]
initial Cl positions:
 [[0.2 0.  0. ]
 [0.  0.2 0. ]
 [0.2 0.2 0.2]
 [0.  0.  0.2]]
initial positions flattened shape:
 [0.  0.  0.  0.2 0.2 0.  0.2 0.  0.2 0.  0.2 0.2 0.2 0.  0.  0.  0.2 0.
 0.2 0.2 0.2 0.  0.  0.2]
initial V  : -16.037660834699643 eV
```
This is the configuration of the NaCl cluster, both in its original 3D representation and as a flattened array. The initial total potential energy indicates the energy associated with this configuration which is -16.037660834699643 eV.

```python
Final optimized cluster positions
[[-0.03000292 -0.03005593 -0.03002371]
 [ 0.22998047  0.22997554 -0.03003224]
 [ 0.22997735 -0.02999559  0.23001184]
 [-0.03005587  0.22999045  0.22994703]
 [ 0.23000552 -0.03003089 -0.0299969 ]
 [-0.03002847  0.22995303 -0.03005858]
 [ 0.22995278  0.23001306  0.22997573]
 [-0.03003057 -0.03001869  0.22998466]]
Final potential: -28.23583056299211 eV
```
These results represent the optimized configuration of the NaCl cluster after the optimization process. The total potential energy has been minimized to the final value of -28.23583056299211 eV . The positions are presented both in their 3D format and as a flattened array.

### Configuration Obtained:
![Cubic Configuration of Nacl Tetramer with Binding energy -28.23583056299211 eV](https://github.com/poojashresthacode/23-Homework7G3/blob/Documentation.md/Cubic_Nacl.png)


## Initialization for the Tetrahedron:

```python
r_na = np.array([[0, 0, 0], [0, 0, 2*a], [0, 2*a, -a/2], [0, a, a]])
r_cl = np.array([[0, 0, a], [0, a, 2*a], [a, a, 0], [a, -a, 0]])
```
## Results of tetrahedron Configuration:

```python
initial Na positions:
 [[ 0.   0.   0. ]
 [ 0.   0.   0.4]
 [ 0.   0.2  0.2]
 [ 0.   0.4 -0.1]]
initial Cl positions:
 [[ 0.   0.   0.2]
 [ 0.   0.2  0.4]
 [ 0.2  0.2  0. ]
 [ 0.2 -0.2  0. ]]
initial positions flattened shape:
 [ 0.   0.   0.   0.   0.   0.4  0.   0.2  0.2  0.   0.4 -0.1  0.   0.
  0.2  0.   0.2  0.4  0.2  0.2  0.   0.2 -0.2  0. ]
initial V  : -18.417354969091683 eV.
```
This information provides an overview of the initial configuration of the NaCl cluster with tetrahedron structure, both in its original 3D representation and as a flattened array. The initial total potential energy indicates the energy associated with this configuration which is -18.417354969091683 eV.

```python
Final optimized cluster positions
[[ 0.19421003  0.0042452   0.06015449]
 [ 0.07512365 -0.27734087  0.25410086]
 [-0.09468067  0.194785    0.21432129]
 [ 0.02441364  0.47637688  0.02038706]
 [-0.01617159 -0.04496002  0.24049382]
 [-0.15771362  0.43377828  0.17650278]
 [ 0.11570388  0.24399271  0.03398245]
 [ 0.25724764 -0.23474607  0.09797901]]
Final potential: -27.729842282541327 eV.
```
The optimized positions of tetrahedron structure  provided correspond to the final configuration of the NaCl cluster following the completion of the optimization procedure. The total potential energy has been minimized, achieving the specified final value of -27.729842282541327 eV. The positions are presented in both their three-dimensional format and as a flattened array.

## Configuration Obtained:
![Tetrahedron Configuration of Nacl Tetramer with Binding energy -27.729842282541327 eV.](https://github.com/poojashresthacode/23-Homework7G3/blob/Documentation.md/tetrahedron.png)















