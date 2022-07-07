# PartitionedStructures.jl: Partitioned derivatives storage and partitioned quasi-Newton updates

## Compatibility
Julia ≥ 1.6.

## How to install
```julia
pkg> add PartitionedStructures
pkg> test PartitionedStructures
```

## Philosophy
Methods exploiting the derivatives of partially-separable functions require specific data structures to store partitioned derivatives.
There are several types of partial separability.
We write a partially-separable function $f: \R^n \to \R$ in the form
```math
  f(x) = \sum_{i=1}^N f_i (U_i(x)),\; f_i : \R^{n_i} \to \R, \; U_i \in \R^{n_i \times n},\; n_i \ll n
```
where:
* $f_i$ is the $i$-th element function whose dimension is smaller than $f$;
* $U_i$ the linear operator selecting the linear combinations of variables that parametrize $f_i$.

In the case of partitioned quasi-Newton methods, they require storing partitioned gradients and the partitioned Hessian approximation.
[PartitionedStructures.jl](https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl) facilitates the definition of those partitioned structures and defines methods to manipulate them.

## Features
$U_i$ may be based on the *elemental* variables or the *internal* variables of $f_i$:
- the elemental variables represent the subset of variables that parametrizes $f_i$, i.e. the rows of $U_i$ are vectors from the Euclidean basis;
```
Ui = [1,3,5] # i.e. [1 0 0 0 0; 0 0 1 0 0; 0 0 0 0 1]
```
- the internal variables are linear combinations of the variables that parametrize $f_i$, i.e. $U_i$ may be a dense matrix.

The implementation of the linear-operator $U_i$, which describe entirely the partially-separable structure of $f$, changes depending on wether we use internal or elemental variables.

At the moment, we only developed the elemental partitioned structures, but we left the door open to the development of internal partitioned structures in the future.

## How to use
Check the [tutorial](https://JuliaSmoothOptimizers.github.io/PartitionedStructures.jl/dev/tutorial/).

## Partitioned structures available
Structure              | Description
-----------------------|------------
`AbstractPartitionedStructure`| The supertype of every partitioned structures
`Elemental_pm`         | An elemental partitioned matrix, each element-matrix is dense
`Elemental_plo_bfgs`   | A limited-memory elemental partitioned matrix, each element limited-memory operator is a `LBFGSOperator`
`Elemental_plo_sr1`    | A limited-memory elemental partitioned matrix, each element limited-memory operator is a `LSR1Operator`
`Elemental_plo`        | A limited-memory elemental partitioned matrix, each element limited-memory operator is a `LBFGSOperator` or a `LSR1Operator`
`Elemental_pv`         | An elemental partitioned vector

## Methods available
Method                 | Description
-----------------------|------------
`identity_epm`         | Create a partitioned matrix with identity element-matrices
`identity_eplo_LBFGS`  | Create a PLBFGS limited-memory partitioned matrix
`identity_eplo_LSR1`   | Create a PLSR1 limited-memory partitioned matrix
`identity_eplo_LOSE`   | Create a PLSE limited-memory partitioned matrix
`update`               | Performs a partitioned quasi-Newton update on a partitioned matrix
`eplo_lbfgs_from_epv`  | Create an `Elemental_plo_bfgs` from the partitioned structure of an `Elemental_pv`
`eplo_lsr1_from_epv`   | Create an `Elemental_plo_sr1` from the partitioned structure of an `Elemental_pv`
`eplo_lose_from_epv`   | Create an `Elemental_plo` from the partitioned structure of an `Elemental_pv`
`epm_from_epv`         | Create an `Elemental_pm` from the partitioned structure of an `Elemental_pv`
`epv_from_epm`         | Create an `Elemental_pv` from the partitioned structure of an `Elemental_pm`
`epv_from_eplo`        | Create an `Elemental_pv` from the partitioned structure of an `Elemental_plo`, an `Elemental_plo_bfgs` or an `Elemental_plo_sr1`
`mul_epm_epv`          | Return a partitioned vector from an elementwise product between a partitioned matrix and a partitioned vector
`mul_epm_vector`       | Return the vector resulting from a partitioned matrix-vector product
`build_v!`             | Build a vector accumulating the element contributions of a partitioned vector 
`set_epv!`             | Set the value of every element-vector
`minus_epv!`           | Apply a unary minus on every element-vector of a partitioned vector
`add_epv!`             | Perform elementwise addition between two partitioned vectors

## Modules using [PartitionedStructures.jl](https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl)
The structures defined here are used in the module
[PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl) inside a trust-region method using partitioned quasi-Newton operators, and in [PartitionedKnetNLPModels.jl](https://github.com/paraynaud/PartitionedKnetNLPModels.jl) to train a classification neural network with a limited-memory partitioned quasi-Newton stochastic method.