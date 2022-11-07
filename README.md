# PartitionedStructures.jl: Partitioned derivatives storage and partitioned quasi-Newton updates

| **Documentation** | **Linux/macOS/Windows/FreeBSD** | **Coverage** | **DOI** |
|:-----------------:|:-------------------------------:|:------------:|:-------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] [![build-cirrus][build-cirrus-img]][build-cirrus-url] | [![codecov][codecov-img]][codecov-url] | [![doi][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://JuliaSmoothOptimizers.github.io/PartitionedStructures.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://JuliaSmoothOptimizers.github.io/PartitionedStructures.jl/dev
[build-gh-img]: https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl/workflows/CI/badge.svg?branch=main
[build-gh-url]: https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl/actions
[build-cirrus-img]: https://img.shields.io/cirrus/github/JuliaSmoothOptimizers/PartitionedStructures.jl?logo=Cirrus%20CI
[build-cirrus-url]: https://cirrus-ci.com/github/JuliaSmoothOptimizers/PartitionedStructures.jl
[codecov-img]: https://codecov.io/gh/JuliaSmoothOptimizers/PartitionedStructures.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/gh/JuliaSmoothOptimizers/PartitionedStructures.jl
[doi-img]: https://zenodo.org/badge/346861459.svg
[doi-url]: https://zenodo.org/badge/latestdoi/346861459

## How to cite

If you use PartitionedStructures.jl in your work, please cite using the format given in [CITATION.bib](CITATION.bib).

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
We write a partially-separable function $f: \mathbb{R}^n \to \mathbb{R}$ in the form

$$  
f(x) = \sum_{i=1}^N f_i (U_i(x)), f_i : \mathbb{R}^{n_i} \to \mathbb{R}, U_i \in \mathbb{R}^{n_i \times n}, n_i \ll n,
$$

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
Check the [tutorial](https://JuliaSmoothOptimizers.github.io/PartitionedStructures.jl/stable/tutorial/).

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
The structures defined here are used in the module:
- [PartiallySeparableSolvers.jl](https://github.com/JuliaSmoothOptimizers/PartiallySeparableSolvers.jl) inside a trust-region method using partitioned quasi-Newton operators;
- [PartitionedVectors.jl](https://github.com/paraynaud/PartitionedVectors.jl) which relies on `Elemental_pv` to make `PartitionedVector <: AbstractVector`;
- [PartitionedKnetNLPModels.jl](https://github.com/paraynaud/PartitionedKnetNLPModels.jl) to train a classification neural network with a limited-memory partitioned quasi-Newton stochastic method.

# Bug reports and discussions

If you think you found a bug, feel free to open an [issue](https://github.com/JuliaSmoothOptimizers/PartitionedStructures.jl/issues).
Focused suggestions and requests can also be opened as issues. Before opening a pull request, start an issue or a discussion on the topic, please.

If you want to ask a question not suited for a bug report, feel free to start a discussion [here](https://github.com/JuliaSmoothOptimizers/Organization/discussions). This forum is for general discussion about this repository and the [JuliaSmoothOptimizers](https://github.com/JuliaSmoothOptimizers), so questions about any of our packages are welcome.
