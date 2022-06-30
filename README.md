# PartitionedStructures.jl: Partitioned derivatives storage and partitioned quasi-Newton updates

| **Documentation** | **Linux/macOS/Windows/FreeBSD** | **Coverage** | **DOI** |
|:-----------------:|:-------------------------------:|:------------:|:-------:|
| [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] [![build-cirrus][build-cirrus-img]][build-cirrus-url] | [![codecov][codecov-img]][codecov-url] | [![doi][doi-img]][doi-url] |

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://paraynaud.github.io/PartitionedStructures.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://paraynaud.github.io/PartitionedStructures.jl/dev
[build-gh-img]: https://github.com/paraynaud/PartitionedStructures.jl/workflows/CI/badge.svg?branch=main
[build-gh-url]: https://github.com/paraynaud/PartitionedStructures.jl/actions
[build-cirrus-img]: https://img.shields.io/cirrus/github/paraynaud/PartitionedStructures.jl?logo=Cirrus%20CI
[build-cirrus-url]: https://cirrus-ci.com/github/paraynaud/PartitionedStructures.jl
[codecov-img]: https://codecov.io/gh/paraynaud/PartitionedStructures.jl/branch/main/graph/badge.svg
[codecov-url]: https://app.codecov.io/gh/paraynaud/PartitionedStructures.jl
[doi-img]: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.822073-blue.svg
[doi-url]: https://doi.org/10.5281/zenodo.822073

## Philosophy
The methods exploiting the derivatives of partially-separable functions require specific structures to store their partitioned derivatives.
In the case of the partitioned quasi-Newton methods, it requires the storage of partitioned-gradients and the partitioned-matrix.
[PartitionedStructures.jl](https://github.com/paraynaud/PartitionedStructures.jl) facilitates the definition of those partitioned structures and define methods to ease their use.

## Compatibility
Julia 1 up to 1.7.

## How to install
```julia
julia> ]
pkg> add https://github.com/paraynaud/PartitionedStructures.jl
pkg> test PartitionedStructures
```

## How to use
Check the [tutorial](https://paraynaud.github.io/PartitionedStructures.jl/dev/tutorial/).

## Partitioned structures available
Structure              | Description
-----------------------|------------
`Part_struct`          | The (abstract) supertype of every partitioned-structures
`Elemental_pm`         | An elemental partitioned-matrix, each element-matrix is dense
`Elemental_plo_bfgs`   | A limited-memory elemental partitioned-matrix, each elemental element limited-memory operator is a `LBFGSOperator`
`Elemental_plo_sr1`    | A limited-memory elemental partitioned-matrix, each elemental element limited-memory operator is a `LSR1Operator`
`Elemental_plo`        | A limited-memory elemental partitioned-matrix, each elemental element limited-memory operator is a `LBFGSOperator` or a `LSR1Operator`
`Elemental_pv`         | An elemental partitioned-vector

## Methods available
Method                 | Description
-----------------------|------------
`identity_epm`         | Creates a partitioned-matrix with identity element-matrix
`identity_eplo_LBFGS`  | Creates a limited-memory partitioned-matrix PLBFGS
`identity_eplo_LSR1`   | Creates a limited-memory partitioned-matrix PLSR1
`identity_eplo_LOSE`   | Creates a limited-memory partitioned-matrix PLSE
`update`               | Performs a partitioned quasi-Newton update onto a partitioned-matrix
`eplo_lbfgs_from_epv`  | Creates an `Elemental_plo_bfgs` from the partitioned-structure of an `Elemental_pv`
`eplo_lsr1_from_epv`   | Creates an `Elemental_plo_sr1` from the partitioned-structure of an `Elemental_pv`
`eplo_lose_from_epv`   | Creates an `Elemental_plo` from the partitioned-structure of an `Elemental_pv`
`epm_from_epv`         | Creates an `Elemental_pm` from the partitioned-structure of an `Elemental_pv`
`epv_from_epm`         | Creates an `Elemental_pv` from the partitioned-structure of an `Elemental_pm`
`epv_from_eplo`        | Creates an `Elemental_pv` from the partitioned-structure of: an `Elemental_plo`, an `Elemental_plo_bfgs` or an `Elemental_plo_sr1`
`mul_epm_epv`          | Return a partitioned-vector from an elementwise product between a partitioned-matrix and a partitioned-vector
`mul_epm_vector`       | Return the vector resulting of a partitioned-matrix vector product
`build_v!`             | Build the vector associated to a partitioned-vector
`get_v`                | Return the vector associated to a partitioned-vector  **Warning: it doesn't build the vector**
`set_epv!`             | Set the value of every element-vectors
`minus_epv!`           | Applies a unary minus on every element-vector of a partitioned-vector
`add_epv!`             | Performs an elementwise addition between two partitioned-vectors

## Modules applying [PartitionedStructures.jl](https://github.com/paraynaud/PartitionedStructures.jl)
These structures are applied in the module
[PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl) inside a trust-region using partitioned quasi-Newton operators and in [PartitionedKnetNLPModel.jl](https://github.com/paraynaud/PartitionedKnetNLPModels.jl) to train a neural network of classification with a limited-memory partitioned quasi-Newton stochastic method.

## Features
For now, PartitionedStructures.jl supports only the elemental Uᵢ, i.e. a linear-operator where the lines of Uᵢ are vectors from the euclidean basis.
Concretely, each Uᵢ is a vector of size nᵢ who's indicates the indices of the variables used by the i-th element function.