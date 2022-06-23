# PartitionedStructures.jl : A partitioned storage for the derivatives of partially separable functions

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

## Compatibility
Julia 1 and up.

## How to install
```julia
julia> ]
pkg> add https://github.com/paraynaud/PartitionedStructures.jl
pkg> test PartitionedStructures
```

## How to use
See the [tutorial](https://paraynaud.github.io/PartitionedStructures.jl/dev/tutorial/)

## Main partitioned structures
Structure              | Description
-----------------------|------------
`Part_struct`          | The supertype of every partitioned structure
`Elemental_pm`         | An elemental partitioned matrix, each element matrix is dense
`Elemental_plom_bfgs`  | A limited-memory elemental partitioned matrix, each elemental element matrix is a `LBFGSOperator`
`Elemental_plom_sr1`   | A limited-memory elemental partitioned matrix, each elemental element matrix is a `LSR1Operator`
`Elemental_plom`       | A limited-memory elemental partitioned matrix, each elemental element matrix is a `LBFGSOperator` or a`LSR1Operator`
`Elemental_pv`         | An elemental partitioned vector

## Main methods 
Method                 | Description
-----------------------|------------
`identity_epm`         | Creates a partitioned matrix with identity element matrix
`identity_eplom_LBFGS` | Creates a limited-memory (LBFGS) partitioned matrix
`identity_eplom_LSR1`  | Creates a limited-memory (LSR1) partitioned matrix
`identity_eplom_LOSE`  | Creates a limited-memory (combines LBFGS and LSR1) partitioned matrix
`update`               | Performs a partitioned quasi-Newton update
`eplom_lbfgs_from_epv` | Creates an `Elemental_plom_bfgs` from the partitioned structure of an `Elemental_pv`
`eplom_lsr1_from_epv`  | Creates an `Elemental_plom_sr1` from the partitioned structure of an `Elemental_pv`
`eplom_lose_from_epv`  | Creates an `Elemental_plom` from the partitioned structure of an `Elemental_pv`
`epm_from_epv`         | Creates an `Elemental_pm` from the partitioned structure of an `Elemental_pv`
`epv_from_epm`         | Creates an `Elemental_pv` from the partitioned structure of an `Elemental_pm`
`epv_from_eplom`       | Creates an `Elemental_pv` from the partitioned structure of an `Elemental_plom` or `Elemental_plom_bfgs` or `Elemental_plom_sr1`


## Modules applying [PartitionedStructures.jl](https://github.com/paraynaud/PartitionedStructures.jl)
These structures are applied in the module 
[PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl) inside a trust-region using partitioned quasi-Newton operators and in [PartitionedKnetNLPModel.jl](https://github.com/paraynaud/PartitionedKnetNLPModels.jl) to train a neural network of classification with a limited-memory partitioned quasi-Newton stochastic method.

