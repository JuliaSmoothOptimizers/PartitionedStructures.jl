# PartitionedStructures.jl : A partitioned storage for the derivatives of a partially separable function 

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


## Motivation

The module implements some partitioned structures, the derivatives of the partially separable functions
$$
f(x) = \sum_{=1}^N \hat{f}_i (U_i x), \quad f \in \R^n \to \R, \quad \hat f_i:\R^{n_i} \to \R, \quad U_i \in \R^{n_i \times n}.
$$
$f$ is a sum of element functions $\hat{f}_i$, and usually $n_i \ll n$. $U_i$ is a linear operator, it selects the variables used by $\hat{f}_i$.

Consequently, the gradient 
$$
\nabla f(x) = \sum_{i=1}^N U_i^\top \nabla \hat{f}_i (U_i x),
$$
and the hessian 
$$
\nabla^2 f(x) = \sum_{i=1}^N U_i^\top \nabla^2 \hat{f_i} (U_i x) U_i,
$$
are the sum of the element derivatives $\nabla \hat{f}_i,  \nabla^2\hat{f}_i$. 
This structure allows to define partitioned quasi-Newton approximation of $\nabla^2 f$
$$
B = \sum_{i=1}^N U_i^\top \hat{B}_{i} U_i
$$
such that each $\hat{B}_i \approx \nabla^2 \hat{f}_i$.
Contrary to the BFGS and SR1 updates, respectively of rank 1 and 2, the rank of update $B$ is proportionnal to $\min(N,n)$.

## Content
PartitionedStructures.jl implements :
- the partitioned quasi-Newton linear operator : PBFGS, PSR1, PSE
- the limited-memory partitioned quasi-Newton linear operator: PLBFGS, PLSR1, PLSE

```julia
using PartitionedStructures

N = 15 # number of elements
n = 20 # size of the structure
nie = 5 # size of the elements
element_variables = map( (i -> rand(1:n,nie) ),1:N) # list of element variables

PSR1_operator = identity_epm(element_variables) # PSR1 operator
PBFGS_operator = identity_epm(element_variables) # PBFGS operator
PSE_operator = identity_epm(element_variables) # PSE operator
PLBFGS_operator = identity_eplom_LBFGS(element_variables, N, n) # PLBFGS operator
PLSR1_operator = identity_eplom_LSR1(element_variables, N, n) # PLSR1 operator
PLSE_operator = identity_eplom_LOSE(element_variables, N, n) # PLSE operator

```
- the partitioned vectors require by $\nabla f(x) = \sum_{i=1}^N U_i^\top \nabla \hat{f}_i (U_i x)$
```julia
partitioned_vector = epv_from_epm(PLSR1_operator) # keep the same partially separable structure
```
Any quasi-Newton operator can be update from a partitioned gradient difference `partitioned_gradient_difference` and a step `s`, a vector of size $n$.

```julia
partitioned_gradient_difference = epv_from_epm(PLSR1_operator) 
s = rand(n)
update(PBFGS_operator, partitioned_gradient_difference, s; name=:pbfgs)
update(PSR1_operator, partitioned_gradient_difference, s; name=:psr1)
update(PSE_operator, partitioned_gradient_difference, s; name=:pse)
update(PLSR1_operator, partitioned_gradient_difference, s)
update(PLBFGS_operator, partitioned_gradient_difference, s)
update(PLSE_operator, partitioned_gradient_difference, s)
```

These structures are applied in the module 
[PartiallySeparableSolvers.jl](https://github.com/paraynaud/PartiallySeparableSolvers.jl) inside a trust-region using partitioned quasi-Newton operators and in [PartitionedKnetNLPModel.jl](https://github.com/paraynaud/PartitionedKnetNLPModels.jl) to train a neural network of classification with a limited-memory partitioned quasi-Newton stochastic method.

## How to install

```
julia> ]
pkg> add https://github.com/paraynaud/PartitionedStructures.jl
pkg> test PartitionedStructures
```