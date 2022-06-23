# PartitionedStructures.jl: Tutorial
This tutorial shows how [PartitionedStructures.jl]() can define partitioned quasi-Newton approximations, by using the partitioned vectors and the partitioned matrices.

## What are the partially separable structure and the partitioned quasi-Newton approximations
The quasi-Newton methods exploiting the partially separable function 
```math
 f(x) = \sum_{i=1}^N f_i (U_i) : \R^n \to \R,\; f_i : \R^{n_i} \to \R, \; U_i \in \R^{n_i \times n},\; n_i < n
```
the sum of element function fᵢ.
The gradient ∇f and the Hessian ∇²f
```math
\nabla f(x) = \sum_{i=1}^N U_i^\top f_i (U_i x), \quad \nabla^2 f(x) = \sum_{i=1}^N U_i^\top f_i (U_i x) U_i,
```
accumulate the element derivatives ∇fᵢ and ∇²fᵢ with respect to the Uᵢ.

These partitioned quasi-Newton methods define partitioned quasi-Newton approximations of the Hessian B ≈ ∇²f, such that B accumulates the element Hessian approximations Bᵢ ≈ ∇²fᵢ
```math
B = \sum_{i=1}^N U_i^\top B_i U_i
```
The partitioned quasi-Newton approximations structurally keep the sparsity structure of ∇²f, which is not the case of classical quasi-Newton approximation (BFGS, SR1).
Moreover, the rank of the partitioned updates may be proportional to the number of elements `N`, whereas classical quasi-Newton approximations are low rank updates.
A partitioned quasi-Newton update must update every approximation of element Hessian Bᵢ at each step s.
It requires Bᵢ, Uᵢs and ∇fᵢ(Uᵢ(x+s)) - ∇²fᵢ(Uᵢx), and therefore we have to store individually every ∇²fᵢ(Uᵢx).

#### Reference
* A. Griewank and P. Toint, *On the unconstrained optimization of partially separable functions*, Numerische Nonlinear Optimization 1981, 39, pp. 301--312, 1982.

## Example: the partitioned structure of a quadratic function
Let's take the quadratic function `f` as an example 
```@example PartitionedStructures
f(x) = x[1]^2 + x[2]^2 + x[3]^2 + x[1]*x[2] + 3x[2]*x[3]
```
`f` can be considered as the sum of two element functions
```@example PartitionedStructures
f1(x) = x[1]^2 + x[1]*x[2]
f2(x) = x[1]^2 + x[2]^2 + 3x[1]*x[2]
```
considering U₁ and U₂ informing the variables required by each element function.
```@example PartitionedStructures
U1 = [1, 2] # [1 0 0; 0 1 0] as a matrix
U2 = [2, 3] # [0 1 0; 0 0 1] as a matrix
```

By gathering the different Uᵢ together
```@example PartitionedStructures
U = [U1, U2]
```
we define the function `f_pss = f` exploiting the partially separable structure as
```@example PartitionedStructures
f_pss(x, U) = f1(x[U[1]]) + f2(x[U[2]])

using Test
x0 = [2., 3., 4.]
@test f(x0) == f_pss(x0, U)
```

Similarly, you can compute: the gradient, the element gradients and explicit how the gradient is partitioned
```@example PartitionedStructures
∇f(x) = [2x[1] + x[2], x[1] + 2x[2] + 3x[3], 2x[3] + 3x[2]]
∇f1(x) = [2x[1] + x[2], x[1]]
∇f2(x) = [2x[1] + 3x[2], 2x[2] + 3x[1]]
function ∇f_pss(x, U)
  gradient = zeros(length(x))
  gradient[U1] = ∇f1(x[U[1]])
  gradient[U2] .+= ∇f2(x[U[2]])
  return gradient
end
@test ∇f(x0) == ∇f_pss(x0, U)
```

However, `∇f_pss` accumulates directly the element gradient and does not store the value of each element gradients `∇f1, ∇f2`.
We would like to store every element gradient, such that afterward it is possible to build the difference element gradients required for the partitioned quasi-Newton update.
Thus, we define the partitioned vector, from `U` and `n`, to store each element gradient and form the ∇f when required
```@example PartitionedStructures
using PartitionedStructures
U = [U1, U2]
n = length(x0)
partitioned_gradient_x0 = create_epv(U, n) # creates the partitioned vector
```
We set the value of each element vector to the corresponding element gradient
```@example PartitionedStructures
vector_gradient_element(x, U) = [∇f1(x[U[1]]), ∇f2(x[U[2]])] :: Vector{Vector{Float64}} # returns every element gradient

set_epv!(partitioned_gradient_x0, vector_gradient_element(x0, U)) # sets each element vector to its corresponding element gradient

build_v!(partitioned_gradient_x0) # builds the gradient vector
@test get_v(partitioned_gradient_x0) == ∇f(x0) # with the same value as the gradient
```
## Approximate the Hessian ∇f²
There is at least two ways to approximate ∇f²: a classical quasi-Newton approximation (ex: BFGS) and a partitioned quasi-Newton approximation.
Both methods are presented, and we expose the sparse structure of partitioned approximation.
### Quasi-Newton approximation of the quadratic (BFGS)
In the case of the BFGS method, you want to approximate the Hessian matrix from `s = x1 - x0`
```@example PartitionedStructures
x1 = [1., 2., 3.]
s = x1 .- x0
```
the gradient difference `y`
```@example PartitionedStructures
y = (∇f(x1) .- ∇f(x0))
```
and the approximation `B`, initially set to the identity
```@example PartitionedStructures
B = [ i==j ? 1. : 0. for i in 1:n, j in 1:n]
```

By applying the BFGS update, you satisfy the secant equation `Bs = y`
```@example PartitionedStructures
B_BFGS = BFGS(s,y,B) # PartitionedStructures.jl implements BFGS

using LinearAlgebra		
@test norm(B_BFGS * s - y) == 0. # numerical verification of the secant equation 
```
but the approximation `B_BFGS` is dense.
```@example PartitionedStructures
B_BFGS
```

### Partitioned quasi-Newton approximation of the quadratic function (PBFGS)
In order to make a sparse quasi-Newton approximation of ∇²f, you may define a partitioned matrix with the same partially separable structure than `partitioned_gradient_x0` where each element matrix is set to the identity
```@example PartitionedStructures
partitioned_matrix = epm_from_epv(partitioned_gradient_x0)
```
```@example PartitionedStructures
Matrix(partitioned_matrix)
```

The second term of the diagonal accumulates two 1.0 from the two initial element approximations.

Then you compute the partitioned gradient at `x1`
```@example PartitionedStructures
partitioned_gradient_x1 = create_epv(U, n)
set_epv!(partitioned_gradient_x1, vector_gradient_element(x1, U))
```
and compute the difference of the partitioned gradients `partitioned_gradient_difference = partitioned_gradient_x1 - partitioned_gradient_x0`
```@example PartitionedStructures
partitioned_gradient_difference = copy(partitioned_gradient_x0) # copy to avoid side effects on partitioned_gradient_x0 
minus_epv!(partitioned_gradient_difference) # applies a unary minus to every element gradient
add_epv!(partitioned_gradient_x1, partitioned_gradient_difference) # add the element vector of partitioned_gradient_x1 to the correspond element vector of partitioned_gradient_difference

build_v!(partitioned_gradient_difference) # computes the vector y
@test get_v(partitioned_gradient_difference) == y
```
Then you can define the partitioned quasi-Newton update PBFGS
```@example PartitionedStructures
B_PBFGS = update(partitioned_matrix, partitioned_gradient_difference, s; name=:pbfgs, verbose=true) # applies the partitioned update PBFGS to partitioned_matrix and returns Matrix(partitioned_matrix)
```

which keeps the sparsity structure of ∇²f.

In addition, `update()` informs the number of element: updated, not updated or untouched, as long as the user don't set `verbose=false`.
The partitioned update verifies the secant equation
```@example PartitionedStructures
@test norm(B_PBFGS*s - y) == 0.
```
which may also be calculated with 
```@example PartitionedStructures
Bs = mul_epm_vector(partitioned_matrix, s) # compute the product partitioned-matrix vector
@test norm(Bs - y) == 0.
```

## Others partitioned quasi-Newton approximations
There exist two categories of partitioned quasi-Newton updates.
In the first category, each element Hessian ∇²fᵢ is approximate with a dense matrix, for example: PBFGS.
In the second category, each element Hessian ∇²fᵢ is approximate with a quasi-Newton linear operator.

### Partitioned quasi-Newton operators
Once the partitioned matrix is allocated, 
```@example PartitionedStructures
partitioned_matrix_PBFGS = epm_from_epv(partitioned_gradient_x0)
partitioned_matrix_PSR1 = epm_from_epv(partitioned_gradient_x0)
partitioned_matrix_PSE = epm_from_epv(partitioned_gradient_x0)
```
you can apply on it any of the three partitioned updates : PBFGS, PSR1, PSE (by default) : 
- PBFGS update each element approximation with BFGS;
- PSR1 update each element approximation with SR1;
- PSE each element approximation is update with BFGS if it is possible or with SR1 otherwise.
```@example PartitionedStructures
B_PBFGS = update(partitioned_matrix_PBFGS, partitioned_gradient_difference, s; name=:pbfgs)
B_PSR1 = update(partitioned_matrix_PSR1, partitioned_gradient_difference, s; name=:psr1)
B_PSE = update(partitioned_matrix_PSE, partitioned_gradient_difference, s) # ; name=:pse by default
```
All these methods satisfy the secant equation as long as every element approximation is update
```@example PartitionedStructures
@test norm(mul_epm_vector(partitioned_matrix_PBFGS, s) - y) == 0.
@test norm(mul_epm_vector(partitioned_matrix_PSR1, s) - y) == 0.
@test norm(mul_epm_vector(partitioned_matrix_PSE, s) - y) == 0.
```

### Limited-memory partitioned quasi-Newton operators
These operators are made to apply the partitioned quasi-Newton methods to the partially separable function with large elements, whose element Hessian approximations can't be store by dense matrices.
The limited-memory partitioned quasi-Newton operators allocate for each element Hessian approximation a quasi-Newton operator LBFGS or LSR1 defined in [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).
It defines three approximations:
- PLBFGS, each element approximation is a `LBFGSOperator`;
- PLSR1, each element approximation is a `LSR1Operator` (issue in LinearOperator.jl);
- PLSE, each element approximation may be a `LBFGSOperator` or `LSR1Operator`.

Contrary to the partitioned quasi-Newton operators, each limited-memory version is typed differently
```@example PartitionedStructures
partitioned_linear_operator_PLBFGS = eplom_lbfgs_from_epv(partitioned_gradient_x0)
partitioned_linear_operator_PLSR1 = eplom_lsr1_from_epv(partitioned_gradient_x0)
partitioned_linear_operator_PLSE = eplom_lose_from_epv(partitioned_gradient_x0)
```
The different types simplify the `update` method, since no argument `name` is required to determine the update that will be applied
```@example PartitionedStructures
B_PLBFGS = update(partitioned_linear_operator_PLBFGS, partitioned_gradient_difference, s)
B_PLSE = update(partitioned_linear_operator_PLSE, partitioned_gradient_difference, s)
B_PLSR1 = update(partitioned_linear_operator_PLSR1, partitioned_gradient_difference, s)

@test norm(B_PLBFGS * s - y) == 0.
@test norm(B_PLSE * s - y) == 0.
# @test norm(B_PLSR1 * s - y) == 0. # the second element hessian approximation is not update, since the element step and the gradient element difference are collinear.
```
That's it, you have all the tools to implement a partitioned quasi-Newton method, enjoy!	