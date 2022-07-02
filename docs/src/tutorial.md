# PartitionedStructures.jl: Tutorial
This tutorial shows how [PartitionedStructures.jl](https://github.com/paraynaud/PartitionedStructures.jl) can define partitioned quasi-Newton approximations, by using partitioned vectors and partitioned matrices.
Those partitioned structures are strongly related to partially-separable functions.

## What are the partially-separable structure and the partitioned quasi-Newton approximations
The partitioned quasi-Newton methods exploit the partially-separable of $f:\R^n \to \R$
```math
 f(x) = \sum_{i=1}^N f_i (U_i x) : \R^n \to \R,\; f_i : \R^{n_i} \to \R, \; U_i \in \R^{n_i \times n},\; n_i \ll n,
```
the sum of element function $f_i$.
The gradient $\nabla f$ and the Hessian $\nabla^2 f$
```math
\nabla f(x) = \sum_{i=1}^N U_i^\top f_i (U_i x), \quad \nabla^2 f(x) = \sum_{i=1}^N U_i^\top f_i (U_i x) U_i,
```
accumulate the element derivatives $\nabla f_i$ and $\nabla^2 f_i$.

The partitioned structure of the Hessian let us the definition of partitioned quasi-Newton approximations $B \approx \nabla^2 f$, such that $B$ accumulates every element Hessian approximation $B_i \approx \nabla^2 f_i$
```math
B = \sum_{i=1}^N U_i^\top B_i U_i
```
The partitioned quasi-Newton approximations structurally respect sparsity structure of $\nabla^2 f$, which is not the case of classical quasi-Newton approximation (e.g. BFGS, SR1).
Moreover, the rank of the partitioned updates may be proportional to the number of elements $N$, whereas classical quasi-Newton approximations are low-rank updates.
A partitioned quasi-Newton update may update every element-Hessian $B_i$ at each step $s$.
It requires $B_i$, $U_i s$ and $\nabla f_i (U_i (x+s)) - \nabla f_i (U_i x)$, and therefore we have to store such an approximation and vectors for every element.

#### Reference
* A. Griewank and P. Toint, *On the unconstrained optimization of partially-separable functions*, Numerische Nonlinear Optimization 1981, 39, pp. 301--312, 1982.

## Example: the partitioned structure of a quadratic
Let's take the quadratic `f` as an example
```@example PartitionedStructuresQuadratic
f(x) = x[1]^2 + x[1]*x[2] + x[2]^2 + x[3]^2 + 3x[2]*x[3]
```
`f` can be considered as the sum of two element functions
```@example PartitionedStructuresQuadratic
f1(x) = x[1]^2 + x[1]*x[2]
f2(y) = y[1]^2 + y[2]^2 + 3y[1]*y[2]
```
Define $U_1$ and $U_2$ indicating which variables are required by each element function:
```@example PartitionedStructuresQuadratic
U1 = [1 0 0; 0 1 0]
U2 = [0 1 0; 0 0 1]
```
However, dense matrix produce memory issues for large problems.
Instead, we use a linear operator only informing the indices of the variables
```@example PartitionedStructuresQuadratic
U1 = [1, 2]
U2 = [2, 3]
```

By gathering the different $U_i$ together
```@example PartitionedStructuresQuadratic
U = [U1, U2]
```
we define the function exploiting the partially-separable structure `f_pss` as
```@example PartitionedStructuresQuadratic
f_pss(x, U) = f1(x[U[1]]) + f2(x[U[2]])

using Test
x0 = [2., 3., 4.]
@test f(x0) == f_pss(x0, U)
```

Similarly, you can compute: the gradient, the element gradients and explicit how the gradient is partitioned
```@example PartitionedStructuresQuadratic
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
We would like to store every element gradient to build afterward the element gradients difference   required for the partitioned quasi-Newton update.
Thus, we define a partitioned vector from `U`, to store each element gradient and form the $\nabla f$ when required
```@example PartitionedStructuresQuadratic
using PartitionedStructures
U = [U1, U2]
n = length(x0)
# create the partitioned vector
partitioned_gradient_x0 = create_epv(U)
```
We set the value of each element vector to the corresponding element gradient
```@example PartitionedStructuresQuadratic
# return every element gradient
vector_gradient_element(x, U) = [∇f1(x[U[1]]), ∇f2(x[U[2]])] :: Vector{Vector{Float64}}
# set each element vector to its corresponding element gradient
set_epv!(partitioned_gradient_x0, vector_gradient_element(x0, U)) 

# Build the gradient vector
build_v!(partitioned_gradient_x0)

# with the same value as the gradient
@test get_v(partitioned_gradient_x0) == ∇f(x0)
```
## Approximate the Hessian $\nabla^2 f$
There is at least two ways to approximate $\nabla^2 f$: a classical quasi-Newton approximation (ex: BFGS) and a partitioned quasi-Newton approximation (ex: PBFGS).
Both methods are presented, and we expose the sparse structure of partitioned approximation.
### Quasi-Newton approximation of the quadratic (BFGS)
In the case of the BFGS method, you want to approximate the Hessian matrix from `s = x1 - x0`
```@example PartitionedStructuresQuadratic
x1 = [1., 2., 2.]
s = x1 .- x0
```
the gradient difference `y`
```@example PartitionedStructuresQuadratic
y = (∇f(x1) .- ∇f(x0))
```
and the approximation `B`, initially set to the identity
```@example PartitionedStructuresQuadratic
B = [ i==j ? 1. : 0. for i in 1:n, j in 1:n]
```

By applying the BFGS update, you satisfy the secant equation `Bs = y`
```@example PartitionedStructuresQuadratic
# PartitionedStructures.jl export a BFGS implementation
B_BFGS = BFGS(s,y,B) 

using LinearAlgebra
# numerical verification of the secant equation
@test norm(B_BFGS * s - y) < atol 
```
but the approximation `B_BFGS` is dense.
```@example PartitionedStructuresQuadratic
B_BFGS
```

### Partitioned quasi-Newton approximation of the quadratic function (PBFGS)
In order to make a sparse quasi-Newton approximation of $\nabla^2 f$, you may define a partitioned matrix with the same partially-separable structure than `partitioned_gradient_x0` where each element matrix is set to the identity
```@example PartitionedStructuresQuadratic
partitioned_matrix = epm_from_epv(partitioned_gradient_x0)
```
It can be visualized with the `Matrix` constructor
```@example PartitionedStructuresQuadratic
Matrix(partitioned_matrix)
```

The second term of the diagonal accumulates two components of value 1.0 from the two initial element approximations.

Then you compute the partitioned gradient at `x1`
```@example PartitionedStructuresQuadratic
partitioned_gradient_x1 = create_epv(U)
set_epv!(partitioned_gradient_x1, vector_gradient_element(x1, U))
```
and compute the elementwise difference of the partitioned gradients `partitioned_gradient_x1 - partitioned_gradient_x0`
```@example PartitionedStructuresQuadratic
# copy to avoid side effects on partitioned_gradient_x0
partitioned_gradient_difference = copy(partitioned_gradient_x0)

# apply in place an unary minus to every element gradient
minus_epv!(partitioned_gradient_difference)

# add in place the element vector of partitioned_gradient_x1 
# to the corresponding element vector of partitioned_gradient_difference
add_epv!(partitioned_gradient_x1, partitioned_gradient_difference)

# compute the vector y
build_v!(partitioned_gradient_difference) 
@test get_v(partitioned_gradient_difference) == y
```
Then you can define the partitioned quasi-Newton update PBFGS
```@example PartitionedStructuresQuadratic
# apply the partitioned update PBFGS to partitioned_matrix 
# and return Matrix(partitioned_matrix)
B_PBFGS = update(partitioned_matrix, partitioned_gradient_difference, s; name=:pbfgs, verbose=true)
```

which keeps the sparsity structure of ∇²f.

In addition, `update()` informs the number of element: updated, not updated or untouched, as long as the user don't set `verbose=false`.
The partitioned update verifies the secant equation
```@example PartitionedStructuresQuadratic
atol = sqrt(eps(eltype(s1)))
@test norm(B_PBFGS*s - y) < atol
```
which may also be calculated with
```@example PartitionedStructuresQuadratic
# compute the product partitioned matrix vector
Bs = mul_epm_vector(partitioned_matrix, s)
@test norm(Bs - y) < atol
```

## Others partitioned quasi-Newton approximations
There exist two categories of partitioned quasi-Newton updates.
In the first category, each element Hessian $\nabla^2 f_i$ is approximated with a dense matrix, for example: PBFGS.
In the second category, each element Hessian $\nabla^2 f_i$ is approximated with a quasi-Newton linear-operator.

### Partitioned quasi-Newton operators
Once the partitioned matrix is allocated,
```@example PartitionedStructuresQuadratic
partitioned_matrix_PBFGS = epm_from_epv(partitioned_gradient_x0)
partitioned_matrix_PSR1 = epm_from_epv(partitioned_gradient_x0)
partitioned_matrix_PSE = epm_from_epv(partitioned_gradient_x0)
```
you can apply on it any of the three partitioned updates : PBFGS, PSR1, PSE (by default) :
- PBFGS updates each element approximation with BFGS;
- PSR1 updates each element approximation with SR1;
- In PSE, each element approximation is updated with BFGS if it is possible or with SR1 otherwise.
```@example PartitionedStructuresQuadratic
B_PBFGS = update(partitioned_matrix_PBFGS, partitioned_gradient_difference, s; name=:pbfgs)
B_PSR1 = update(partitioned_matrix_PSR1, partitioned_gradient_difference, s; name=:psr1)
B_PSE = update(partitioned_matrix_PSE, partitioned_gradient_difference, s) # ; name=:pse by default
```
All these methods satisfy the secant equation as long as every element approximation is updated
```@example PartitionedStructuresQuadratic
@test norm(mul_epm_vector(partitioned_matrix_PBFGS, s) - y) < atol
@test norm(mul_epm_vector(partitioned_matrix_PSR1, s) - y) < atol
@test norm(mul_epm_vector(partitioned_matrix_PSE, s) - y) < atol
```

### Limited-memory partitioned quasi-Newton operators
These operators are made to apply the partitioned quasi-Newton methods to the partially-separable function with large elements, whose element Hessian approximations can't be store by dense matrices.
The limited-memory partitioned quasi-Newton operators allocate for each element Hessian approximation a quasi-Newton operator LBFGS or LSR1 defined in [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).
It defines three approximations:
- PLBFGS, each element approximation is a `LBFGSOperator`;
- PLSR1, each element approximation is a `LSR1Operator`;
- PLSE, each element approximation may be a `LBFGSOperator` or `LSR1Operator`.

Contrary to the partitioned quasi-Newton operators, each partitioned limited-memory quasi-Newton operators has a different type
```@example PartitionedStructuresQuadratic
partitioned_linear_operator_PLBFGS = eplo_lbfgs_from_epv(partitioned_gradient_x0)
partitioned_linear_operator_PLSR1 = eplo_lsr1_from_epv(partitioned_gradient_x0)
partitioned_linear_operator_PLSE = eplo_lose_from_epv(partitioned_gradient_x0)
```
The different types simplify the `update` method, since no argument `name` is required to determine which update is applied
```@example PartitionedStructuresQuadratic
B_PLBFGS = update(partitioned_linear_operator_PLBFGS, partitioned_gradient_difference, s)
B_PLSE = update(partitioned_linear_operator_PLSE, partitioned_gradient_difference, s)
B_PLSR1 = update(partitioned_linear_operator_PLSR1, partitioned_gradient_difference, s)

@test norm(B_PLBFGS * s - y) < atol
@test norm(B_PLSE * s - y) < atol
@test norm(B_PLSR1 * s - y) < atol
```

That's it, you have all the tools to implement a partitioned quasi-Newton method, enjoy!

### Tips
The main issue about the definition of partitioned structures is informing the $f_i$ and $U_i$.
To address this issue you may want to take a look at [ExpressionTreeForge.jl](https://github.com/paraynaud/CalculusTreeTools.jl) which detect automatically the partially separable structure from an [ADNLPModels](https://github.com/JuliaSmoothOptimizers/ADNLPModels.jl) or a [JuMP model](https://github.com/jump-dev/JuMP.jl).