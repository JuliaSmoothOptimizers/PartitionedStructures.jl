# PartitionedStructures.jl: Tutorial

## Reminder of the structure partially separable
The quasi-Newton methods exploiting the partially separable function 
$$
 f(x) = \sum_{i=1}^N \hat{f}_i (U_i) : \R^n \to \R,,
$$
where $\hat{f}_i : \R^{n_i} \to \R, \; U_i \in \R^{n_i \times n},\; n_i < n$, leads to manipulate the partitioned derivatives
$$ 
\nabla f(x) = \sum_{i=1}^N U_i^\top \hat{f}_i (U_i x), \quad \nabla^2 f(x) = \sum_{i=1}^N U_i^\top \hat{f}_i (U_i x) U_i,
$$
which accumulate the element derivatives $\hat{f}_i$ and $\nabla^2 \hat{f}_i$.

These partitioned quasi-Newton methods define a partitioned quasi-Newton approximations of the Hessian $B \approx \nabla^2 f$.
$B$ approximate every element Hessian $\hat{B}_i \approx \nabla^2 \hat{f}_i$ and accumulate them with respect to $U_i$.
The partitioned quasi-Newton approximation structurally keeps the sparsity structure of $\nabla^2 f$, which is not the case of classical quasi-Newton approximation.
Moreover, the rank of a such an update may be proportional to the number of elements `N`, whereas classical quasi-Newton approximation are low rank updates. 


#### Reference
* A. Griewank and P. Toint, *On the unconstrained optimization of partially separable functions*, Numerische Nonlinear Optimization 1981, 39, pp. 301--312, 1982.


## The partitioned structure of a quadratic

Let's take an example of `f` a quadratic function
```julia
f(x) = x[1]^2 + x[2]^2 + x[3]^2 + x[1]*x[2] + 3x[2]*x[3]
```
`f` can be considered as the sum of two element function
```julia
f1(x) = x[1]^2 + x[1]*x[2]
f2(x) = x[1]^2 + x[2]^2 + 3x[1]*x[2]
```
applied considering
```julia
U1 = [1, 2]
U2 = [2, 3]
```

By gathering the different $U_i$ together
```julia
U = [U1, U2]
```
we define the function `f` by exploiting explicitly the partially separable structure
```julia
f_ps(x, U) = f1(x[U[1]]) + f2(x[U[2]])

using Test
x0 = [2., 3., 4.]
@test f(x0) == f_ps(x0)
```

In a similar way, you can compute: the gradient, every element gradients and explicit how the gradient is partitioned
```julia
∇f(x) = [2x[1] + x[2], x[1] + 2x[2] + 3x[3], 2x[3] + 3x[2]]
∇f1(x) = [2x[1] + x[2], x[1]]
∇f2(x) = [2x[1] + 3x[2], 2x[2] + 3x[1]]
function ∇f_ps(x, U)
  gradient = zeros(length(x))
  gradient[U1] = ∇f1(x[U[1]])
  gradient[U2] .+= ∇f2(x[U[2]])
  return gradient
end
@test ∇f(x0) == ∇f_ps(x0, U)
```

However, `∇f_ps` accumulates directly the element gradient and does not store the value of each element gradients `∇f1, ∇f2`.
We define the partitioned vector, from `U` and `n`, to store each element gradient and form the gradient when required
```julia
using PartitionedStructures
U = [U1, U2]
n = length(x0)
partitioned_gradient = create_epv(U, n)
```
We set the value of each element vector to the corresponding element gradient
```julia
vector_gradient_element(x, U) = [∇f1(x[U[1]]), ∇f2(x[U[2]])] :: Vector{Vector{Float64}}
 # returns each element gradient
set_epv!(partitioned_gradient, vector_gradient_element(x0, U)) # sets the partitioned vector to the partitioned gradient

build_v!(partitioned_gradient) # builds the gradient vector
@test get_v(partitioned_gradient) == ∇f(x0) # with the same value as the gradient
```

## Quasi-Newton approximation of the quadratic
In the case of the BFGS method, you want to approximate the Hessian matrix from `s = x1 - x0`
```julia
x1 = [1., 2., 3.]
s = x1 .- x0
```
the gradient difference `y`
```julia
y = (∇f(x1) .- ∇f(x0))
```
and considering a matrix `B`, initially set to the identity
```julia
B = [ i==j ? 1. : 0. for i in 1:n, j in 1:n]
```

By applying the BFGS update, you satisfy the secant equation `Bs = y`
```julia
B_BFGS = BFGS(s,y,B) # PartitionedStructures.jl implements BFGS

using LinearAlgebra		
@test norm(B_BFGS1 * s1 - y1) == 0. # numerical verification of the secant equation 
```
but the approximation `B_BFGS` is dense.
```julia
julia> B_BFGS
3×3 Matrix{Float64}:
 1.30952   0.952381  0.738095
 0.952381  3.2381    1.80952
 0.738095  1.80952   2.45238
```

In order to make a sparse quasi-Newton approximation of $\nabla^2 f$, define a partitioned matrix with the same partially separable structure than the partitioned_gradient
```julia
partitioned_matrix = epm_from_epv(partitioned_gradient)
```
where each element matrix is set to the identity
```julia
julia> Matrix(partitioned_matrix)
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  2.0  0.0
 0.0  0.0  1.0
```
Then you compute the partitioned gradient at `x1`
```julia
partitioned_gradient1 = create_epv(U, n)
set_epv!(partitioned_gradient1, vector_gradient_element(x1, U))
```
and compute the difference of the partitioned gradients `partitioned_gradient_difference = partitioned_gradient1 - partitioned_gradient`
```julia
partitioned_gradient_difference = copy(partitioned_gradient) # copy of partitioned_gradient to avoid side effects
minus_epv!(partitioned_gradient_difference) # apply a unary minus to every element gradient
add_epv!(partitioned_gradient1, partitioned_gradient_difference) # add partitioned_gradient1 and partitioned_gradient_difference

build_v!(partitioned_gradient_difference) # compute the vector y
@test get_v(partitioned_gradient_difference) == y
```
Then you can define the partitioned quasi-Newton update PBFGS
```julia
B_PBFGS = update(partitioned_matrix, partitioned_gradient_difference, s; name=:pbfgs) # apply the partitioned update to partitioned_matrix and returns Matrix(partitioned_matrix)
```
with the same sparsity structure than $\nabla^2 f$
```julia
julia> B_PBFGS
3×3 Matrix{Float64}:
 2.75  0.25  0.0
 0.25  3.75  2.0
 0.0   2.0   3.0
```
and verify the secant equation
```julia
Bs = mul_epm_vector(partitioned_matrix, s) # compute the product partitioned matrix vector
@test norm(Bs-y) == 0.
```

## Other partitioned quasi-Newton approximation




On the other hand, if you take advantage of the partially separable structure, you can define a quasi-Newton




## Features
For now, PartitionedStructures supports only the elemental $U_i$, i.e. the lines of $U_i$ are vectors from the euclidean basis.

exemple build 
écrire la quadratique de f avec des vecteurs partitionés

gradient de f
```julia

```
gradient de f1

gradient de f2
```julia

```


création des vecteurs

```julia
```


```julia
```

Affichage des courbes de niveau avec $x_2$ fixée



PartitionedStructures.jl define :$.
- the partitioned vectors and partitioned matrices affiliate respectively to $\nabla f(x)$ and $\nabla^2 f(x);
- some

