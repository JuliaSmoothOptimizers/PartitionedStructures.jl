# PartitionedStructures.jl Tutorial

The optimization of the partially separable function 
$$
 f(x) = \sum_{i=1}^N \hat{f}_i (U_i) x,
$$
where $\hat{f}_i : \R^{n_i} \to \R, \; n_i < n$, leads to manipulate the partitioned derivatives
$$ 
\nabla f(x) = \sum_{i=1}^N U_i^\top \hat{f}_i (U_i x), \quad \nabla^2 f(x) = \sum_{i=1}^N U_i^\top \hat{f}_i (U_i x) U_i.
$$
This partial separability is parametrized by $U_i$.

For now, PartitionedStructures supports only the elemental $U_i$, i.e. the lines of $U_i$ are vectors from the euclidean basis.

PartitionedStructures.jl define :$.
- the partitioned vectors and partitioned matrices affiliate respectively to $\nabla f(x)$ and $\nabla^2 f(x);
- some