## Elemental and internal variables

There are several types of partial separability.
We write a partially-separable function $f: \R^n \to \R$ in the form
```math
  f(x) = \sum_{i=1}^N f_i (U_i(x)),\; f_i : \R^{n_i} \to \R, \; U_i \in \R^{n_i \times n},\; n_i \ll n
```
where:
* $f_i$ is the $i$-th element function whose dimension is smaller than $f$;
* $U_i$ is the linear operator selecting the linear combinations of variables that parametrize $f_i$.

Uᵢ may be based on the *elemental* variables or the *internal* variables of $f_i$:
- the elemental variables represent the subset of variables that parametrizes $f_i$, i.e. the rows of $U_i$ are vectors from the Euclidean basis;
- the internal variables are linear combinations of the variables that parametrize $f_i$, i.e. $U_i$ may be a dense matrix.

The implementation of the linear-operator $U_i$, which describe entirely the partially-separable structure of $f$, changes depending on wether we use internal or elemental variables.
At the moment, we only developed the elemental partitioned structures, but we left the door open to the development of internal partitioned structures in the future.

## Abbreviations in the code
If you take a look at the code, you will see the following acronyms

Acronyms  | Description
----------|------------
`eev`     | elemental element vector
`epv`     | elemental partitioned vector
`eem`     | elemental element matrix
`eelo`    | elemental partitioned limited-memory operator
`epm`     | elemental partitioned matrix
`eplo`    | elemental partitioned limited-memory operator
`ees`     | elemental element structure
`eps`     | elemental partitioned structure