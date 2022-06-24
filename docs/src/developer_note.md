## Elemental and internal variables

There is several types of partial separability 
```math
  f(x) = \sum_{i=1}^N f_i U_i(x) : \R^n \to \R,\; f_i : \R^{n_i} \to \R, \; U_i \in \R^{n_i \times n},\; n_i < n
```
Uᵢ may be based from the *elemental* variables or the *internal* variables of fᵢ:
- the elemental variables represent the subset of variables that parametrizes fᵢ, i.e. the lines of Uᵢ are vectors from the euclidean basis;
- the internal variables are the linear combination of the variables that parametrizes fᵢ, i.e. Uᵢ may be a dense matrix.

In consequence, the implementation of the linear operator Uᵢ, which support entirely the partial separability, change depending on internal or elemental variables.
At the moment, we mainly developed the elemental partitioned structures, but we left the door open to the development of internal partitioned structures in the future.


## Abbreviations in the code
If you take a look at the code, you will see some

Acronyms  | Description
----------|------------
`eev`     | elemental element vector
`iev`     | internal element vector
`epv`     | elemental partitioned vector
`ipv`     | internal partitioned vector
`eem`     | elemental element matrix
`eelom`   | limited-memory elemental element matrix
`iem`     | internal element matrix
`epm`     | elemental partitioned matrix
`ipm`     | internal partitioned matrix
`eplom`   | limited-memory elemental partitioned matrix