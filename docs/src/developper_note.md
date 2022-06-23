## Important things to know
There is several type of partial separability 
$$  f(x) = \sum_{i=1}^N \widehat{f}_i U_i(x), $$
$U_i$ may be based from the elemental variables or the internal variables of $\widehat{f}_i$.
The elemental variables represent the subset of variables that parametrizes $\hat{f}_i$ while the internal variables are the linear combination of the variables that parametrizes $\hat{f}_i$.
In consequence the implementation of the linear operator $U_i$, which support entirely the partial separability, change depending interal or elemental variables.

If you take a look at the code, you will see some
```julia
eev # refers to an elemental element vector
iev # refers to an internal element vector
epv # refers to an elemental partitioned vector
ipv # refers to an internal partitioned vector
eem # refers to an elemental element matrix
eelom # refers to limited-memory elemental element matrix
iem # refers to an internal element matrix
epm # refers to an elemental partitioned matrix
ipm # refers to an internal partitioned matrix
eplom # refers to limited-memory elemental partitioned matrix
```
At the moment, we mainly developed the elemental partitioned structures, but we left the door open to the development of internal partitioned structures in the future.
