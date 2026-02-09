# TUR violations in the three-level maser in a driven-dissipative cavity

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> Quantum_TLM_TUR

## Publication

This project is based on the following publication:

[Quantum TUR violations in the three-level maser](https://arxiv.org/abs/2602.06744)

## Dependencies

This project uses the [QuantumFCS.jl](https://github.com/marcelojbp/QuantumFCS.jl) package as a backend for full-counting statistics calculations.


To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "Quantum_TLM_TUR"
```
which auto-activate the project and enable local path handling from DrWatson.
