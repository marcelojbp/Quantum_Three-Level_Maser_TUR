module QuantumTLMFunctions
using QuantumOptics
using LinearAlgebra
using SparseArrays
using CairoMakie

export nB, nBtoTω,
       liouvQTLM, liouvQTLM_adb, liouv_semiclassical,
       adiabatic_elimination, noise_drazin, var_dotN,
       set_broken_axis!, add_figure_break_indicators!

"""
Module `QuantumTLMFunctions` — extracted helpers from the notebook.

Note: many functions reference objects that are defined in the notebook
(global scope, `Main`), for example `σs`, `a`, `ad`, `Id_cav`, etc. The
functions below access those notebook-defined globals via `Main.<name>` so
you should `include("quantum-TLM_functions.jl"); using .QuantumTLMFunctions`
from the notebook after the notebook has created the required globals.
"""

# -- Utilities ---------------------------------------------------------------

"""
    nB(ω, T)

Bose-Einstein occupation number for frequency `ω` at temperature `T`.
"""
nB(ω, T) = 1/(exp(ω/T) - 1)

"""
    nBtoTω(n)

Inverse relation used to convert occupation `n` to temperature via ω/T = log(1 + 1/n).
Returns the factor ω/T (used as part of a mapping to temperature elsewhere).
"""
nBtoTω(n) = 1/log(1 + 1/n)


# -- Liouvillian builders ---------------------------------------------------

"""
    liouvQTLM(Δ, g, ℰ, Jumps, rates)

Construct Hamiltonian `H` and jump operator vector `J` for the full
quantum three-level + cavity model used in the notebook.
"""
function liouvQTLM(Δ, g, ℰ, Jumps, rates)
    H = g*Main.ad⊗Main.σs[1,2]
    H += QuantumOptics.dagger(H)
    H +=  Δ*Main.Id_cav⊗Main.σs[2,2] + Δ*Main.ad*Main.a⊗Main.Id_tls
    H +=  1im * ℰ * (Main.a⊗one(Main.b_tls) - Main.ad⊗one(Main.b_tls))
    J = [rates[i]*Jumps[i] for i=1:length(rates)]
    return H, J
end


"""
    liouvQTLM_adb(Δ, g, ℰ, rates)

Liouvillian builder for the adiabatically-eliminated model (cavity adiabatically eliminated).
"""
function liouvQTLM_adb(Δ, g, ℰ, rates)
    a_adb = adiabatic_elimination(Δ, g, Main.κ, ℰ)
    ad_adb = QuantumOptics.dagger(a_adb)
    H = g * Main.σs[2,1] *  a_adb
    H += QuantumOptics.dagger(H)
    H +=  Δ * Main.σs[2,2] + Δ * ad_adb * a_adb
    H +=   1im * ℰ * (a_adb - ad_adb)

    J = [Main.σs[1, 3], Main.σs[2, 3], a_adb, Main.σs[3, 1], Main.σs[3, 2], ad_adb]
    J = [rates[i]*J[i] for i=1:length(rates)]
    return H, J
end


"""
    liouv_semiclassical(Δ, ℰ, rates)

Liouvillian builder for the semiclassical three-level model used for comparisons.
"""
function liouv_semiclassical(Δ, ℰ, rates)
    H = Δ*Main.σs[2,2] + (conj(ℰ)*Main.σs[2,1] + ℰ*Main.σs[1,2])
    J = [Main.σs[1, 3], Main.σs[2, 3], Main.σs[3, 1], Main.σs[3, 2]]
    J = [rates[i]*J[i] for i=1:length(rates)]
    return H, J
end




# -- Adiabatic elimination & noise helper -----------------------------------

"""
    adiabatic_elimination(Δ, g, κ, ℰ)

Return the adiabatically eliminated cavity annihilation operator (approximate)
used in the notebook's adiabatic-elimination model.
"""
function adiabatic_elimination(Δ, g, κ, ℰ)
    tc = 1/(1 + 2im*Δ/κ)
    a_adb =  (2 * tc/κ) * (g * Main.σs[1,2] - ℰ * Main.Id_tls)
    return a_adb
end


"""
    noise_drazin(Op, H, J, ρss)

Compute the noise using a Drazin inverse approach via `drazin_apply` from
`QuantumFCS` (the same logic used in the notebook). Returns a real scalar.
"""
function noise_drazin(Op, H, J, ρss)
    ρtilde = Op * ρss
    L = QuantumOptics.liouvillian(H, J).data
    vtilde = sparse(vec(ρtilde.data))
    vss = sparse(vec(ρss.data))

    n = size(ρss, 1)
    l = n*n
    # Vectorized identity as a sparse vector: indices 1:(n+1):l (column-major)
    diag_idx = collect(1:(n+1):l)
    vId = SparseArrays.SparseVector{ComplexF64,Int}(l, diag_idx, fill(1.0 + 0.0im, n))
    vId_dense = Vector{ComplexF64}(vId)
    vD = Main.drazin_apply(L, vtilde, vss, vId_dense; F = LinearAlgebra.lu(L))

    return -2*real(tr(Op.data * reshape(vD, n, n)))
end


# -- Analytics ---------------------------------------------------------------

"""
    var_dotN(gamma_H, gamma_C, n_H, n_C, epsilon, Delta)

Variance per unit time of the stochastic cycle count N(t) in the three-level maser,
given Lindblad parameters:

- gamma_H : coupling rate to the hot bath (γ_H)
- gamma_C : coupling rate to the cold bath (γ_C)
- n_H     : Bose occupation of the hot bath (n_H)
- n_C     : Bose occupation of the cold bath (n_C)
- epsilon : drive amplitude (ε)
- Delta   : detuning (Δ)

Returns: var(Ṅ) = lim_{t→∞} var(N(t))/t (quantum model).
"""
function var_dotN(gamma_H, gamma_C, n_H, n_C, epsilon, Delta)
    # Broadening
    Gamma = 0.5 * (gamma_C * n_C + gamma_H * n_H)

    # Effective coherent transition rate (quantum γ_c)
    gamma_coh = 2 * epsilon^2 * Gamma / (Delta^2 + Gamma^2)

    # Common denominator D
    D = gamma_H * gamma_C * (3n_H * n_C + n_H + n_C) +
        2 * gamma_coh * (3Gamma + gamma_H + gamma_C)

    # Mean cycle current ⟨Ṅ⟩
    Idot = gamma_coh * gamma_H * gamma_C * (n_H - n_C) / D

    # Classical part C_cl
    C_cl = (2gamma_coh + 4Gamma + gamma_H + gamma_C) / D

    # Quantum correction: **note the division by Gamma**
    C_q = (Gamma^2 - Delta^2) / (Delta^2 + Gamma^2) *
          (gamma_H * gamma_C / Gamma) *
          (3n_H * n_C + n_H + n_C) / D

    C = C_cl + C_q

    # Fano factor F = var(Ṅ) / ⟨Ṅ⟩
    F = (n_H * (n_C + 1) + n_C * (n_H + 1)) / (n_H - n_C) - 2 * Idot * C

    # Variance per unit time
    return Idot * F
end



# -- Plot helpers -----------------------------------------------------------

"""
    set_broken_axis!(ax_top, ax_bottom, x, high_lims, low_lims)

Configure two stacked axes to behave like a broken axis (top and bottom ranges).
This keeps styling done in one place.
"""
function set_broken_axis!(ax_top, ax_bottom, x, high_lims, low_lims)
    # Set both x and y limits to match the data exactly
    x_min = minimum(x)
    x_max = maximum(x)
    xlims!(ax_bottom, x_min, x_max)
    xlims!(ax_top, x_min, x_max)
    ylims!(ax_bottom, low_lims...)
    ylims!(ax_top, high_lims...)

    # Style the axes
    hidexdecorations!(ax_top, grid=false)
    ax_top.bottomspinevisible = false
    ax_bottom.topspinevisible = false

    # Link x-axes
    linkxaxes!(ax_top, ax_bottom)
end


"""
    add_figure_break_indicators!(fig, left_pos, right_pos)

Place two small break markers ("//") on the figure scene at positions given in
relative figure coordinates. `left_pos` and `right_pos` are 2-tuples of relative
coordinates, e.g. `(0.11, 0.43)`.
"""
function add_figure_break_indicators!(fig, left_pos, right_pos)
    # Left side break indicator - positioned exactly on left y-axis frame
    text!(fig.scene, Point2f(left_pos), text="//",
          fontsize=16, color=:black, rotation=1.5*π/4,
          align=(:center, :center),
          space=:relative, overdraw=true)

    # Right side break indicator - positioned exactly on right y-axis frame
    text!(fig.scene, Point2f(right_pos), text="//",
          fontsize=16, color=:black, rotation=1.5*π/4,
          align=(:center, :center),
          space=:relative, overdraw=true)

    # # Add a second, slightly offset copy for a double-line effect
    # text!(fig.scene, Point2f(left_pos[1]-0.005, left_pos[2]+0.005), text="//",
    #       fontsize=16, color=:black, rotation=1.5*π/4,
    #       align=(:center, :center),
    #       space=:relative, overdraw=true)
    # text!(fig.scene, Point2f(right_pos[1]+0.005, right_pos[2]+0.005), text="//",
    #       fontsize=16, color=:black, rotation=1.5*π/4,
    #       align=(:center, :center),
    #       space=:relative, overdraw=true)
end


# delta_power(a_avg) = -Ω*κ*(a + ad) + Ω*κ*(a_avg + conj(a_avg))*Id_cav
# delta_power_io(a_avg) =  delta_power(a_avg) - Ω *(κ/2) *(conj(a_avg)*a + a_avg*ad - 2*abs(a_avg)^2*Id_cav)  


# nBtoTω(n) = 1/log(1+1/n)
# nB(ω, T) = 1/(exp(ω/T)-1)

end # module
