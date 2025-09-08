
#10/5/2024
#preformance evaluation to be run on cluster to see if we are getting a speed up when using F32

using LinearAlgebra
using MultiFloats
MultiFloats.use_bigfloat_transcendentals()
#using Plots
#using BenchmarkTools

flush(stdout)

function _complex_root_of_unity(::Type{T}, N) where {T}
    # first try the "direct" route (works for Float64, Float32, and MultiFloat if exp is available)
    try
        θ = T(-2π) / T(N)
        return exp(im * θ) |> Complex{T}
    catch err
        # fallback: compute in BigFloat and cast
        θ_big = BigFloat(-2π) / BigFloat(N)
        ω_big = exp(im * θ_big)                       # Complex{BigFloat}
        # cast real and imag parts back to T safely
        reT = T(real(ω_big))
        imT = T(imag(ω_big))
        return Complex{T}(reT, imT)
    end
end

"""
    dft_matrix(N, T)

Return the N×N DFT matrix of type Complex{T}, normalized by sqrt(N).
This is a naive O(N^2) construction but works for MultiFloat types.
"""
function dft_matrix(N::Int, ::Type{T}) where {T}
    ω = _complex_root_of_unity(T, N)   # Complex{T}
    F = Array{Complex{T}}(undef, N, N)
    for m in 1:N, n in 1:N
        # exponent is integer -> safe
        F[m, n] = ω^((m - 1) * (n - 1))
    end
    F ./= sqrt(T(N))
    return F
end

"""
    idft_matrix(N, T)

Return the inverse DFT matrix (i.e., inv(F)) in Complex{T}. Uses linear solve
to remain numerically stable.
"""
function idft_matrix(N::Int, ::Type{T}) where {T}
    F = dft_matrix(N, T)
    # inv(F) computed via factorization/solve; returns Complex{T} matrix
    # here we compute it explicitly once (O(N^3)) which is fine for small N
    return inv(F)
end

"""
    fourier_diff_matrix_naive(N; L=2π, T=Float64)

Return the N×N real differentiation matrix (first derivative) using spectral
(DFT) approach. Works with `T` equal to standard floats or MultiFloat types.
"""
function fourier_diff_matrix_naive(N::Int; L::Real = 2π, T = Float64)
    @assert iseven(N) "N must be even."

    # wavenumbers (integer)
    k = vcat(0:N ÷ 2, -N ÷ 2 + 1:-1)
    # make k in type T
    kT = T(2π) / T(L) .* T.(k)
    D1 = Complex{T}.(im * T(1) .* kT)   # i*k in Complex{T}

    # Build DFT and its inverse in Complex{T}
    F = dft_matrix(N, T)               # Complex{T} matrix (normalized)
    Finv = inv(F)                       # small N - okay; type-stable Complex{T}

    # spectral differentiation: D = real(F_inv * Diagonal(D1) * F)
    # use linear solve to avoid extra inversions? we already have Finv here
    Dmat = Finv * Diagonal(D1) * F
    Dreal = real(Dmat)                  # convert to real matrix of T
    return Matrix{T}(Dreal)
end


# L2 Error
norm_L2(u, dx) = sqrt(sum(abs2, u) * dx)

function newtons_method(f, x0, Dx, dt, tol=1e-10, maxiter=1000)
    x = x0
    Id = Matrix{typeof(x0[1])}(I, length(x0), length(x0))
    for i in 1:maxiter
        fx = f(x)
        if norm(fx) < tol
            return x
        end
        J = (dt / 2) * -Dx * Diagonal(x) - Id
        x = x - J \ fx
    end
    error("Newton's method did not converge")
end


function implicit_midpoint_step(u::Vector{H}, D::Matrix{H}, Dlow::Matrix{L}, dt::H, ::Type{H}, ::Type{L}) where {H, L}
    ulow = L.(u)
    f = y -> -y + ulow + L(dt)/2 * (L(-0.5) * Dlow * (y.^2))
    tol = 10 * eps(L) 
    y1 = newtons_method(f, ulow, Dlow, L(dt), tol) #as long as this is computed in low precision it should be fine
    y1 = H.(y1)
    return u + dt*(-0.5*D*(y1.^2))
end

function rk4_step(u, D, dt)
    k1 = -0.5*D*(u.^2)
    k2 = -0.5*D*(u + 0.5 * dt * k1).^2
    k3 = -0.5*D*(u + 0.5 * dt * k2).^2
    k4 = -0.5*D*(u + dt * k3).^2
    return u + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
end




using Dates  # for timing


function run_mixed_precision_burgers(low::Type, high::Type, L, x,dx, u0,uref; N=20, T_Final=0.8)
    # Use naive diff for high, FFTW for low as long as low is fp64 or less
    Dx_high = fourier_diff_matrix_naive(N; L=L, T=high)
    Dx_low  = fourier_diff_matrix_naive(N; L=L, T=low)

    #cast everything to high precision
    u0 = high.(u0)
    x = high.(x)
    dx = high(dx)

    # Run mixed-precision IMR
    dt = 0.001
    nt = Int(round(T_Final / dt))


    #we can "warm up" the impilict_midpoint_step so it compiles once before and the timeing wont then count the compile time
    throw_away_u = implicit_midpoint_step(high.(u0),Dx_high,Dx_low,high(dt),high,low)
    result = @timed begin
        temp_u = copy(u0)
        temp_u = high.(temp_u)
        for _ in 1:nt
            temp_u = implicit_midpoint_step(high.(temp_u), Dx_high, Dx_low, high(dt), high, low)
        end
        temp_u
    end

    runtime = result.time  # Convert nanoseconds to seconds
    u = result.value  # Final value after all steps

    uref = high.(uref)

    error_L2 = norm_L2(uref - u, dx)

    return (
        error = error_L2,
        time = runtime,
        u = u,
        uref = uref,
        x = x,
    )

end

errors = Float64[]
times = Float64[]
labels = String[]

precision_pairs = [
    (Float32,Float32),
    (Float32,Float32),
    (Float32, Float64),
    (Float32,Float64),
    (Float32, Float64x2),
    (Float64, Float64x2),
    (Float64x2, Float64x4),
]

# Initial condition and reference solution in high precision
T = Float64x4
N = 32
L = T.(2π)
x = L * (0:N-1) / N
u0 = T.(sin.(BigFloat.(x)))
dx = L / N
dt_ref = 0.0001
T_Final = 0.8
nt_ref = Int(round(T_Final/dt_ref))
Dx_ref = fourier_diff_matrix_naive(N; L=L, T=T)
uref = copy(u0)
function compute_reference(u0, Dx_ref, dt_ref, nt_ref)
    uref = copy(u0)
    for _ in 1:nt_ref
        uref = rk4_step(uref, Dx_ref, dt_ref)
    end
    return uref
end

uref = compute_reference(u0, Dx_ref, dt_ref, nt_ref)




println("Running mixed precision Burgers' equation solver...")
for (low, high) in precision_pairs
    println("\nTesting low = $(low), high = $(high)")
    result = run_mixed_precision_burgers(low, high, L, x, dx, u0, uref; N=N, T_Final=0.8)
    println("L2 error = ", result.error)
    println("Runtime = ", result.time, " seconds")

end



