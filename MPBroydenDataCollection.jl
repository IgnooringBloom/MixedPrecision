using Statistics
using LinearAlgebra
using FFTW
using Dates
using TimerOutputs
using DelimitedFiles # For CSV output
using Printf        # For formatted printing
using XLSX          # For writing to Excel files
using DataFrames


# -----------------------------
# Precision setup
# -----------------------------
const FullFloat = eval(Meta.parse(get(ARGS, 1, "Float64")))
const ReduFloat = eval(Meta.parse(get(ARGS, 2, "Float32")))

# -----------------------------
# Build spatial grid and derivative matrix
# -----------------------------
function setup_problem(Nx::Int, L::FullFloat, FullFloatType::DataType, ReduFloatType::DataType)
    x = LinRange(FullFloat(0), L, Nx + 1)[1:end-1]
    dx = x[2] - x[1]

    # Define k_vals in un-shifted order
    k_vals = collect(FullFloatType, vcat(0:((Nx-1) ÷ 2), -((Nx-1) ÷ 2):-1))

    # Full precision derivative matrix
    Iden_high = Matrix{FullFloatType}(I, Nx, Nx)
    K_high = Diagonal(k_vals)
    Dx_high = real(ifft((im * one(FullFloatType)) * K_high * fft(Iden_high, 1), 1))

    # Low precision derivative matrix
    Iden_low = Matrix{ReduFloatType}(I, Nx, Nx)
    K_low = Diagonal(ReduFloatType.(k_vals))
    Dx_low = real(ifft((im * one(ReduFloatType)) * K_low * fft(Iden_low, 1), 1))
    
    return x, dx, Dx_high, Dx_low
end

# -----------------------------
# Residual function for nonlinear system (for midpoint solve)
# -----------------------------
function residual(u::Vector{FullFloat}, un::Vector{FullFloat}, dt::FullFloat, Dx::Matrix{FullFloat})
    return u .- un .+ FullFloat(0.5) .* dt .* Dx * (FullFloat(0.5) .* (u .^ 2))
end

# -----------------------------
# Implicit midpoint step (mixed precision)
# -----------------------------
function implicit_midpoint_step_newton(u::Vector{FullFloat}, Dx_high::Matrix{FullFloat}, 
                     Dx_low::Matrix{ReduFloat}, dt::FullFloat, Tol::FullFloat)
    ulow = ReduFloat.(u)
    dt_low = ReduFloat(dt)
    tol_low = ReduFloat(Tol) # Cast tolerance to low precision
    
    f_low = y -> y .- ulow .- (dt_low / ReduFloat(2)) .* (ReduFloat(-0.5) .* Dx_low * (y .^ ReduFloat(2)))
    
    y_low, iters = newton_solve(f_low, ulow, Dx_low, dt_low, tol=tol_low)
    y_high = FullFloat.(y_low)
    
    u_new = u .+ dt .* (FullFloat(-0.5) .* Dx_high * (y_high.^FullFloat(2)))
    return u_new, iters
end

# -----------------------------
# Newton solver for one step (low precision)
# -----------------------------
function newton_solve(f::Function, x0::Vector{ReduFloat}, Dx_low::Matrix{ReduFloat}, dt_low::ReduFloat;
                      tol::ReduFloat = 10*eps(ReduFloat), maxiter::Int = 100)
    x = copy(x0)
    Id = Matrix{ReduFloat}(I, length(x0), length(x0))
    h2 = dt_low / ReduFloat(2)
    for iter in 1:maxiter
        fx = f(x)
        if norm(fx, Inf) < tol
            return x, iter # Return converged solution and iteration count
        end
        J = Id .+ (h2) .* (Dx_low * Diagonal(x))
        x .= x .- J \ fx # Corrected update step
    end
    # Return the non-converged solution and max iterations
    return x, maxiter
end

# -----------------------------
# Time integration using Newton's method (Mixed Precision)
# -----------------------------
function run_newton_mp(x::Vector{FullFloat}, Dx_high::Matrix{FullFloat}, Dx_low::Matrix{ReduFloat}; 
                       Nt::Int, dt::FullFloat, Tol::FullFloat)
    u = sin.(x)
    iaCount = zeros(Int, Nt)

    for i in 1:Nt
        u, its = implicit_midpoint_step_newton(u, Dx_high, Dx_low, dt, Tol)
        iaCount[i] = its
        if any(isnan, u)
            println("Warning: NaN detected in Mixed-Precision Newton run.")
            break   
        end
    end
    return (u = u, Iter = sum(iaCount))
end

# -----------------------------
# Broyden solver for semi-implicit midpoint rule
# -----------------------------
function run_broyden_midpoint(x::Vector{FullFloat}, Dx_high::Matrix{FullFloat}; 
                              Nx::Int, Nt::Int, dt::FullFloat, ResTol::FullFloat, MP=true)
    un = sin.(x)
    iaCount = zeros(Int, Nt)
    Iden = Matrix{FullFloat}(I, Nx, Nx)
    J = Iden .+ FullFloat(0.5) * dt .* Dx_high * Diagonal(un)

    Ji = MP ? Matrix{FullFloat}(inv(ReduFloat.(J))) : inv(J)
    good = true

    for i in 1:Nt

        # print("Nt = ",i)
        u_mid, its, Ji = broyden_solve(un, dt, Dx_high, Ji, ResTol; good=good)
        iaCount[i] = its
        f1 = -Dx_high * (FullFloat(0.5) .* (u_mid .^ FullFloat(2)))
        un = un .+ dt .* f1

        if any(isnan, un)
            println("Warning: NaN detected in Broyden run (MP=$MP).")
            break
        end
    end

    return (u = un, Iter = sum(iaCount))
end

function run_broyden_midpoint_one_newton(x::Vector{FullFloat}, Dx_high::Matrix{FullFloat}; 
                              Nx::Int, Nt::Int, dt::FullFloat, ResTol::FullFloat, MP=true)
    un = sin.(x)
    iaCount = zeros(Int, Nt)
    Iden = Matrix{FullFloat}(I, Nx, Nx)
    J = Iden .+ FullFloat(0.5) * dt .* Dx_high * Diagonal(un)

    Ji = MP ? Matrix{FullFloat}(inv(ReduFloat.(J))) : inv(J)
    good = true

    for i in 1:Nt
        
        u = copy(un)
        fun = residual(u,un,dt,Dx_high)
        s = -Ji * fun
        u .= u .+ s
        un = copy(u)
        fun = residual(u,un,dt,Dx_high)

        infnorm = norm(fun, Inf)
        print("residual_1n = ",infnorm," tol = ", ResTol)

        if  infnorm > ResTol
            print(" Nt = ",i)
            u_mid, its, Ji = broyden_solve(un, dt, Dx_high, Ji, ResTol; good=good)
        end

        iaCount[i] = its
        f1 = -Dx_high * (FullFloat(0.5) .* (u_mid .^ FullFloat(2)))
        un = un .+ dt .* f1

        if any(isnan, un)
            println("Warning: NaN detected in Broyden run (MP=$MP).")
            break
        end
    end

    return (u = un, Iter = sum(iaCount))
end

# -----------------------------
# Broyden solver for one time step
# -----------------------------
function broyden_solve(un::Vector{FullFloat}, dt::FullFloat, Dx::Matrix{FullFloat}, 
                       Ji::Matrix{FullFloat}, ResTol::FullFloat; good::Bool=true, maxiter::Int=50)
    u = copy(un)
    fun_curr = residual(u, un, dt, Dx)

    for ia in 1:maxiter
        fun_old = copy(fun_curr)
        s = -Ji * fun_old
        u .= u .+ s

        fun_curr = residual(u, un, dt, Dx)
        infnorm = norm(fun_curr, Inf)

        if infnorm < ResTol
            # println(" N_corr = ",ia," residual = ",infnorm)
            return u, ia, Ji
        end

        y = fun_curr - fun_old
        b = good ? (s' * Ji) : y'
        Ji = Ji + ((s - Ji * y) * b) / (b * y)
    end

    return u, maxiter, Ji
end


# -----------------------------
# NEWTON SOLVER for IMR (Full Precision)
# -----------------------------
function run_newton_full_precision(x::Vector{FullFloat}, Dx_high::Matrix{FullFloat};
                                   Nx::Int, Nt::Int, dt::FullFloat,
                                   maxiter::Int=50, ResTol::FullFloat)
    
    un = sin.(x)
    iaCount = zeros(Int, Nt)
    Iden = Matrix{FullFloat}(I, Nx, Nx)

    for i in 1:Nt
        u = copy(un)
        loopconvergence = false
        for ia in 1:maxiter
            res = u .- un .+ (dt / 4) .* (Dx_high * (u .^ 2))
            if norm(res, Inf) < ResTol
                iaCount[i] = ia
                loopconvergence = true
                break
            end
            J = Iden .+ (dt / 2) .* (Dx_high * Diagonal(u))
            s = - (J \ res)
            u .= u .+ s
        end
        
        if !loopconvergence
            iaCount[i] = maxiter
        end
        
        f1 = -Dx_high * (0.5 .* (u .^ 2))
        un .= un .+ dt .* f1
        
        if any(isnan, un)
            println("Warning: NaN detected in full-precision Newton.")
            break
        end
    end
    
    return (u = un, Iter = sum(iaCount))
end

# -----------------------------
# Main Experiment Function
# -----------------------------
function run_experiment()
    # --- Experiment Parameters ---
    Nx_values = [801, 1601]
    Nt_values = [10, 80, 160]
    tFinal = 0.7
    L = 2 * pi
    NUM_RUNS = 4
    Nt_ref = 320
    FullFloat = Float64
    ReduFloat = Float32
    ResTol_ref = 10 * eps(FullFloat)

    # --- DataFrame Initialization ---
    # Added columns for the reference Newton solver results.
    results_df = DataFrame(
        Nx = Int[],
        Nt = Int[],
        dx = Float64[],
        dt = Float64[],
        CFL = Float64[],
        Time_Newton_Ref_s = Float64[],
        Avg_Iter_Newton_Ref = Float64[],
        Time_Newton_MP_s = Float64[],
        Iter_Newton_MP = Float64[],
        Error_Newton_MP = Float64[],
        Time_Broyden_FP_s = Float64[],
        Iter_Broyden_FP = Float64[],
        Error_Broyden_FP = Float64[],
        Speedup_Broyden_FP = Float64[],
        Time_Broyden_MP_s = Float64[],
        Iter_Broyden_MP = Float64[],
        Error_Broyden_MP = Float64[],
        Speedup_Broyden_MP = Float64[]
    )

    println("--- Starting Benchmark ---")
    println("Averaging timings over $NUM_RUNS runs for each solver.")
    println("High precision (FullFloat): ", FullFloat)
    println("Low precision (ReduFloat): ", ReduFloat)
    println("Reference Nt for error calculation: $Nt_ref")
    println("-"^60)

    for Nx in Nx_values
        println("Setting up for Nx = $Nx...")
        x_linrange, dx, Dx_high, Dx_low = setup_problem(Nx, L, FullFloat, ReduFloat)
        x = collect(x_linrange)

        println(" Generating high-accuracy reference solution...")
        dt_ref = tFinal / Nt_ref
        
        # Time the reference solution generation and capture its results
        ref_time = 0.0
        ref_result = nothing
        ref_time = @elapsed begin
            ref_result = run_newton_full_precision(x, Dx_high; Nx=Nx, Nt=Nt_ref, dt=dt_ref, ResTol=ResTol_ref)
        end
        u_ref = ref_result.u
        ref_avg_iter = ref_result.Iter
        @printf " Reference solution generated in %.4fs with %.1f total iterations.\n" ref_time ref_avg_iter

        for Nt in Nt_values
            dt = tFinal / Nt
            cfl = dt / dx
            Tol = 1e-4
            @printf " Running case: Nx=%d, Nt=%d, CFL=%.4f, Tol=%.2e\n" Nx Nt cfl Tol

            newton_mp_result = broyden_fp_result = broyden_mp_result = nothing

            total_time_newton_mp = fill(0.0, NUM_RUNS)
            for i in 1:NUM_RUNS
                total_time_newton_mp[i] = @elapsed begin
                    newton_mp_result = run_newton_mp(x, Dx_high, Dx_low; Nt=Nt, dt=dt, Tol=Tol)
                end
            end
            newton_mp_time = mean(total_time_newton_mp[2:end])
            error_newton_mp = norm(newton_mp_result.u .- u_ref, Inf)

            total_time_broyden_fp = fill(0.0, NUM_RUNS)
            for i in 1:NUM_RUNS
                total_time_broyden_fp[i] = @elapsed begin
                    broyden_fp_result = run_broyden_midpoint(x, Dx_high; Nx=Nx, Nt=Nt, dt=dt, ResTol=Tol, MP=false)
                end
            end
            broyden_fp_time = mean(total_time_broyden_fp[2:end])
            error_broyden_fp = norm(broyden_fp_result.u .- u_ref, Inf)

            total_time_broyden_mp = fill(0.0, NUM_RUNS)
            for i in 1:NUM_RUNS
                total_time_broyden_mp[i] = @elapsed begin
                    broyden_mp_result = run_broyden_midpoint(x, Dx_high; Nx=Nx, Nt=Nt, dt=dt, ResTol=Tol, MP=true)
                end
            end
            broyden_mp_time = mean(total_time_broyden_mp[2:end])
            error_broyden_mp = norm(broyden_mp_result.u .- u_ref, Inf)

            speedup_broyden_fp = newton_mp_time / broyden_fp_time
            speedup_broyden_mp = newton_mp_time / broyden_mp_time

            # --- Push a new row to the DataFrame with reference solver data ---
            new_row = (
                Nx = Nx, Nt = Nt, dx = dx, dt = dt, CFL = cfl,
                Time_Newton_Ref_s = ref_time,
                Avg_Iter_Newton_Ref = ref_avg_iter,
                Time_Newton_MP_s = newton_mp_time, Iter_Newton_MP = newton_mp_result.Iter, Error_Newton_MP = error_newton_mp,
                Time_Broyden_FP_s = broyden_fp_time, Iter_Broyden_FP = broyden_fp_result.Iter, Error_Broyden_FP = error_broyden_fp, Speedup_Broyden_FP = speedup_broyden_fp,
                Time_Broyden_MP_s = broyden_mp_time, Iter_Broyden_MP = broyden_mp_result.Iter, Error_Broyden_MP = error_broyden_mp, Speedup_Broyden_MP = speedup_broyden_mp
            )
            push!(results_df, new_row)
        end
        println("-"^60)
    end

    println("\n--- Benchmark Results Summary (from DataFrame) ---")

    # --- Generate a unique filename using a timestamp ---
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    # output_filename = "benchmark_results_$(timestamp).xlsx"

    # Make sure this file doesn't already exist in your directory
    output_filename = "IMR_BRG_1e4.xlsx"

    # --- Export the DataFrame ---
    # Now you don't need `overwrite=true` because the filename is always new.
    XLSX.writetable(output_filename, results_df, sheetname="Results")

    println("\nBenchmark finished.")
    println("Successfully exported results to '$(output_filename)'.")

end

function main()

    Nx = 2001
    Nt = 10
    tFinal = 0.7
    L = 2 * pi
    FullFloat = Float64
    ReduFloat = Float32
    

    x_linrange, dx, Dx_high, Dx_low = setup_problem(Nx, L, FullFloat, ReduFloat)
    x = collect(x_linrange)

    dt = tFinal / Nt
    Tol = 1e-4
    
    println("Reusing; Full Precision")
    broyden_fp_result = run_broyden_midpoint_one_newton(x, Dx_high; Nx=Nx, Nt=Nt, dt=dt, ResTol=Tol, MP=false)

    println(broyden_fp_result.Iter)

end

# -----------------------------
# Run the experiment
# -----------------------------
run_experiment()



