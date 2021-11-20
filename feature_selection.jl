IAI

# using Statistics, Random, DataFrames, CSV, JuMP, Gurobi, GLMNet, Plots,  LinearAlgebra, LaTeXStrings, CategoricalArrays
using Gurobi, StatsBase, CSV, DataFrames, JuMP, LinearAlgebra, Distributions, Random, GLMNet


function sparse_regression(X,Y,k,γ,s0=[])
    
    m = Model(Gurobi.Optimizer)
    
    ###
    # Step 1: Define the Variables:
    ###
    @variable(m, s[1:p], Bin)
    @variable(m, t >= 0)
    
    ###
    # Step 2: Set Up Constraints and Objective
    ###
    @constraint(m, sum(s) <= k)
    # Initial solution: if none is provided, start at arbitrary point
    if length(s0) == 0
        s0 = zeros(p)
        s0[1:k] .= 1
    end
    obj0, grad0 = solve_inner_problem(X,Y, s0, γ)
    @constraint(m, t >= obj0 + dot(grad0, s - s0))
    # Objective
    @objective(m, Min, t)
    
    ###
    # Step 3: Define the outer approximation function
    ###
    function outer_approximation(cb_data)
        s_val = []
        for i = 1:p
            s_val = [s_val;callback_value(cb_data, s[i])]
        end
        obj, grad = solve_inner_problem(X,Y, s_val, γ)
        # add the cut: t >= obj + sum(∇s * (s - s_val))
        offset = sum(grad .* s_val)
        con = @build_constraint(t >= obj + sum(grad[j] * s[j] for j=1:p) - offset)    
        MOI.submit(m, MOI.LazyConstraint(cb_data), con)
    end
    MOI.set(m, MOI.LazyConstraintCallback(), outer_approximation)

    ###
    # Step 4: Solve
    ###
    optimize!(m)
    s_opt = JuMP.value.(s)
    s_nonzeros = findall(x -> x>0.5, s_opt)
    β = zeros(p)
    X_s = X[:, s_nonzeros]
    # Formula for the nonzero coefficients
    β[s_nonzeros] = γ * X_s' * (Y - X_s * ((I / γ + X_s' * X_s) \ (X_s'* Y)))
    
    return Dict("support" => s_opt, "coefs" => β, "selected_features" => s_nonzeros)
    
end

# LOAD X
seed = 1;
filepath = "data\\output\\preprocessed.csv"
df     = DataFrame(CSV.File(filepath; header=1,  pool=true))

# LOAD Y
X =  df[:, Not("income_total")]
X = coalesce.(X,0)
Xmat = Matrix(X)

y = df.income_total;
y = convert(Array{Float64}, coalesce.(y,0));

# (train_X, train_y), (test_X, test_y) = IAI.split_data(X, y, seed = seed)

