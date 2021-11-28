using JuMP, Gurobi, Random, Statistics, Combinatorics, LinearAlgebra
using DataFrames, CSV, IterTools
using Random
using GLMNet, StatsBase
using TimerOutputs

const sparseTo = TimerOutput()

seed = 2
gurobi_env = Gurobi.Env()
Random.seed!(seed)

df_path = "data/output/preprocessed.csv"
predictor_col = "income_total"
normalization_type = "std"


function calc_r2(X, y, beta)
    X = augment_X(X)
    SSres = sum( (y .- X*beta).^2 )
    SStot = sum( (y .- Statistics.mean(y)).^2 )
    return 1-SSres/SStot
end

function calc_mse(X, y, beta)
    X = augment_X(X)
    n,p = size(X)
    return sum((X*beta .- y).^2)/n
end

# function compute_mse(X, y, beta, beta0)
#     n,p = size(X)
#     return sum((beta0 .+ X*beta .- y).^2)/n
# end

function grid_search(X, y, solver_func, error_func, error_strategy="Min",train_val_ratio=0.7;params... )

    # Split the data into training/validation
    X_train, y_train, X_val, y_val = partitionTrainTest(X, y, train_val_ratio);
    
    # Create the grid (i.e. all the combinations of the given parameters)
    param_names = keys(params)
    param_combinations = [
        Dict(param_names[i]=>p[i] for i in 1:length(param_names)) 
        for p in product([params[i] for i in keys(params)]...)
    ]
    
    # Initialize variables used to hold validation information
    error_multiplier = error_strategy == "Min" ? 1 : -1
    best_error = Inf # We consider minimization
    best_param_set = []
    
    # Iterate over all combinations of parameters
    for param_comb in param_combinations
        
        # Optimize model and find optimal variables
        model_vars = solver_func(X_train,y_train;param_comb...)
        
        # Evaluate model error on validation set
        if model_vars isa Tuple
            error = error_multiplier*error_func(X_val, y_val, model_vars...)
        else
            error = error_multiplier*error_func(X_val, y_val, model_vars)
        end
        
        # If error is better than the best error so far, keep track 
        # of the error and the params
        if error < best_error
            best_error = error 
            best_param_set = param_comb
        end
    end
    
    # Retrain the model on the whole training set 
    # using the best set of params
    model_vars = solver_func(X,y;best_param_set...)
    
    # Return the model variable and the best params
    return model_vars, best_param_set
end

function normalize_data(X, method="minmax"; is_train=true)
    X = copy(X)
    if is_train
        global nonzero_idx = findall([maximum(X[:,i])-minimum(X[:,i]) for i = 1:size(X,2)].>=0.01)
        if method == "std"
            global dt=fit(ZScoreTransform, X[:,nonzero_idx]; dims=1, center=true, scale=true)
        elseif method == "minmax"
            global dt=fit(UnitRangeTransform, X[:,nonzero_idx]; dims=1, unit=true)
        end
    end
    X[:,nonzero_idx] = StatsBase.transform(dt, X[:,nonzero_idx])
    
    return X
end


function partitionTrainTest(X,y, at = 0.7, s=seed)
    n = size(X,1)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    return X[train_idx,:], y[train_idx], X[test_idx,:], y[test_idx]
end

function augment_X(X)
    return [ones(size(X,1),1) X]
end

function solve_holistic_regr(X,y;gamma,rho,k)
    C = cor(X)
    n,p = size(X)
    X_aug = augment_X(X)
    M = 10^5
    m = Model(with_optimizer(Gurobi.Optimizer, gurobi_env))
    set_optimizer_attribute(m, "OutputFlag", 0)
    set_optimizer_attribute(m, "PSDTol", 1)
    @variable(m, beta[1:(p+1)])
    @variable(m, z[1:p],Bin)
    @variable(m, t[1:p])
    @objective(m, Min, 1/2*sum((X_aug*beta.-y).^2)+gamma*sum(t[i] for i=1:p))
    @constraint(m, [i=1:p], t[i]>= beta[i])
    @constraint(m, [i=1:p], t[i]>= -beta[i])
    @constraint(m, [i=1:p], beta[i]<= M*z[i])
    @constraint(m, [i=1:p], -M*z[i]<=beta[i])
    @constraint(m, sum(z)<=k)
    #@constraint(m, [i=1:4:p-3], sum(z[i+j] for j=0:3)<=1)
    #     for i in 1:p
    #         for j in i+1:p
    #             if abs(C[i,j]) > rho
    #                 @constraint(m, z[i]+z[j] <= 1)
    #             end
    #         end
    #     end
    optimize!(m)
    return JuMP.value.(beta)
end

function fit_lasso(X, y)
    cv = glmnetcv(X, y);
    id_best = argmin(cv.meanloss);
    betas = [GLMNet.coef(cv);cv.path.a0[id_best]];
    return betas
end

function solve_inner_problem(X,Y,s,γ)
    indices = findall(s .> 0.5)
    n = length(Y)
    denom = 2*n
    Xs = X[:, indices]
    α = Y - Xs * (inv(I / γ + Xs' * Xs) * (Xs'* Y))
    obj = dot(Y, α) / denom
    tmp = X' * α
    grad = -γ .* tmp .^ 2 ./ denom
  return obj, grad
end

function sparse_regression(X,Y,k,γ,s0=[],is_binary=false; outFlag = 1)
    @timeit sparseTo "Sparse Regression" begin
        m = Model(Gurobi.Optimizer)
        set_optimizer_attribute(m, "OutputFlag", outFlag)
        set_optimizer_attribute(m, "TimeLimit", 45)
        n,p = size(X)
        
        ###
        # Step 1: Define the Variables:
        ###
        
        if is_binary
            @variable(m, s[1:p], Bin)
            #@constraint(m, s[1:p] >= 0)
        else
            @variable(m, s[1:p]>=0)
            @constraint(m, [i=1:p], s[i] <= 1)
        end
        
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
            @timeit sparseTo "Sparse Outter Approximation" begin
                s_val = []
                for i = 1:p
                    s_val = [s_val;callback_value(cb_data, s[i])]
                end
                @timeit sparseTo "Sparse Inner Problem" obj, grad = solve_inner_problem(X,Y, s_val, γ)
                # add the cut: t >= obj + sum(∇s * (s - s_val))
                offset = sum(grad .* s_val)
                con = @build_constraint(t >= obj + sum(grad[j] * s[j] for j=1:p) - offset)    
                MOI.submit(m, MOI.LazyConstraint(cb_data), con)
            end
        end
        MOI.set(m, MOI.LazyConstraintCallback(), outer_approximation)

        ###
        # Step 4: Solve
        ###
        optimize!(m)
        s_opt = JuMP.value.(s)
        
        s_nonzeros = []
        # println(s_opt)
        # println("t: $(JuMP.value(t))")
        if !is_binary
            idxes = sortperm(s_opt, rev=true)
            s = zeros(p)
            s[idxes[1:k]] = ones(k)
            s_nonzeros = idxes
        else
            s_nonzeros = findall(x -> x>0.5, s_opt)
        end
        β = zeros(p)
        X_s = X[:, s_nonzeros]
        # Formula for the nonzero coefficients
        β[s_nonzeros] = γ * X_s' * (Y - X_s * ((I / γ + X_s' * X_s) \ (X_s'* Y)))
        
        #return Dict("support" => s_opt, "coefs" => β, "selected_features" => s_nonzeros)
        return is_binary ? [0;β] : β 
    end
end

df = DataFrame(CSV.File(df_path, header=1))
# df = last(df, 1000)
df = df[shuffle(1:nrow(df))[1:300000], :]

names(df)

excluded_cols = [
    "earnings_total",
    "income_interest_dividends_rental",
    "income_retirement",
    "income_all",
    "income_social_security",
    "income_supplementary_security",
    "income_total",
    "income_self_employment",
    "income_household",
    "income_to_poverty_ratio",
    "income_public_assistance",
    "income_family",
    "income_wages_salary",
    "monthly_owner_costs",
    "gross_rent",
    "person_number",
    "rent_monthly",
    "property_value"
]
cols = filter(x -> x ∉ excluded_cols, names(df))
X, y = Matrix{Float32}(df[!, filter(x -> x != predictor_col, cols)]), df[!,predictor_col]

X_train, y_train, X_test, y_test = partitionTrainTest(X, y, 0.7);
X_train = normalize_data(X_train, normalization_type; is_train=true);
# y_train = normalize_data(y_train; is_train=true);
X_test = normalize_data(X_test, normalization_type; is_train=false);
# y_test = normalize_data(y_test; is_train=false);


k = 50
#betas, params = grid_search(X_train, y_train, solve_holistic_regr, calc_r2, "Max", 0.7; gamma=[0.1], rho=[0.7], k=[20])
#betas = fit_lasso(X_train, y_train)
reset_timer!(sparseTo)
betas_lasso = fit_lasso(X_train, y_train)
betas = sparse_regression(X_train, y_train, k ,1/sqrt(size(X_train,1)), 1.0*(betas_lasso[2:end] .>= 0.5), true)
sparseTo


r2_c = calc_r2(X_test, y_test, betas)
mse_c = calc_mse(X_test, y_test, betas)

println("r^2 train $(calc_r2(X_train, y_train, betas))")
println("r^2 test $(calc_r2(X_test, y_test, betas))")
println("mse train $(calc_mse(X_train, y_train, betas))")
println("mse test $(calc_mse(X_test, y_test, betas))")

reset_timer!(sparseTo)

#cols = filter(x -> x ∉ excluded_cols, names(df))
#important_features = cols[findall(abs.(betas_hol) .>= 0.01)]
THRESHOLD = 0.000001
println("Most important features:")
for i in sortperm(abs.(betas[2:end]), rev=true)
    if abs(betas[i+1])<=THRESHOLD
        continue
    end
    println("- $(cols[i]) : $(betas[i+1])")
end

function iai2betas(learner, p)
    beta0 = IAI.get_prediction_constant(learner)
    betas = IAI.get_prediction_weights(learner)[1]
    
    features = string.(collect(keys(betas)))

    beta_coeffs = zeros(p)
    for i = 1:p
        if "x$i" in features
            beta_coeffs[i] = betas[Symbol("x$i")]
        end
    end
            
    return [beta0 ; beta_coeffs]
end

@time begin    
    m = IAI.OptimalFeatureSelectionRegressor(
        sparsity=50
    )
    res = IAI.fit!(m, X_train, y_train)
end

betas_iai = iai2betas(m, size(X,2))

IAI.score(m, X_train, y_train)
IAI.score(m, X_test, y_test)

r2_c = calc_r2(X_test, y_test, betas_iai)
mse_c = calc_mse(X_test, y_test, betas_iai)

println("r^2 train $(calc_r2(X_train, y_train, betas_iai))")
println("r^2 test $(calc_r2(X_test, y_test, betas_iai))")
println("mse train $(calc_mse(X_train, y_train, betas_iai))")
println("mse test $(calc_mse(X_test, y_test, betas_iai))")


THRESHOLD = 0.000001
for i in sortperm(abs.(betas_iai[2:end]), rev=true)
    if abs(betas_iai[i+1])<=THRESHOLD
        continue
    end
    println("- $(cols[i]) : $(betas_iai[i+1])")
end

