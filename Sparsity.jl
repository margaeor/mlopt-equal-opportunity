using JuMP, Gurobi, Random, Statistics, Combinatorics, LinearAlgebra
using DataFrames, CSV, IterTools
using Random
using GLMNet, StatsBase
using TimerOutputs
using Gadfly

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


function grid_search(X, y, solver_func, error_func, groups, groupKs, error_strategy="Min",train_val_ratio=0.7; params... )

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
    # println("----------------------------------------")    
    # println(param_combinations)
    # println("----------------------------------------")
    # println(param_combinations)


    # Iterate over all combinations of parameters
    for param_comb in param_combinations
        println("**********************")
        println(param_comb)
        println("**********************")
        
        # Optimize model and find optimal variables
        global model_vars = solver_func(X_train,y_train, groups, groupKs;param_comb...)
        
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

#########################################################################################################
#########################################################################################################
#########################################################################################################
function solve_holistic_regr(X,y, groups, groupKs;gamma,rho,k, outFlag = 1)
    C = cor(X)
    n,p = size(X)
    X_aug = augment_X(X, true)
    M = 10^5
    # m = Model(with_optimizer(Gurobi.Optimizer, gurobi_env))
    m = Model(with_optimizer(Gurobi.Optimizer))
    set_optimizer_attribute(m, "OutputFlag", outFlag)
    set_optimizer_attribute(m, "PSDTol", 1)
    # set_optimizer_attribute(m, "NonConvex", 2)
    set_optimizer_attribute(m, "TimeLimit", 60)
    @variable(m, beta[1:(p+1)])
    @variable(m, z[1:p],Bin)
    @variable(m, t[1:p])
    @objective(m, Min, 1/2*sum((X_aug*beta.-y).^2)+gamma*sum(t[i] for i=1:p))
    @constraint(m, [i=1:p], t[i]>= beta[i])
    @constraint(m, [i=1:p], t[i]>= -beta[i])
    @constraint(m, [i=1:p], beta[i]<= M*z[i])
    @constraint(m, [i=1:p], -M*z[i]<=beta[i])
    @constraint(m, sum(z)<=k)

    for (index, group) in enumerate(groups)
        @constraint(m, sum(z[i] for i in group)<=groupKs[index])
    end

    for i in 1:p
        for j in i+1:p
            if abs(C[i,j]) > rho
                @constraint(m, z[i]+z[j] <= 1)
            end
        end
    end

    optimize!(m)
    return JuMP.value.(beta)
end
#########################################################################################################
#########################################################################################################
#########################################################################################################

function printFeatures(betas, cols, isGroups = false; groups = [])    
    if isGroups
        for (index, group) in enumerate(groups)
            println("Features selected from Group $(index) :")  
            
            # sortperm(abs.(betas[2:end]), rev=true)
            grpCounter = 0
            tmpBetas = betas_holistic[sort!(collect(group))]
            tmpCols = cols[sort!(collect(group))]
            for i in sortperm(abs.(tmpBetas), rev=true)
                if tmpBetas[i] != 0
                    grpCounter = grpCounter + 1;
                    println("$i - $(tmpBetas[i]) - $(tmpCols[i])")
                end
            end
            println("Total: $(grpCounter) Features from Group $(index)")
            println("--------------------------------------------------")
        end
    else
        THRESHOLD = 0.000001
        grpCounter = 0

        for i in sortperm(abs.(betas[2:end]), rev=true)
            grpCounter = grpCounter + 1;

            if abs(betas[i+1])<=THRESHOLD
                grpCounter = grpCounter - 1;
                continue
            end
            println("- $(cols[i]) : $(betas[i+1])")
        end
        println("Total: $(grpCounter) Features")
        println("-----------------------------")
    end 
end

function getMetrics(betas, X_train, y_train, X_test, y_test)    
    r2_c = calc_r2(X_test, y_test, betas)
    mse_c = calc_mse(X_test, y_test, betas)
    println("r^2 train $(calc_r2(X_train, y_train, betas))")
    println("r^2 test $(calc_r2(X_test, y_test, betas))")
    println("mse train $(calc_mse(X_train, y_train, betas))")
    println("mse test $(calc_mse(X_test, y_test, betas))")
    return r2_c, mse_c
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

function augment_X(X, flag = false)
    if flag
        return [X ones(size(X,1),1)]
    else        
        return [ones(size(X,1),1) X]
    end
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

function sparse_regression(X,Y,k,γ,s0=[],is_binary=false; outFlag = 1, timeLimit = 60)
    @timeit sparseTo "Sparse Regression" begin
        m = Model(Gurobi.Optimizer)
        set_optimizer_attribute(m, "OutputFlag", outFlag)
        set_optimizer_attribute(m, "TimeLimit", timeLimit)
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
names(df)
dfSmall = df[shuffle(1:nrow(df))[1:10000], :]
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
    # "property_value",
    "mortgage_first_payment",
    "gross_rent_pcnt_income",
    "electricity_cost",
    "cost_gas",
    "cost_fuel",
    "income_adjustment_factor"
]
cols = filter(x -> x ∉ excluded_cols, names(df))


################################################################### 
## █▀ █▀█ ▄▀█ █▀█ █▀ █▀▀   █▀█ █▀▀ █▀▀ █▀█ █▀▀ █▀ █▀ █ █▀█ █▄░█  ##
## ▄█ █▀▀ █▀█ █▀▄ ▄█ ██▄   █▀▄ ██▄ █▄█ █▀▄ ██▄ ▄█ ▄█ █ █▄█ █░▀█  ##
###################################################################
X, y = Matrix{Float32}(df[!, filter(x -> x != predictor_col, cols)]), df[!,predictor_col]
X_train, y_train, X_test, y_test = partitionTrainTest(X, y, 0.7);
X_train = normalize_data(X_train, normalization_type; is_train=true);
X_test = normalize_data(X_test, normalization_type; is_train=false);


k = 50
reset_timer!(sparseTo)
betas_lasso = fit_lasso(X_train, y_train)
getMetrics(betas_lasso, X_train, y_train, X_test, y_test)

betas_sparse = sparse_regression(X_train, y_train, k ,1/sqrt(size(X_train,1)), 1.0*(betas_lasso[1:end-1] .>= 0.5), true, timeLimit = 120)
reset_timer!(sparseTo)
getMetrics(betas_sparse, X_train, y_train, X_test, y_test)
printFeatures(betas_sparse, cols, false)


#################################################################################
## █ ▄▀█ █   █▀▀ █▀▀ ▄▀█ ▀█▀ █░█ █▀█ █▀▀   █▀ █▀▀ █░░ █▀▀ █▀▀ ▀█▀ █ █▀█ █▄░█   ##
## █ █▀█ █   █▀░ ██▄ █▀█ ░█░ █▄█ █▀▄ ██▄   ▄█ ██▄ █▄▄ ██▄ █▄▄ ░█░ █ █▄█ █░▀█   ##
#################################################################################

@time begin
    m = IAI.OptimalFeatureSelectionRegressor(
        sparsity=70
    )
    res = IAI.fit!(m, X_train, y_train)
end

betas_iai = iai2betas(m, size(X,2))

IAI.score(m, X_train, y_train)
IAI.score(m, X_test, y_test)
r2_c, mse_c = getMetrics(betas_iai, X_train, y_train, X_test, y_test)
printFeatures(betas_iai, cols)


using DataStructures

grp1betas, grp1cols, grp2betas, grp2cols = plotGroups(betas_iai, cols)

Gadfly.push_theme(:dark)
set_default_plot_size(60cm, 60cm)
using dataset

using RDatasets

Gadfly.plot(grp1betas, x="Chest", y="Count", Geom.bar)

plot(dataset("MASS", "nlschools"), x="IQ", y="Lang", color="COMB",
            Geom.point, Geom.smooth(method=:lm), Guide.colorkey("Multi-Grade"))

set_default_plot_size(60cm, 300cm)
vstack(p1, p2)
plot(p1, p2)
theme(:dark)

Pkg.add("Cairo")
using Cairo 

p1 = Gadfly.plot([sin,cos], 0, 2pi)
p2 = Gadfly.plot((x,y)->sin(x)+cos(y), 0, 2pi, 0, 2pi)
p3 = Gadfly.spy(ones(33)*sin.(0:(pi/16):2pi)' + cos.(0:(pi/16):2pi)*ones(33)')
hstack(p1,p2,p3)





function plotGroups(betas, cols)

    grp1betas   = Float64[]
    grp1cols    = String[]
    grp1Counter = 0
    grp2betas   = Float64[]
    grp2cols    = String[]
    grp2Counter = 0
    grp3betas   = Float64[]
    grp3cols    = String[]
    grp3Counter = 0
    grp4betas   = Float64[]
    grp4cols    = String[]
    grp4Counter = 0
    THRESHOLD = 0.000001

    for i in sortperm(abs.(betas[2:end]), rev=true)
        if abs(betas[i+1])>=THRESHOLD

            if occursin("field", cols[i])
                println("$i - Group 1 : $(cols[i])")
                push!(grp1betas, betas[i+1])
                push!(grp1cols, cols[i])
                grp1Counter += 1
            elseif occursin("occupation", cols[i])
                println("$i - Group 2 : $(cols[i])")
                push!(grp2betas, betas[i+1])
                push!(grp2cols, cols[i])
                grp2Counter += 1
            elseif occursin("selfcare", cols[i]) || occursin("sex", cols[i]) || occursin("cognitive", cols[i]) || occursin("race", cols[i]) || occursin("parents", cols[i])
                println("$i - Group 3 : $(cols[i])")
                push!(grp3betas, betas[i+1])
                push!(grp3cols, cols[i])
                grp3Counter += 1
            else
                println("$i - Group 4 : $(cols[i])")
                push!(grp4betas, betas[i+1])
                push!(grp4cols, cols[i])
                grp4Counter += 1
            end

        end
    end
    println()
    println("Group 1: $(grp1Counter) - Group 2: $(grp2Counter) - Group 3: $(grp3Counter) - Group 4: $(grp4Counter)")
    return grp1betas, grp1cols, grp2betas, grp2cols
end

########################################################################
## █░█ █▀█ █░░ █ █▀ ▀█▀ █ █▀▀   █▀█ █▀▀ █▀▀ █▀█ █▀▀ █▀ █▀ █ █▀█ █▄░█  ##
## █▀█ █▄█ █▄▄ █ ▄█ ░█░ █ █▄▄   █▀▄ ██▄ █▄█ █▀▄ ██▄ ▄█ ▄█ █ █▄█ █░▀█  ##
########################################################################

seed = 4
Nhol = 5000
grpAll = Set(1:length(cols))
fodInit = 205; fodEnd = 241
 # FIELD OF DEGREE GROUP ------------------------------- 1
grp1 = Set(fodInit:fodEnd)
socInit = 102; socEnd = 127
# OCCUPATION CODE GROUP -------------------------------- 2
grp2 = Set(socInit:socEnd) 
# A-PRIORI TRAITS (Sex, Race, Disabilities) GROUP ------ 3
# SEX GROUP
sexInit = 173; sexEnd = 175;
sexGrp = Set(sexInit:sexEnd)
# DISABILITIES GROUP
disInit1 = 9; disEnd1 = 11;
disInit2 = 170; disEnd2 = 172;
disGrp = union(Set(disInit1:disEnd1), Set(disInit2:disEnd2))
# RACE GROUP
raceInit = 137; raceEnd = 163;
raceGrp = Set(raceInit:raceEnd)
# UNITE ALL
grp3 = union(raceGrp, disGrp, sexGrp)
grp4 = setdiff(grpAll, union(grp1, grp2, grp3))

groups = [grp1 grp2 grp3 grp4]
groupKs = [20 25 30 25]

global indexArr = Int[]
global nzArr = Int[]
global rsqArr = Float64[]

# for seed = [19]
for seed = 15:25
cols = filter(x -> x ∉ excluded_cols, names(df))

    Random.seed!(seed)
    df2 = df[shuffle(1:nrow(df))[1:Nhol], :]
    X, y = Matrix{Float32}(df2[!, filter(x -> x != predictor_col, cols)]), df2[!,predictor_col]
    X_train, y_train, X_test, y_test = partitionTrainTest(X, y, 0.7);
    X_train = normalize_data(X_train, normalization_type; is_train=true);
    X_test = normalize_data(X_test, normalization_type; is_train=false);

    try
        betas_holistic, params_holistic = grid_search(X_train, y_train, solve_holistic_regr, calc_r2,  groups, groupKs , "Max", 0.7; gamma=[0.5 1], rho=[0.5 0.7], k=[50 75])
        println("Workeed -- $(seed)")
        nzeros = (length(betas_holistic[betas_holistic .!= 0]))
        println("NONZEROS = $(length(betas_holistic[betas_holistic .!= 0]))")
        push!(indexArr, seed)
        push!(nzArr, nzeros)

        betas_holistic2 = [betas_holistic[end] ; betas_holistic[1:end-1]]
        r2_c = calc_r2(X_test, y_test, betas_holistic2)
        push!(rsqArr, r2_c)
    catch
        println("Error (probably psd) -- $(seed)")
    end
# end

############################################################################################### 
printFeatures(betas_holistic, cols, true; groups)
###############################################################################################
betas_holistic2 = [betas_holistic[end] ; betas_holistic[1:end-1]]

r2_c = calc_r2(X_test, y_test, betas_holistic2)
mse_c = calc_mse(X_test, y_test, betas_holistic2)

for i in 1:length(cols)
    println("$i - $(cols[i])")
end

