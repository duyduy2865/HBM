"""
    sim_bhm(data_layer::Array{BlockMaxima}; δ₀::Real,
    warmup::Int, thin::Int, niter::Int, adapt::Symbol)

Take BlockMaxima structure input and return a Chains (Mamba structure) object
containing simulation HBM model outputs (all MCMC iterations for each parameter).

# Arguments
- `data_layer::Array{BlockMaxima}`: Extremes data structure, output from `format_data`
- `δ₀::Real`: initial value for Metropolis-Hastings exploration stepsize
- `warmup::Int`: number of MCMC iterations needed for burn-in period
- `thin::Int`: number of MCMC iterations for thinning
- `niter::Int`: number of total MCMC iterations
- `adapt::Symbol`: takes three values, :warmup or :all or :none

"""

function sim_bhm(data_layer::Array{BlockMaxima}; δ₀::Real=0.5,
    warmup::Int=10000, thin::Int=10, niter::Int=20000, adapt::Symbol=:warmup)

    m = length(data_layer)
    n = Extremes.nparameter(data_layer[1])

    #vector initialization
    params = zeros(m, n, niter)
    ν = zeros(n, niter)
    τ = ones(n, niter)

    #acceptance counts for the Metropolis-Hastings step
    acc = zeros(m)

    #initialization
    Σ = Matrix{Float64}[]

    for i in 1:m
        fd = Extremes.fit(data_layer[i])
        params[i, :, 1] = fd.θ̂
        push!(Σ, inv(Symmetric(Extremes.hessian(fd))))
    end
    for i in 1:m
        if !isposdef(Σ[i])
            Σ[i] = Matrix(I, n, n)
        end
    end
    δ = δ₀ * ones(m)
    ν[:,1] = mean(params[:,:,1], dims=1)
    τ[:,1] = std(params[:,:,1], dims=1)
    u = rand(m)


    @showprogress for iter=2:niter

        #Updating the data layer parameters

        rand!(u)

        for i = 1:m

            #Normal random walk is used for Metropolis-Hastings step
            candidates = rand( MvNormal( params[i , : , iter-1], δ[i]*Σ[i] ) )

            logpd = Extremes.loglike(data_layer[i], candidates) -
                    Extremes.loglike(data_layer[i], params[i , : , iter-1]) +
                    logpdf(MvNormal(ν[:, iter-1], diagm(τ[:, iter-1])), candidates) -
                    logpdf(MvNormal(ν[:, iter-1], diagm(τ[:, iter-1])), params[i , : , iter-1])

            if logpd > log(u[i])
                params[i , : , iter] = candidates
                acc[i] += 1
            else
                params[i , : , iter] = params[i , : , iter-1]
            end
        end

        # Updating the process layer parameters
        for j in 1:n
            ν[j,iter], τ[j,iter] = invgamma_sampling(params[:, j, iter])
        end

        # Adapting the stepsize
        if iter % 50 == 0
            if !(adapt == :none)
                if (iter <= warmup) | (adapt==:all)
                    accrate = acc ./ 50
                    δ = update_stepsize.(δ, accrate)
                    acc = zeros(m)
                    for i in 1:m
                        covMat = StatsBase.cov(params[i, :, 1:iter]')
                        Σ[i] =  covMat .+ 1e-4 * tr(covMat) * Matrix(I, n, n)
                    end
                end
            end
        end
    end

    #Extracting output
    parmnames = String[]
    res = Array{Float64, 2}

    if n == 3
        parmnames = vcat(["ν_μ", "ν_ϕ", "ν_ξ", "τ_μ", "τ_ϕ", "τ_ξ"],
            ["μ[$i]" for i=1:m], ["ϕ[$i]" for i=1:m], ["ξ[$i]" for i=1:m])
        res = vcat(ν, τ, params[:, 1, :], params[:, 2, :], params[:, 3, :])
    end

    if n == 4
        parmnames = vcat(["ν_μ₀", "ν_μ₁", "ν_ϕ", "ν_ξ", "τ_μ₀", "τ_μ₁", "τ_ϕ", "τ_ξ"],
            ["μ₀[$i]" for i=1:m], ["μ₁[$i]" for i=1:m], ["ϕ[$i]" for i=1:m], ["ξ[$i]" for i=1:m])
        res = vcat(ν, τ, params[:, 1, :], params[:, 2, :], params[:, 3, :], params[:, 4, :])
    end

    if n == 5
        parmnames = vcat(["ν_μ₀", "ν_μ₁", "ν_ϕ₀", "ν_ϕ₁", "ν_ξ", "τ_μ₀", "τ_μ₁", "τ_ϕ₀", "τ_ϕ₁", "τ_ξ"],
        ["μ₀[$i]" for i=1:m], ["μ₁[$i]" for i=1:m], ["ϕ₀[$i]" for i=1:m], ["ϕ₁[$i]" for i=1:m], ["ξ[$i]" for i=1:m])
        res = vcat(ν, τ, params[:, 1, :], params[:, 2, :], params[:, 3, :], params[:, 4, :], params[:, 5, :])
    end

    res = Mamba.Chains(collect(res'), names=parmnames)

    res = res[warmup+1:thin:niter, :,:]

    #println( "Exploration stepsize after warmup: ", δ )
    println( "Acceptance rate: ", mean(acc) / (niter-warmup) )
    return res
end
