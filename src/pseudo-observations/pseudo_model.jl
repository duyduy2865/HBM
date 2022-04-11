function getgevparams(vec::Array{Float64}; x::Float64=0.)
    n = length(vec)
    if n == 3
        return vec[1], exp(vec[2]), vec[3]
    elseif n == 4
        return vec[1] + x * vec[2], exp(vec[3]), vec[4]
    elseif n == 5
        return vec[1] + x * vec[2], exp(vec[3] + x * vec[4]), vec[5]
    end
end

"""
    obs_bhm(μh::Matrix{Float64}, σh::Matrix{Float64}; δ₀::Real,
    warmup::Int, thin::Int, niter::Int, adapt::Symbol)

Take BlockMaxima structure input and return a Chains (Mamba structure) object
containing simulation HBM model outputs (all MCMC iterations for each parameter).

# Arguments
- `μh::Matrix{Float64}`: mean parameter matrix, output from format_data2
- `σh::Matrix{Float64}`: standard deviation parameter matrix, output from format_data2
- `δ₀::Real`: initial value for Metropolis-Hastings exploration stepsize
- `warmup::Int`: number of MCMC iterations needed for burn-in period
- `thin::Int`: number of MCMC iterations for thinning
- `niter::Int`: number of total MCMC iterations
- `adapt::Symbol`: takes three values, :warmup or :all or :none

"""

function obs_bhm(μh::Matrix{Float64}, σh::Matrix{Float64}; δ₀::Real=0.5,
    warmup::Int=10000, thin::Int=10, niter::Int=20000, adapt::Symbol=:warmup)

    years, S = size(μh)

    Y = zeros(years, niter)
    params = zeros(3, niter)

    #acceptance counts for the Metropolis-Hastings step
    accY = zeros(years)
    accgev = zeros(3)

    #initialization
    Y[:, 1] = mean(exp.(μh), dims = 2)
    fd = Extremes.gevfit(Y[:, 1])
    params[:, 1] = fd.θ̂

    δY = δ₀ * ones(years)
    δ = δ₀ * ones(3)
    uY = rand(years)

    @showprogress for iter=2:niter

        rand!(uY)

        #Updating the maxima

        for y = 1:years

            candidate = max(1e-3, rand(Normal(Y[y, iter-1], δY[y])))
            gev = getgevparams(params[:, iter-1])

            logpd = sum(logpdf.(LogNormal.(μh[y, :], σh[y , :]), candidate)) -
            sum(logpdf.(LogNormal.(μh[y, :], σh[y , :]), Y[y, iter-1])) +
            #logpd = sum(logpdf.(Normal.(μh[y, :], σh[y , :]), log(candidate))) -
            #sum(logpdf.(Normal.(μh[y, :], σh[y , :]), log(Y[y, iter-1]))) - log(candidate) + log(Y[y, iter-1]) +
            logpdf(GeneralizedExtremeValue(gev...), candidate) -
            logpdf(GeneralizedExtremeValue(gev...), Y[y, iter-1])

            if logpd > log(uY[y])
                Y[y, iter] = candidate
                accY[y] += 1
            else
                Y[y, iter] = Y[y, iter-1]
            end
        end

        data_layer = BlockMaxima(Variable("y", Y[:, iter]))

        #Updating the GEV parameters

        params[:, iter] = params[:, iter - 1]

        new_η = rand(Normal(params[1, iter], δ[1]))
        new = vcat(new_η, params[2:end, iter])
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter])

        if logpd > log(rand())
            params[1, iter] = new_η
            accgev[1] += 1
        end

        new_ζ = rand(Normal(params[2, iter], δ[2]))
        new = vcat(params[1, iter], new_ζ, params[3, iter])
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter])

        if logpd > log(rand())
            params[2, iter] = new_ζ
            accgev[2] += 1
        end

        new_ξ = rand(Normal(params[3, iter], δ[3]))
        new = vcat(params[1:2, iter], new_ξ)
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter]) +
        logpdf(Beta(6., 9.), new_ξ + 0.5) - logpdf(Beta(6., 9.), params[3, iter] + 0.5)

        if logpd > log(rand())
            params[3, iter] = new_ξ
            accgev[3] += 1
        end

        # Updating the stepsize
        if iter % 50 == 0
            if !(adapt == :none)
                if (iter <= warmup) | (adapt==:all)
                    accrate = accY ./ 50
                    δY = update_stepsize.(δY, accrate)
                    accY = zeros(years)
                    accrate = accgev ./ 50
                    δ = update_stepsize.(δ, accrate)
                    accgev = zeros(3)
                end
            end
        end
    end

    #Extracting output
    parmnames = vcat(["η", "ζ", "ξ"], ["Y[$y]" for y = 1:years])
    res = vcat(params, Y)

    res = Mamba.Chains(collect(res'), names=parmnames)
    res = res[warmup+1:thin:niter, :, :]

    println( "Exploration stepsize after warmup: ", δY )
    println( "Acceptance rate for maxima: ", mean(accY) / (niter-warmup) )

    println( "Exploration stepsize after warmup: ", δ )
    println( "Acceptance rate for GEV params: ", accgev / (niter-warmup) )
    return res
end

function obs_bhm(μh::Matrix{Float64}, σh::Matrix{Float64}, xcov::Array{Float64}; δ₀::Real=0.5,
    warmup::Int=10000, thin::Int=10, niter::Int=20000, adapt::Symbol=:warmup)

    years, S = size(μh)

    Y = zeros(years, niter)
    params = zeros(4, niter)

    #acceptance counts for the Metropolis-Hastings step
    accY = zeros(years)
    accgev = zeros(4)

    #initialization
    Y[:, 1] = mean(exp.(μh), dims = 2)
    data_layer = BlockMaxima(Variable("y", Y[:, 1]), locationcov = [Variable("co2", xcov)])
    fd = Extremes.fit(data_layer)
    params[:, 1] = fd.θ̂
    δY = δ₀ * ones(years)
    δ = δ₀ * ones(4)
    uY = rand(years)

    @showprogress for iter=2:niter

        rand!(uY)

        #Updating the maxima

        for y = 1:years

            candidate = max(1e-3, rand(Normal(Y[y, iter-1], δY[y])))
            gev = getgevparams(params[:, iter-1], x=xcov[y])

            logpd = sum(logpdf.(LogNormal.(μh[y, :], σh[y , :]), candidate)) -
            sum(logpdf.(LogNormal.(μh[y, :], σh[y , :]), Y[y, iter-1])) +
            #logpd = sum(logpdf.(Normal.(μh[y, :], σh[y , :]), log(candidate))) -
            #sum(logpdf.(Normal.(μh[y, :], σh[y , :]), log(Y[y, iter-1]))) - log(candidate) + log(Y[y, iter-1]) +
            logpdf(GeneralizedExtremeValue(gev...), candidate) -
            logpdf(GeneralizedExtremeValue(gev...), Y[y, iter-1])

            if logpd > log(uY[y])
                Y[y, iter] = candidate
                accY[y] += 1
            else
                Y[y, iter] = Y[y, iter-1]
            end
        end

        data_layer = BlockMaxima(Variable("y", Y[:, iter]), locationcov = [Variable("co2", xcov)])

        #Updating the GEV parameters

        params[:, iter] = params[:, iter - 1]

        new_η0 = rand(Normal(params[1, iter], δ[1]))
        new = vcat(new_η0, params[2:end, iter])
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter])

        if logpd > log(rand())
            params[1, iter] = new_η0
            accgev[1] += 1
        end

        new_η1 = rand(Normal(params[2, iter], δ[2]))
        new = vcat(params[1, iter], new_η1, params[3:end, iter])
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter])

        if logpd > log(rand())
            params[2, iter] = new_η1
            accgev[2] += 1
        end

        new_ζ = rand(Normal(params[3, iter], δ[3]))
        new = vcat(params[1:2, iter], new_ζ, params[end, iter])
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter])

        if logpd > log(rand())
            params[3, iter] = new_ζ
            accgev[3] += 1
        end

        new_ξ = rand(Normal(params[4, iter], δ[4]))
        new = vcat(params[1:3, iter], new_ξ)
        logpd = Extremes.loglike(data_layer, new) - Extremes.loglike(data_layer, params[:, iter]) +
        logpdf(Beta(6., 9.), new_ξ + 0.5) - logpdf(Beta(6., 9.), params[4, iter] + 0.5)

        if logpd > log(rand())
            params[4, iter] = new_ξ
            accgev[4] += 1
        end

        # Updating the stepsize
        if iter % 50 == 0
            if !(adapt == :none)
                if (iter <= warmup) | (adapt==:all)
                    accrate = accY ./ 50
                    δY = update_stepsize.(δY, accrate)
                    accY = zeros(years)
                    accrate = accgev ./ 50
                    δ = update_stepsize.(δ, accrate)
                    accgev = zeros(4)
                end
            end
        end
    end

    #Extracting output
    parmnames = vcat(["η₀", "η₁", "ζ", "ξ"], ["Y[$y]" for y = 1:years])
    res = vcat(params, Y)

    res = Mamba.Chains(collect(res'), names=parmnames)
    res = res[warmup+1:thin:niter, :, :]

    println( "Exploration stepsize after warmup: ", δY )
    println( "Acceptance rate for maxima: ", mean(accY) / (niter-warmup) )

    println( "Exploration stepsize after warmup: ", δ )
    println( "Acceptance rate for GEV params: ", accgev / (niter-warmup) )
    return res
end
