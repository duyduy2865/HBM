function findposteriormode(output::Chains, mb::Int, data_layer::Array{BlockMaxima})

    m = length(data_layer)
    npar = Extremes.nparameter(data_layer[1])
    θ = Extremes.slicematrix(output.value[:, 2*npar+mb : m : 2*npar+mb+(npar-1)*m, 1], dims=2)
    l = Extremes.loglike.(data_layer[mb], θ)
    ind = argmax(l)
    θ̂ = θ[ind]

    return θ̂
end

function ecdf2(θ::Array{Float64}, mb::Int, data_layer::Array{BlockMaxima})

    dist = Extremes.getdistribution(data_layer[mb], θ)

    μ = location.(dist)
    σ = Distributions.scale.(dist)
    ξ = shape.(dist)

    z = Extremes.standardize.(data_layer[mb].data.value, μ, σ, ξ)
    return Extremes.ecdf(z)
end

"""
    qqplot_map(output::Chains, mb::Int, data_layer::Array{BlockMaxima})

Return standardized QQ-plot for HBM `output` of member `mb`, from input `data_layer`.
Here the estimator used is the Maximum A Posteriori estimator for GEV parameters.

"""
function qqplot_map(output::Chains, mb::Int, data_layer::Array{BlockMaxima})

    θ̂ = findposteriormode(output, mb, data_layer)
    #standardized residuals
    val, p = ecdf2(θ̂, mb, data_layer)
    return DataFrame(Model = quantile.(Gumbel(), p), Empirical = val)
end

"""
    qqplot_mean(output::Chains, mb::Int, data_layer::Array{BlockMaxima})

Return standardized QQ-plot for HBM `output` of member `mb`, from input `data_layer`.
Here the estimator used is the mean of each MCMC iteration estimate for GEV parameters.

"""
function qqplot_mean(output::Chains, mb::Int, data_layer::Array{BlockMaxima})

    m = length(data_layer)
    npar = Extremes.nparameter(data_layer[1])
    θ = Extremes.slicematrix(output.value[:, 2*npar+mb : m : 2*npar+mb+(npar-1)*m, 1], dims=2)
    z = [ecdf2(θ[i], mb, data_layer)[1] for i = 1:length(θ)]
    p = ecdf2(θ[1], mb, data_layer)[2]
    zp = Extremes.unslicematrix(z)'
    lower = [quantile(zp[:, i], 0.025) for i = 1:size(zp,2)]
    upper = [quantile(zp[:, i], 0.975) for i = 1:size(zp,2)]
    return DataFrame(Model = quantile.(Gumbel(), p), Empirical = mean(z), Lower = lower, Upper= upper)
end
