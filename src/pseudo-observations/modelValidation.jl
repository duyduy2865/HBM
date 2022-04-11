function ecdf2(data::Array{Float64}, μ::Float64, σ::Float64, ξ::Float64)

    z = Extremes.standardize.(data, μ, σ, ξ)
    return Extremes.ecdf(z)
end

function ecdf2(data::Array{Float64}, μ::Array{Float64}, σ::Float64, ξ::Float64)

    z = Extremes.standardize.(data, μ, σ, ξ)
    return Extremes.ecdf(z)
end

"""
    qqplot_mean(output::Chains, mb::Int, data_layer::Array{BlockMaxima})

Return standardized QQ-plot for HBM `output` of member `mb`, from input `data_layer`.
Here the estimator used is the mean of each MCMC iteration estimate for GEV parameters.

"""
function qqplot_mean(output::Chains; xcov::Array{Float64}=zeros(1))
    val = output.value
    μ = 0.; σ = 0.; ξ = 0.; Y = 0.
    if "η₁" in output.names
        μ₀ = val[:, 1, 1]; μ₁ = val[:, 2, 1]; ϕ = val[:, 3, 1]; ξ = val[:, 4, 1]
        μ = μ₀ .+ μ₁ .* xcov'; σ = exp.(ϕ)
        Y = val[:, 5:end, 1]
        z = [ecdf2(Y[i, :], μ[i, :], σ[i], ξ[i])[1] for i = 1:size(val, 1)]
        p = ecdf2(Y[1, :], μ[1, :], σ[1], ξ[1])[2]
    else
        μ = val[:, 1, 1]; ϕ = val[:, 2, 1]; ξ = val[:, 3, 1]
        σ = exp.(ϕ)
        Y = val[:, 4:end, 1]
        z = [ecdf2(Y[i, :], μ[i], σ[i], ξ[i])[1] for i = 1:size(val, 1)]
        p = ecdf2(Y[1, :], μ[1], σ[1], ξ[1])[2]
    end

    zp = Extremes.unslicematrix(z)'
    lower = [quantile(zp[:, i], 0.025) for i = 1:size(zp,2)]
    upper = [quantile(zp[:, i], 0.975) for i = 1:size(zp,2)]
    return DataFrame(Model = quantile.(Gumbel(), p), Empirical = mean(z), Lower = lower, Upper= upper)
end
