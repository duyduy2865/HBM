"""
    getsimpd(output::Chains, mb::Int, data_layer::Array{BlockMaxima}, norm_moy::Float64, norm_std::Float64)

Return all estimated rescaled GEV distributions for member of simulation `mb` of BHM simulation model `output` and input `data_layer`.
There is one GEV distribution for each MCMC iteration and each year (for non-stationary models).

# Arguments
- `norm_moy::Float64`: normalizing mean of input maxima data
- `norm_std::Float64`: normalizing standard deviation of input maxima data

"""

function getsimpd(output::Chains, mb::Int, data_layer::Array{BlockMaxima}, norm_moy::Float64, norm_std::Float64)

    val = output.value
    m = length(data_layer)
    npar = Extremes.nparameter(data_layer[1])
    it = size(val,1)
    μ = zeros(it, npar); σ = zeros(it, npar); ξ = zeros(it, npar)

    if npar == 5
        xcov = data_layer[mb].location.covariate[1].value
        μ₀ = val[:, 2*npar+mb, 1]
        μ₁ = val[:, 2*npar+m+mb, 1]
        ϕ₀ = val[:, 2*npar+2*m+mb, 1]
        ϕ₁ = val[:, 2*npar+3*m+mb, 1]

        ξ = val[:, 2*npar+4*m+mb, 1]
        μ = (μ₀ .+ μ₁ .* xcov') .* norm_std .+ norm_moy
        σ = exp.(ϕ₀ .+ ϕ₁ .* xcov') .* norm_std
    end

    if npar == 4
        xcov = data_layer[mb].location.covariate[1].value
        μ₀ = val[:, 2*npar+mb, 1]
        μ₁ = val[:, 2*npar+m+mb, 1]
        ϕ = val[:, 2*npar+2*m+mb, 1]

        ξ = val[:, 2*npar+3*m+mb, 1]
        μ = (μ₀ .+ μ₁ .* xcov') .* norm_std .+ norm_moy
        σ = exp.(ϕ) .* norm_std
    end

    if npar == 3
        μ = val[:, 2*npar+mb, 1]
        ϕ = val[:, 2*npar+m+mb, 1]

        ξ = val[:, 2*npar+2*m+mb, 1]
        μ = μ .* norm_std .+ norm_moy
        σ = exp.(ϕ) .* norm_std
    end

    return GeneralizedExtremeValue.(μ, σ, ξ)
end

"""
    getobspd(output::Chains)

Return estimated GEV distributions for pseudo-observation BHM model `output`.
There is one GEV distribution for each MCMC iteration.

"""

function getobspd(output::Chains; xcov::Array{Float64}=zeros(1))

    val = output.value
    if "ζ₁" in output.names
        μ₀ = val[:, 1, 1]; μ₁ = val[:, 2, 1]; ϕ₀ = val[:, 3, 1]; ϕ₁ = val[:, 4, 1]; ξ = val[:, 5, 1]
        μ = μ₀ .+ μ₁ .* xcov'; σ = exp.(ϕ₀ .+ ϕ₁ .* xcov')
        return GeneralizedExtremeValue.(μ, σ, ξ)
    elseif "η₁" in output.names
        μ₀ = val[:, 1, 1]; μ₁ = val[:, 2, 1]; ϕ = val[:, 3, 1]; ξ = val[:, 4, 1]
        μ = μ₀ .+ μ₁ .* xcov'; σ = exp.(ϕ)
        return GeneralizedExtremeValue.(μ, σ, ξ)
    else
        μ = val[:, 1, 1]; ϕ = val[:, 2, 1]; ξ = val[:, 3, 1]
        σ = exp.(ϕ)
        return GeneralizedExtremeValue.(μ, σ, ξ)
    end
end
