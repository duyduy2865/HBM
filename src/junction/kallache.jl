"""
    kallache(soutput::Chains, data_layer::Array{BlockMaxima}, ooutput::Chains,
    ref::Int, mb::Int, smoy::Float64, sstd::Float64, syears::Array{Int64})

Return estimated future GEV distributions after linking the simulation and pseudo-observation models, for member `mb`.
There is one GEV distribution for each MCMC iteration and each year (for non-stationary models).

# Arguments
- `soutput::Chains`: simulation model output from `sim_bhm`
- `ooutput::Chains`: pseudo-observation model output from `obs_bhm`
- `data_layer::Array{BlockMaxima}`: data input for simulation model
- `smoy::Float64`: normalizing mean of input simulation data
- `sstd::Float64`: normalizing standard deviation of input simulation data
- `ref::Int`: reference year for Kallache transfer function
- `syears::Array{Int64}: simulation data time period

"""

function kallache(soutput::Chains, data_layer::Array{BlockMaxima}, ooutput::Chains, obs_x::Array{Float64},
    ref::Int, mb::Int, smoy::Float64, sstd::Float64, syears::Array{Int64}, hyears::Array{Int64})

    k = ref - syears[1] + 1
    h = ref - hyears[1] + 1

    obspd = getobspd(ooutput, xcov = obs_x)
    if size(obspd, 2) > 1
        obspd = obspd[:, h]
    end

    μ1 = Distributions.location.(obspd)
    σ1 = Distributions.scale.(obspd)
    ξ = Distributions.shape.(obspd)

    simpd = getsimpd(soutput, mb, data_layer, smoy, sstd)
    if size(simpd, 2)==1
        μ1 = μ1 .* ones(length(syears))'
        return GeneralizedExtremeValue.(μ1, σ1, ξ)
    end

    μ3 = Distributions.location.(simpd)
    σ3 = Distributions.scale.(simpd)

    μ2 = Distributions.location.(simpd[:, k])
    σ2 = Distributions.scale.(simpd[:, k])

    σ = σ1 .* σ3 ./ σ2
    μ = μ3 .+ σ3 ./ σ2 .* (μ1 .- μ2)
    return GeneralizedExtremeValue.(μ, σ, ξ)
end
