function invgamma_sampling(y::AbstractVector{<:Real}, α::Real=0.01, β::Real=0.01)

    n = length(y)
    s² = (n-1) * var(y)
    m = mean(y)

    pd = InverseGamma(n/2+α, s²/2 + β)
    σ² = rand(pd)
    σ = sqrt(σ²)

    std = σ/sqrt(n)
    pd = Normal(m, std)
    μ = rand(pd)

    return μ, σ

end

function update_stepsize(δ::Real, accrate::Real)
    Δδ = 0.01 * (2 * (accrate > 0.44) - 1)
    return δ * exp(Δδ)
end

# To be used if informative prior is necessary
# μ ∼ Normal(ν, τ), σ ∼ InvGamma(α, β)
function invgamma_normal_sampling(y::AbstractVector{<:Real}, ν::Float64=0., τ::Float64=1e4, α::Float64=0.01, β::Float64=0.01)

    n = length(y)
    s² = (n-1) * var(y)
    m = mean(y)

    pd = InverseGamma(n/2+α, s²/2 + β)
    σ² = rand(pd)
    σ = sqrt(σ²)

    m̄ = (τ^2 * n * m + σ² * ν) / (n * τ^2 + σ²)
    std = sqrt( σ² * τ^2 / ( σ² + n * τ^2 ) )
    pd = Normal(m̄, std)
    μ = rand(pd)

    return μ, σ

end
