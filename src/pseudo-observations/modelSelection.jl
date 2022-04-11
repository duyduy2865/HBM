function getDIC3_obs(output::Chains, μh::Matrix{Float64}, σh::Matrix{Float64})

    val = output.value
    S = size(μh, 2)
    μ = val[:, 1, 1]
    ϕ = val[:, 2, 1]
    ξ = val[:, 3, 1]
    y = val[:, 4:end, 1]

    μ̄ = mean(μ)
    ϕ̄ = mean(ϕ)
    ξ̄ = mean(ξ)
    mean_y = vec(mean(y, dims = 1))
    dic1 = sum(logpdf.(LogNormal.(μh, σh), mean_y)) + sum(logpdf.(GeneralizedExtremeValue(μ̄, exp(ϕ̄), ξ̄), mean_y))

    dic2 = 0.

    for s in 1:S
        d = mean(logpdf.(LogNormal.(μh[:, s], σh[:, s]), y'), dims = 2)
        dic2 += sum(d)
    end

    d = mean(logpdf.(GeneralizedExtremeValue.(μ, exp.(ϕ), ξ), y), dims = 1)
    dic2 +=  sum(d)
    return -2 * dic2 + dic1
end

function getDIC4_obs(output::Chains, μh::Matrix{Float64}, σh::Matrix{Float64}, xcov::Array{Float64})

    val = output.value
    S = size(μh, 2)
    μ₀ = val[:, 1, 1]
    μ₁ = val[:, 2, 1]
    ϕ = val[:, 3, 1]
    ξ = val[:, 4, 1]
    y = val[:, 5:end, 1]

    μ̄₀ = mean(μ₀)
    μ̄₁ = mean(μ₁)
    ϕ̄ = mean(ϕ)
    ξ̄ = mean(ξ)
    mean_y = vec(mean(y, dims = 1))

    dic1 = sum(logpdf.(LogNormal.(μh, σh), mean_y)) +
    sum(logpdf.(GeneralizedExtremeValue.(μ̄₀ .+ μ̄₁ .* xcov, exp(ϕ̄), ξ̄), mean_y))

    dic2 = 0.

    for s in 1:S
        d = mean(logpdf.(LogNormal.(μh[:, s], σh[:, s]), y'), dims = 2)
        dic2 += sum(d)
    end

    d = mean(logpdf.(GeneralizedExtremeValue.(μ₀ .+ μ₁ .* xcov', exp.(ϕ), ξ), y), dims = 1)
    dic2 +=  sum(d)
    return -2 * dic2 + dic1
end
