function getDIC3(output::Chains, data::Array{BlockMaxima})

    val = output.value
    m = length(data)

    ν = val[:, 1:3, 1]
    τ = val[:, 4:6, 1]
    μ = val[:, 6+1:6+m, 1]
    ϕ = val[:, 6+m+1:6+2*m, 1]
    ξ = val[:, 6+2*m+1:6+3*m, 1]

    mean_μ = mean(μ, dims = 1)
    mean_ϕ = mean(ϕ, dims = 1)
    mean_ξ = mean(ξ, dims = 1)
    mean_ν = vec(mean(ν, dims = 1))
    mean_τ = vec(mean(τ, dims = 1))

    mean_params = Extremes.slicematrix(vcat(mean_μ, mean_ϕ, mean_ξ))
    dic1 = sum(Extremes.loglike.(data, mean_params)) + sum([logpdf(MvNormal(mean_ν, diagm(mean_τ)), params) for params in mean_params])
    dic2 = 0.

    ν2 = Extremes.slicematrix(ν, dims = 2)
    τ2 = Extremes.slicematrix(τ, dims = 2)

    for i in 1:m
        e = hcat(μ[:, i], ϕ[:, i], ξ[:, i])
        params = Extremes.slicematrix(e, dims = 2)
        d = mean([Extremes.loglike(data[i], p) for p in params])
        d2 = mean([logpdf(MvNormal(ν2[p], diagm(vec(τ2[p]))), params[p]) for p in 1:length(params)])
        dic2 += (d+d2)
    end

    return -2 * dic2 + dic1
end

function getDIC4(output::Chains, data::Array{BlockMaxima})

    val = output.value
    m = length(data)

    ν = val[:, 1:4, 1]
    τ = val[:, 5:8, 1]
    μ₀ = val[:, 8+1:8+m, 1]
    μ₁ = val[:, 8+m+1:8+2*m, 1]
    ϕ = val[:, 8+2*m+1:8+3*m, 1]
    ξ = val[:, 8+3*m+1:8+4*m, 1]

    mean_ν = vec(mean(ν, dims = 1))
    mean_τ = vec(mean(τ, dims = 1))
    mean_μ₀ = mean(μ₀, dims = 1)
    mean_μ₁ = mean(μ₁, dims = 1)
    mean_ϕ = mean(ϕ, dims = 1)
    mean_ξ = mean(ξ, dims = 1)

    mean_params = Extremes.slicematrix(vcat(mean_μ₀, mean_μ₁, mean_ϕ, mean_ξ))
    dic1 = sum(Extremes.loglike.(data, mean_params)) + sum([logpdf(MvNormal(mean_ν, diagm(mean_τ)), params) for params in mean_params])
    dic2 = 0.

    ν2 = Extremes.slicematrix(ν, dims = 2)
    τ2 = Extremes.slicematrix(τ, dims = 2)

    for i in 1:m
        e = hcat(μ₀[:, i], μ₁[:, i], ϕ[:, i], ξ[:, i])
        params = Extremes.slicematrix(e, dims = 2)
        d = mean([Extremes.loglike(data[i], p) for p in params])
        d2 = mean([logpdf(MvNormal(ν2[p], diagm(vec(τ2[p]))), params[p]) for p in 1:length(params)])
        dic2 += (d+d2)
    end

    return -2 * dic2 + dic1
end

function getDIC5(output::Chains, data::Array{BlockMaxima})

    val = output.value
    m = length(data)

    ν = val[:, 1:5, 1]
    τ = val[:, 6:10, 1]
    μ₀ = val[:, 10+1:10+m, 1]
    μ₁ = val[:, 10+m+1:10+2*m, 1]
    ϕ₀ = val[:, 10+2*m+1:10+3*m, 1]
    ϕ₁ = val[:, 10+3*m+1:10+4*m, 1]
    ξ = val[:, 10+4*m+1:10+5*m, 1]

    mean_ν = vec(mean(ν, dims = 1))
    mean_τ = vec(mean(τ, dims = 1))
    mean_μ₀ = mean(μ₀, dims = 1)
    mean_μ₁ = mean(μ₁, dims = 1)
    mean_ϕ₀ = mean(ϕ₀, dims = 1)
    mean_ϕ₁ = mean(ϕ₁, dims = 1)
    mean_ξ = mean(ξ, dims = 1)

    mean_params = Extremes.slicematrix(vcat(mean_μ₀, mean_μ₁, mean_ϕ₀, mean_ϕ₁, mean_ξ))
    dic1 = sum(Extremes.loglike.(data, mean_params)) +
    sum([logpdf(MvNormal(mean_ν, diagm(mean_τ)), params) for params in mean_params])
    ν2 = Extremes.slicematrix(ν, dims = 2)
    τ2 = Extremes.slicematrix(τ, dims = 2)

    dic2 = 0.

    for i in 1:m
        e = hcat(μ₀[:, i], μ₁[:, i], ϕ₀[:, i], ϕ₁[:, i], ξ[:, i])
        params = Extremes.slicematrix(e, dims = 2)
        d = mean([Extremes.loglike(data[i], p) for p in params])
        d2 = mean([logpdf(MvNormal(ν2[p], diagm(vec(τ2[p]))), params[p]) for p in 1:length(params)])
        dic2 += (d+d2)
    end

    return -2 * dic2 + dic1
end
