"""
    format_data(filename::String)

Take netCDF flow data input and return log-normal parameter matrixes and yearspan
as input for the HBM pseudo-observation model

# Arguments
- `filename::String`: path directory to the netCDF file for flow data

"""

function format_data(filename::String)

    data = ncread(filename, "Dis")
    years = year.(Date(1950, 1, 1) .+ Day.(ncread(filename, "time")))
    q2 = quantile(Normal(), 0.75)
    q1 = quantile(Normal(), 0.25)
    x2 = log.(data[16, :, :])
    x1 = log.(data[6, :, :])

    σ = (x2 .- x1) ./ (q2 - q1)
    μ = (x1 .* q2 .- x2 .* q1) ./ (q2 - q1)

    return μ, σ, years
end
