"""
    format_data(filename::String, GCM_id::String, RCM_id::String, co2data::DataFrame)

Take netCDF flow data input and return BlockMaxima (Extremes.jl) structures,
normalizing factors and yearspan as input for the HBM simulation model

# Arguments
- `filename::String`: path directory to the netCDF file for flow data
- `GCM_id::String`: ID for GCM used, a string of 3 characters
- `RCM_id::String`: ID for RCM used, a string of 2 characters
- `co2data::DataFrame`: CO2 emissions CSV file

"""

function format_data(filename::String, GCM_id::String, RCM_id::String, co2data::DataFrame)

    scenario_id = string.(ncread(filename, "scenario_id"))
    data = ncread(filename, "Dis")
    syears = year.(Date(1950, 1, 1) .+ Day.(ncread(filename, "time")))

    # Normalize output Y and covariate c before MCMC simulation for better performance

    Ys = Vector{Float64}[]
    Xs = Vector{Float64}[]
    ys = Float64[]
    years = Float64[]

    for index in 1:size(data,2)
        if string(scenario_id[38:40, index]...) == GCM_id && string(scenario_id[42:43, index]...) == RCM_id
            #Loading data and extracting annual maxima
            dis = DataFrame(Year = syears, Discharge = data[:, index])
            allowmissing!(dis)
            replace!(dis.Discharge, 1.0e20 => missing)
            dropmissing!(dis)

            if size(dis, 1) > 0
                push!(Ys, dis.Discharge)
                append!(ys, dis.Discharge)

                #Loading and normalizing covariate data
                co2 = filter(row -> row.Year in dis.Year, co2data)
                xs = co2.RCP85
                if string(scenario_id[46:47, index]...) == "R4" || string(scenario_id[46:47, index]...) == "41"
                    xs = co2.RCP45
                end
                co2all = vcat(co2.RCP85, co2.RCP45)
                xs = (xs .- mean(co2all)) ./ std(co2all)
                push!(Xs, xs)
                years = dis.Year
            end
        end
    end

    #Normalizing annual maxima
    for i in 1:length(Ys)
        Ys[i] = (Ys[i] .- mean(ys)) ./ std(ys)
    end

    model_stat = BlockMaxima[]
    model_loc = BlockMaxima[]
    model_full = BlockMaxima[]

    # Extremes.jl structures
    for (x, y) in zip(Xs, Ys)
        model = BlockMaxima(Variable("y",y))
        push!(model_stat, model)

        model = BlockMaxima(Variable("y", y), locationcov = [Variable("co2", x)])
        push!(model_loc, model)

        model = BlockMaxima(Variable("y", y), locationcov = [Variable("co2", x)], logscalecov = [Variable("co2", x)])
        push!(model_full, model)
    end
    return model_stat, model_loc, model_full, mean(ys), std(ys), years
end
