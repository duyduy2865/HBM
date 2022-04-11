module HBM

using Statistics, DataFrames, Extremes, Distributions, Mamba
using Dates, StatsBase, NetCDF, LinearAlgebra, Random, ProgressMeter

include("utils.jl")
include("simulations/dataformatting.jl")
include("simulations/modelSelection.jl")
include("simulations/sim_model.jl")
include("simulations/modelValidation.jl")
include("pseudo-observations/modelValidation.jl")
include("pseudo-observations/modelSelection.jl")
include("pseudo-observations/pseudo_model.jl")
include("pseudo-observations/dataformat.jl")
include("junction/kallache.jl")
include("junction/utils.jl")

export format_data, format_data2
export sim_bhm, obs_bhm, obs_bhm_bis
export qqplot_map, qqplot_mean
export getsimpd, getobspd, kallache
export getDIC3_obs, getDIC4_obs, getDIC3, getDIC4, getDIC5

end
