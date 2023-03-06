using BSON: @save, @load
using Plots, ArgParse, DelimitedFiles

function parseCommandLine(; dir=0)

        # initialize the settings (the description is for the help screen)
        # s = ArgParseSettings(description = "Example 1 for argparse.jl: minimal usage.")
		s = ArgParseSettings(description = "Dynamical Structure Factor code.")

        @add_arg_table! s begin
           "--Filename", "-f"         # another option, with short form
                help = "File name"
                arg_type = String
                default = "Ge00Sb00Te100T823K.cpmd"
			"--pathIn", "-i"         # another option, with short form
                 help = "Directory where input file is stored."
                 arg_type = String
                 default = pwd()
			"--pathOut", "-o"         # another option, with short form
                 help = "Directory where everything will be saved. This directory can be created before or during execution. Default is a new directory named after the filename, located where the filename is."
                 arg_type = String
                 default = "temp"
           "--n", "-N"
                    help = "number of particles"
					arg_type = Int
                    default = 300
			"--mass", "-m"
                     help = "mass"
 					arg_type = Float32
                     default = 1.0f0
           "--Lchar", "-L"
                help = "box length"
                arg_type = Float32
                default = 42.2976f0
           "--t_max", "-t"
                help = "t maximum. NO LONGER NEEDED"
                arg_type = Int64
                default = 0
			"--dt", "-s"
                 help = "time step between configurations"
                 arg_type = Float32
                 default = 10.0f0
			 "--dim", "-d"
	 			 help = "dimensions of time step between configurations"
	 			 arg_type = String
	 			 default = "ps"
			"--sparse", "-b"
				help = "Use sparse matrices? Should be faster..."
				arg_type = Int
				default = 0
			"--time_window", "-w"
				help = "List of time window of samples. Should be space separated"
				nargs = '*'
				arg_type = Int64
				default = [100, 500, 1000]
			"--knum", "-k"
				help = "List with kmin and kmax. Should be space separated"
				nargs = '*'
				arg_type = Int64
				default = [1, 10]
			"--test"
				help = "Test code"
				arg_type = Bool
				default = false
	   end

        return parse_args(s) # the result is a Dict{String,Any}
end

function init_dict(kwargs)
	if kwargs["test"] == false
		filename = kwargs["Filename"]
		path_to_file = kwargs["pathIn"] * "/" * kwargs["Filename"]
	    pathout = kwargs["pathOut"] == "temp" ? kwargs["pathIn"] * "/" * split(kwargs["Filename"],".")[1] * "/" : kwargs["pathOut"]

		dict = Dict()
	    dict[:PathIn] = path_to_file
	    dict[:PathOut] = pathout
		dict[:t_max_parsed] = kwargs["t_max"]
		dict[:partitions] = Dict([(:t_samp, []), (:samples, []), (:t_max, [])])
	    dict[:L_char] = kwargs["Lchar"]
		dict[:num_part] = kwargs["n"]
	    dict[:title] = filename
		dict[:dt] = kwargs["dt"]
		dict[:dim] = kwargs["dim"]
		dict[:t_samp_list] = kwargs["time_window"]
		dict[:knum] = kwargs["knum"]
		dict[:mass] = kwargs["mass"]
	else
		kwargs["Filename"] = "Test"
		filename = kwargs["Filename"]
		path_to_file = kwargs["pathIn"] * "/" * kwargs["Filename"]
	    pathout = kwargs["pathIn"] * "/" * kwargs["Filename"] * "/"
		dict = Dict()
	    dict[:PathIn] = path_to_file
	    dict[:PathOut] = pathout
		dict[:t_max_parsed] = kwargs["t_max"]
		dict[:partitions] = Dict([(:t_samp, []), (:samples, []), (:t_max, [])])
	    dict[:L_char] = 10 #kwargs["Lchar"]
		dict[:num_part] = 1000 #kwargs["n"]
	    dict[:title] = filename
		dict[:dt] = 0.01 #kwargs["dt"]
		dict[:dim] = kwargs["dim"]
		dict[:t_samp_list] = Array{Int64}([1] ./ dict[:dt]) #kwargs["time_window"]
		dict[:knum] = kwargs["knum"]
		dict[:mass] = kwargs["mass"]
		dict[:t_max] = 10
	end

    isdir(pathout) || mkdir(pathout)

    @save (dict[:PathOut] * "Dict.bson") dict
    return dict
end

# function init_dict(path_to_file, pathout, t_MAX, L_char, num_part, title, Δt, dim, t_samp_list, kmax, kmin, m)
#     isdir(pathout) || mkdir(pathout)
#     dict = Dict()
#     dict[:PathIn] = path_to_file
#     dict[:PathOut] = pathout
# 	dict[:t_max] = t_MAX
# 	dict[:partitions] = Dict([(:t_samp, []), (:samples, []), (:t_max, [])])
#     dict[:L_char] = L_char
# 	dict[:num_part] = num_part
#     dict[:title] = title
# 	dict[:dt] = Δt
# 	dict[:dim] = dim
# 	dict[:t_samp_list] = t_samp_list
# 	dict[:knum] = [kmin, kmax]
# 	dict[:mass] = m
#
#     @save (dict[:PathOut] * "Dict.bson") dict
#     return dict
# end

function load_dict(path)
	@load (path * "Dict.bson") dict
	return dict
end

function load_data(dict)
	t_max = dict[:t_max_parsed]
	N = dict[:num_part]
	path = dict[:PathIn]
    data1 = readdlm(path)
    x = reshape(data1[:,2],N,:)'
    y = reshape(data1[:,3],N,:)'
    z = reshape(data1[:,4],N,:)'
    vx = reshape(data1[:,5],N,:)'
    vy = reshape(data1[:,6],N,:)'
    vz = reshape(data1[:,7],N,:)'
    # samp = Int(floor(t_max/t_sample))
    # d = [(x[r,:],y[r,:],z[r,:],vx[r,:],vy[r,:],vz[r,:]) for r in Iterators.partition(1:t_max,t_sample)][1:samp]

    # t_max = t_max == 0 ? end : t_max
    d = t_max != 0 ? (x[1:t_max,:],y[1:t_max,:],z[1:t_max,:],vx[1:t_max,:],vy[1:t_max,:],vz[1:t_max,:]) : (x,y,z,vx,vy,vz)
	dict[:t_max] = size(x,1)
	@info "Data loaded. $(dict[:t_max]) configurations detected."
	d
end

function saveData(ω_list, freq_list, CT, CT_arr, dict; flag="T")
	fig_name = split(dict[:title],".")[1]
	writedlm(dict[:PathOut] * "/" * fig_name * "_modes_$(flag).dat", ω_list')
	writedlm(dict[:PathOut] * "/" * fig_name * "_freq_$(flag).dat", freq_list)

	isdir(dict[:PathOut] * "/" * fig_name * "_CorrelationDataTimeWindowsMixed") || mkdir(dict[:PathOut] * "/" * fig_name * "_CorrelationDataTimeWindowsMixed")
	for k in 1:size(CT,1)
	    writedlm(dict[:PathOut] * "/" * fig_name * "_CorrelationDataTimeWindowsMixed/C$(flag)_k_$k.dat", CT[k,:,:])
	end

	isdir(dict[:PathOut] * "/" * fig_name * "_CorrelationData") || mkdir(dict[:PathOut] * "/" * fig_name * "_CorrelationData")
	for k in 1:size(CT_arr,2), s in 1:size(CT_arr,4)
		tw = dict[:t_samp_list][s]
	    tw2 = Int(floor(tw/2))
	    writedlm(dict[:PathOut] * "/" * fig_name * "_CorrelationData/C$(flag)_k_$(k)_TW_$tw.dat", [CT_arr[1:tw2,k,2,s] CT_arr[1:tw2,k,1,s]])
	end
end

function saveData(CT_arr, dict; flag="T")
	fig_name = split(dict[:title],".")[1]
	# isdir(dict[:PathOut] * "/" * fig_name * "_CorrelationDataTimeWindowsMixed") || mkdir(dict[:PathOut] * "/" * fig_name * "_CorrelationDataTimeWindowsMixed")
	# for k in 1:size(CT,1)
	#     writedlm(dict[:PathOut] * "/" * fig_name * "_CorrelationDataTimeWindowsMixed/C$(flag)_k_$k.dat", CT[k,:,:])
	# end

	isdir(dict[:PathOut] * "/" * fig_name * "_CorrelationData") || mkdir(dict[:PathOut] * "/" * fig_name * "_CorrelationData")
	for k in 1:size(CT_arr,2), s in 1:size(CT_arr,4)
		tw = dict[:t_samp_list][s]
	    tw2 = Int(floor(tw/2))
	    writedlm(dict[:PathOut] * "/" * fig_name * "_CorrelationData/C$(flag)_k_$(k)_TW_$tw.dat", [CT_arr[1:tw2,k,2,s] CT_arr[1:tw2,k,1,s]])
	end
end

function savePlots(ω_list, freq_list, CT, CT_arr, dict; flag="T", mixedWindows=true)
	dim = dict[:dim]
	#Mean frequency over different time windows.
	m = reshape(mean(freq_list, dims=2),:)
	st = reshape(std(freq_list, dims=2),:)
	fig = plot(1:10, m, ribbon=st, fillalpha=0.2, label="mean", lw=4,
	    	frame=:box, xlabel="k (wave number)", ylabel="freq (1/$dim)", legend=:topleft,
	    	title=dict[:title], margin = 10Plots.mm, ms=5, markershapes = :circle, markerstrokewidth=0)

	# Correlation in mixed windows
	figs_CT = []
    for k in 1:size(CT,1)
		wmaxBound = minimum([200, size(CT,2)])
        f_CT = plot(CT[k,1:wmaxBound,1],CT[k,1:wmaxBound,2], frame=:box, xlabel="freq (1/$dim)", ylabel="C$(flag)", label="k = $k", margin = 5Plots.mm, ms=4, markershapes = :circle, markerstrokewidth=0, legend=:outertop)
        push!(figs_CT, f_CT)
    end

	# Max freq. Max modes. Correlation per time window.
    figs_freq = []
    figs_ω = []
    figs_CT_arr = []
    for s in 1:size(ω_list,1)
		tw = dict[:t_samp_list][s]
        f_ω = plot(ω_list[s,:], frame=:box, xlabel="k", ylabel="w max", label="Time window = $tw", margin = 5Plots.mm, ms=4, markershapes = :circle, markerstrokewidth=0, legend=:outertop)
        push!(figs_ω, f_ω)

        f_freq = plot(freq_list[:,s], frame=:box, xlabel="k", ylabel="freq max (1/$dim)", label="Time window = $tw", margin = 5Plots.mm, ms=4, markershapes = :circle, markerstrokewidth=0, legend=:outertop)
        push!(figs_freq, f_freq)

        for k in 1:size(CT_arr,2)
			tw2 = Int(floor(tw/2))
            f_CT_arr = plot(CT_arr[1:tw2,k,2,s], CT_arr[1:tw2,k,1,s], frame=:box, xlabel="freq (1/$dim)", ylabel="C$(flag)", label="k = $k, Time Window $tw", margin = 5Plots.mm, ms=4, markershapes = :circle, markerstrokewidth=0, legend=:outertop)
            push!(figs_CT_arr, f_CT_arr)
        end
    end

	fig_name = split(dict[:title],".")[1]
	savefig(fig, dict[:PathOut] * "/" * fig_name * "_freqMean.png")
	savefig(plot(figs_CT..., size=(1500,900)), dict[:PathOut] * "/" * fig_name * "_C$(flag).png")
	savefig(plot(figs_freq..., size=(1500,900)), dict[:PathOut] * "/" * fig_name * "_freq_$(flag).png")
	savefig(plot(figs_ω..., size=(1500,900)), dict[:PathOut] * "/" * fig_name * "_modes_$(flag).png")
	savefig(plot(figs_CT_arr..., size=(2500,1900)), dict[:PathOut] * "/" * fig_name * "_C$(flag)_arr.png")
end

function savePlots(CT_arr, dict; flag="T_im", mixedWindows=true)
	dim = dict[:dim]

	# # Correlation in mixed windows
	# figs_CT = []
    # for k in 1:size(CT,1)
    #     f_CT = plot(CT[k,1:200,1],CT[k,1:200,2], frame=:box, xlabel="time ($dim)", ylabel="C$(flag)", label="k = $k", margin = 5Plots.mm, ms=4, markershapes = :circle, markerstrokewidth=0)
    #     push!(figs_CT, f_CT)
    # end

	# Correlation per time window.
    figs_CT_arr = []
    for s in 1:size(CT_arr,4)
		tw = dict[:t_samp_list][s]
        # f_ω = plot(ω_list[s,:], frame=:box, xlabel="k", ylabel="w max", label="Time window = $tw", margin = 5Plots.mm, ms=4, markershapes = :circle, markerstrokewidth=0)
        # push!(figs_ω, f_ω)

        # f_freq = plot(freq_list[:,s], frame=:box, xlabel="k", ylabel="freq max (1/$dim)", label="Time window = $tw", margin = 5Plots.mm, ms=4, markershapes = :circle, markerstrokewidth=0)
        # push!(figs_freq, f_freq)

        for k in 1:size(CT_arr,2)
			tw2 = Int(floor(tw/2))
            f_CT_arr = plot(CT_arr[1:tw2,k,2,s], CT_arr[1:tw2,k,1,s], frame=:box, xlabel="time ($dim)", ylabel="C$(flag)", label="k = $k, Time Window $tw", margin = 5Plots.mm, ms=4, markershapes = :circle, markerstrokewidth=0, legend=:outertop)
            push!(figs_CT_arr, f_CT_arr)
        end
    end

	fig_name = split(dict[:title],".")[1]
	# savefig(plot(figs_CT..., size=(1500,900)), dict[:PathOut] * "/" * fig_name * "_C$(flag).png")

	savefig(plot(figs_CT_arr..., size=(2500,1900)), dict[:PathOut] * "/" * fig_name * "_C$(flag)_arr.png")
end
