using Statistics, FFTW, LinearAlgebra#, JLD
# using BSON: @save, @load
# using DelimitedFiles, ArgParse,
include("./inNout.jl")
include("./corrFunctions.jl")

mutable struct structureFactor
	ωT_list
	freqT_list
	CT_mixedWindows
	CT_arr

	ωL_list
	freqL_list
	CL_mixedWindows
	CL_arr

	ω_list
	freq_list
	C_mixedWindows
	C_arr

	CL_Im_mixedWindows
	CL_Im_arr

	CT_Im_mixedWindows
	CT_Im_arr
end

#=
#Loading Functions
=#

# function arrange_arrays(data, N)
#     #=
#     This arranges the data as vectors of x,y,z,vx,vy and vz
#     =#
#     x_indices = [2 + 6*i for i in 0:N-1]
#     y_indices = [3 + 6*i for i in 0:N-1]
#     z_indices = [4 + 6*i for i in 0:N-1]
#     vx_indices = [5 + 6*i for i in 0:N-1]
#     vy_indices = [6 + 6*i for i in 0:N-1]
#     vz_indices = [7 + 6*i for i in 0:N-1]
#
#     x, y, z = data[:,x_indices], data[:,y_indices], data[:,z_indices]
#     vx, vy, vz = data[:,vx_indices], data[:,vy_indices], data[:,vz_indices]
#     x, y, z, vx, vy, vz
# end

# function arrange_arrays(idx::Int, N)
#     pos, vel = load_data(i=idx, N=N)
#
#     x = hcat([pos[:,1+3*i] for i in 0:N-1]...)
#     y = hcat([pos[:,2+3*i] for i in 0:N-1]...)
#     z = hcat([pos[:,3+3*i] for i in 0:N-1]...)
#     vx = hcat([vel[:,1+3*i] for i in 0:N-1]...)
#     vy = hcat([vel[:,2+3*i] for i in 0:N-1]...)
#     vz = hcat([vel[:,3+3*i] for i in 0:N-1]...)
#     x, y, z, vx, vy, vz
# end

# function load_data(; i=1, path=PATH, N=num_part)
#     cd(path)
#     files = readdir(path * "/$i")
#     positions = readdlm(path * "/$i/positions.dat")
#     velocities = readdlm(path * "/$i/velocity.dat")
#     Array(reshape(positions, 3N, :)'), Array(reshape(velocities, 3N, :)')
# end


#=
Tests
=#
function test_data(;Δt = 0.01, steps = 5000, thermal_steps=1000,
			N=300, L = 1, dims=3, arrange = "Crystal", typeHessian="random")

	r, v, M = generateData(N; steps, dims, Δt, arrange, typeHessian)
	((r[1,thermal_steps+1:end,:]/L, r[2,thermal_steps+1:end,:]/L, r[3,thermal_steps+1:end,:]/L,
	v[1,thermal_steps+1:end,:], v[2,thermal_steps+1:end,:], v[3,thermal_steps+1:end,:]), M)
end

function partition_data(d, t_sample, dict)
    t_max = size(d[1],1)
    samp = Int(floor(t_max/t_sample))
    data = [(d[1][r,:],d[2][r,:],d[3][r,:],d[4][r,:],d[5][r,:],d[6][r,:]) for r in Iterators.partition(1:t_max,t_sample)][1:samp]
    @info "Data partitioned"
	append!(dict[:partitions][:t_samp], t_sample)
	append!(dict[:partitions][:samples], samp)
	append!(dict[:partitions][:t_max], samp * t_sample)
    data, samp * t_sample
end


function get_Corr_over_all_samples(d, dict; in_Fourier=false, flag="Transversal")

    data = d[1]
	N = dict[:num_part]
	t_max = dict[:partitions][:t_max][end]
	t_samp = dict[:partitions][:t_samp][end]
	samp = dict[:partitions][:samples][end]
	kmin, kmax = dict[:knum]
	L = dict[:L_char]
	m = dict[:mass]

	@info "Computing $flag part"
	if flag == "Transversal"
		@info "Sample 1"
	    CT, CTIm = getTransversalCorrelationFunction(data, N, t_samp; kmin = kmin, kmax = kmax, L = L, m=m)
	    CT_av, CTIm_av = reshape(CT,1,kmax-kmin+1,t_samp), reshape(CTIm,1,kmax-kmin+1,t_samp)
	    for i in 2:samp
	        @info "Sample $i"
	        data = d[i]
	        CT, CTIm = getTransversalCorrelationFunction(data, N, t_samp; kmin = kmin, kmax = kmax, L = L, m=m)

	        CT_av = vcat(CT_av, reshape(CT,1,kmax-kmin+1,t_samp))
	        CTIm_av = vcat(CTIm_av, reshape(CTIm,1,kmax-kmin+1,t_samp))
	    end
		return do_the_fou(CT_av, CTIm_av, t_samp; in_Fourier, L)
	elseif flag == "Longitudinal"
		@info "Sample 1"
		CL, CLIm = getLongitudinalCorrelationFunction(data, N, t_samp; kmin = kmin, kmax = kmax, L = L, m=m)
	    CL_av, CLIm_av = reshape(CL,1,kmax-kmin+1,t_samp), reshape(CLIm,1,kmax-kmin+1,t_samp)
	    for i in 2:samp
	        @info "Sample $i"
	        data = d[i]
			CL, CLIm = getLongitudinalCorrelationFunction(data, N, t_samp; kmin = kmin, kmax = kmax, L = L, m=m)

			CL_av = vcat(CL_av, reshape(CL,1,kmax-kmin+1,t_samp))
	        CLIm_av = vcat(CLIm_av, reshape(CLIm,1,kmax-kmin+1,t_samp))
	    end
		return do_the_fou(CL_av, CLIm_av, t_samp; in_Fourier, L)
	end
	# return CT_av, CTIm_av
end

function do_the_fou(CT_av, CTIm_av, tmax; kmin = 1, kmax = 10,
			L = 1, m=1, in_Fourier=true)
	if in_Fourier
        CT_av, CTIm_av = reshape(mean(CT_av,dims=1),kmax-kmin+1,tmax), reshape(mean(CTIm_av,dims=1),kmax-kmin+1,tmax)
        CT_FT = abs.(hcat(fft.([CT_av[i,:] for i in 1:kmax-kmin+1])...))
        CTIm_FT = abs.(hcat(fft.([CTIm_av[i,:] for i in 1:kmax-kmin+1])...))
        # return CT_FT, CTIm_FT
		return CT_FT, Array(transpose(CTIm_av))
    else
        return reshape(mean(CT_av,dims=1),kmax-kmin+1,tmax), reshape(mean(CTIm_av,dims=1),kmax-kmin+1,tmax)
    end
end

function get_ω_sample(data, dict, c::Bool)

    ωT_list = zeros(size(dict[:t_samp_list],1),dict[:knum][2]-dict[:knum][1] + 1)
	ωL_list = zeros(size(dict[:t_samp_list],1),dict[:knum][2]-dict[:knum][1] + 1)
	ωTim_list = zeros(size(dict[:t_samp_list],1),dict[:knum][2]-dict[:knum][1] + 1)
	ωLim_list = zeros(size(dict[:t_samp_list],1),dict[:knum][2]-dict[:knum][1] + 1)
	ω_list = zeros(size(dict[:t_samp_list],1),dict[:knum][2]-dict[:knum][1] + 1)
	CT_arr = zeros(maximum(dict[:t_samp_list]),dict[:knum][2]-dict[:knum][1] + 1,2,size(dict[:t_samp_list],1))
	CL_arr = zeros(maximum(dict[:t_samp_list]),dict[:knum][2]-dict[:knum][1] + 1,2,size(dict[:t_samp_list],1))
	CT_Im_arr = zeros(maximum(dict[:t_samp_list]),dict[:knum][2]-dict[:knum][1] + 1,2,size(dict[:t_samp_list],1))
	CL_Im_arr = zeros(maximum(dict[:t_samp_list]),dict[:knum][2]-dict[:knum][1] + 1,2,size(dict[:t_samp_list],1))
	C_arr = zeros(maximum(dict[:t_samp_list]),dict[:knum][2]-dict[:knum][1] + 1,2,size(dict[:t_samp_list],1))

	for (j, ts) in enumerate(dict[:t_samp_list])
        # ts = t_samp_list[j]
        d_part, t_m = partition_data(data, ts, dict)
        @info "$j : t_samp set to $ts timesteps. $(dict[:partitions][:samples][end]) samples for given t_samp."
        if dict[:partitions][:t_samp][end] * dict[:partitions][:samples][end] != dict[:partitions][:t_max][end]
            @warn "t_sample x n_sample does NOT equal t_m"
            break
        end

        CT, CT_Im = get_Corr_over_all_samples(d_part, dict; in_Fourier = c, flag = "Transversal")
		CL, CL_Im = get_Corr_over_all_samples(d_part, dict; in_Fourier = c, flag = "Longitudinal")
		C_total = CT .+ CL


		ωlim = Int(floor(ts/2))

		gatherDataToArrays(j, ωlim, CT, ts, dict, ωT_list, CT_arr)
		gatherDataToArrays(j, ωlim, CL, ts, dict, ωL_list, CL_arr)
		gatherDataToArrays(j, ωlim, CT_Im, ts, dict, ωTim_list, CT_Im_arr, in_fourier=false)
		gatherDataToArrays(j, ωlim, CL_Im, ts, dict, ωLim_list, CL_Im_arr, in_fourier=false)
		gatherDataToArrays(j, ωlim, C_total, ts, dict, ω_list, C_arr)

    end
    freqT_list = hcat([(ωT_list[i,:] .- 1) ./(ts * dict[:dt]) for (i,ts) in enumerate(dict[:t_samp_list])]...)
	freqL_list = hcat([(ωL_list[i,:] .- 1) ./(ts * dict[:dt]) for (i,ts) in enumerate(dict[:t_samp_list])]...)
	freq_list = hcat([(ω_list[i,:] .- 1) ./(ts * dict[:dt]) for (i,ts) in enumerate(dict[:t_samp_list])]...)

	CT_mixedWindows = getMixedWindows(CT_arr, dict)
	CL_mixedWindows = getMixedWindows(CL_arr, dict)
	# CT_Im_mixedWindows = getMixedWindows(CT_Im_arr, dict)
	# CL_Im_mixedWindows = getMixedWindows(CL_Im_arr, dict)
	C_mixedWindows = getMixedWindows(C_arr, dict)

	# ωT_list, freqT_list, CT_mixedWindows, CT_arr, ωL_list, freqL_list, CL_mixedWindows, CL_arr
	structureFactor(ωT_list, freqT_list, CT_mixedWindows, CT_arr, ωL_list, freqL_list, CL_mixedWindows, CL_arr, ω_list,	freq_list, C_mixedWindows, C_arr, undef, CL_Im_arr, undef, CT_Im_arr)
end

function gatherDataToArrays(j, ωlim, C, ts, dict, ω_list, C_arr; in_fourier=true)
	if in_fourier
		ω_list[j,:] = vcat([findall(x->x == maximum(C[1:ωlim,i]), C[1:ωlim,i]) for i in 1:size(C,2)]...)
		freq = hcat([[l for l in 0:size(C,1)-1] ./ (ts * dict[:dt]) for i in 1:size(C,2)]...)
		C_arr[1:size(C,1),:,:,j] = cat(C,freq, dims=3)
	else
		C_arr[1:size(C,1),:,:,j] = cat(C, hcat([dict[:dt] .* collect(1:size(C,1)) for i in 1:size(C,2)]...), dims=3)
	end
end

function getMixedWindows(C_arr, dict)
	C_mixedWindows = arrange_Corr(C_arr; len=Int(floor(minimum(dict[:t_samp_list])/2)), Δk = dict[:knum][2]-dict[:knum][1] + 1)
	C_mixedWindows = envelope.([C_mixedWindows[i,:,:] for i in 1:dict[:knum][2]-dict[:knum][1] + 1])
	C_mixedWindows = permutedims(cat(C_mixedWindows...,dims=3),(3,1,2))
	C_mixedWindows
end

function arrange_Corr(CT_arr; len=30, Δk=10)
        t_len=size(CT_arr,4)
        l = [i*len for i in 1:t_len]
        new_CT = zeros(Δk,sum(l),2)
        for k in 1:Δk
                x = vcat([CT_arr[1:i*len,k,2,i] for i in 1:t_len]...)
                y = vcat([CT_arr[1:i*len,k,1,i] for i in 1:t_len]...)
                idx = sortperm(x)
                new_CT[k,:,:] = hcat([x[idx],y[idx]]...)
        end
        new_CT
end

function envelope(ar)
    arr = ar[:,1]
    arr2 = ar[:,2]

    l = zeros(size(arr))
    ind = 1
    for i in 1:size(l,1)
        if ind <= size(arr2,1)
            ind = findall(x->x == maximum(arr2[arr[ind] .== arr]), arr2)[1]
            l[i] = ind
            ind = maximum(indexin(arr2[arr[ind] .== arr], arr2)) + 1
        end
    end
    ll = Array{Int,1}(l[l .>0])
    ar[ll,:]
end

function get_freq_max(CT_arr)
        freq_max = zeros(10,2)
        for k in 1:10
                idx = sortperm(CT_arr[k,:,2])[end]
                freq_max[k,:] = [k, CT_arr[k,idx,1]]
        end
        freq_max
end

function get_freqs(dict; test=false, ω = 0.0002, Δt = 10.0, δ = 0.5)
    if test
		t_MAX = dict[:t_max]
	    L_char = dict[:L_char]
	    num_part = dict[:num_part]
		t_samp_list = dict[:t_samp_list]
        d, M = test_data(steps = t_MAX + 2000, thermal_steps=2000, L = L_char,
				Δt = dict[:dt], N=num_part, arrange = "Crystal", typeHessian="random")
    else
        d = load_data(dict)
    end

    get_ω_sample(d, dict, true)
end

function main(dict)
	# dim = dict[:dim]
	@info "Starting..."
	# ω_list, freq_list, CT_mixedWindows, CT_arr, ωL_list, freqL_list, CL_mixedWindows, CL_arr = get_freqs(dict)
	s = get_freqs(dict)
	@info "Computation done"

	@info "Generating Plots..."
	savePlots(s.ωT_list, s.freqT_list, s.CT_mixedWindows, s.CT_arr, dict, flag="T")
	savePlots(s.ωL_list, s.freqL_list, s.CL_mixedWindows, s.CL_arr, dict, flag="L")
	savePlots(s.ω_list, s.freq_list, s.C_mixedWindows, s.C_arr, dict, flag="total")
	savePlots(s.CL_Im_arr, dict, flag="L_im")
	savePlots(s.CT_Im_arr, dict, flag="T_im")

	@info "Saving Data..."
	saveData(s.ωT_list, s.freqT_list, s.CT_mixedWindows, s.CT_arr, dict, flag="T")
	saveData(s.ωL_list, s.freqL_list, s.CL_mixedWindows, s.CL_arr, dict, flag="L")
	saveData(s.ω_list, s.freq_list, s.C_mixedWindows, s.C_arr, dict, flag="total")
	saveData(s.CL_Im_arr, dict, flag="L_im")
	saveData(s.CT_Im_arr, dict, flag="T_im")
	@info "Done!"
	writedlm(dict[:PathOut] * "/" * split(dict[:title],".")[1] * "_freq_mixedWindows_T.dat",
			get_freq_max(s.CT_mixedWindows))
	writedlm(dict[:PathOut] * "/" * split(dict[:title],".")[1] * "_freq_mixedWindows_L.dat",
			get_freq_max(s.CL_mixedWindows))
	writedlm(dict[:PathOut] * "/" * split(dict[:title],".")[1] * "_freq_mixedWindows_total.dat",
			get_freq_max(s.C_mixedWindows))
end
