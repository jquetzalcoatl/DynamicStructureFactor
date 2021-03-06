using Plots, Statistics, FFTW, SparseArrays, DelimitedFiles, ArgParse, LinearAlgebra

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

function parseCommandLine(; dir=0)

        # initialize the settings (the description is for the help screen)
        s = ArgParseSettings(description = "Example 1 for argparse.jl: minimal usage.")

        @add_arg_table! s begin
           "--Filename", "-f"         # another option, with short form
                help = "path to file"
                arg_type = String
                default = "Ge00Sb00Te100T823K.cpmd"
			"--pathOut", "-o"         # another option, with short form
                 help = "pathOut"
                 arg_type = String
                 default = pwd()
           "--n", "-N"
                    help = "number of particles"
					arg_type = Int
                    default = 300
           "--Lchar", "-L"
                help = "box length"
                arg_type = Float32
                default = 42.2976f0
           "--t_max", "-t"
                help = "t maximum"
                arg_type = Int64
                default = 15000
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
	   end

        return parse_args(s) # the result is a Dict{String,Any}
end

function init_dict(path_to_file, pathout, t_MAX, L_char, num_part, title, ??t, dim)
    # dict_path = PATH_out * "Dicts"
    # isdir(dict_path) || mkdir(dict_path)
    # dict_path = PATH_out * "Dicts/$(parsed_args["n"])"
    # isdir(dict_path) || mkdir(dict_path)
    dict = Dict()
    dict[:PathIn] = path_to_file
    dict[:PathOut] = pathout == 0 ? pwd() : pathout
	dict[:t_max] = t_MAX
    dict[:L_char] = L_char
	dict[:num_part] = num_part
    dict[:title] = title
	dict[:dt] = ??t
	dict[:dim] = dim

    # @save dict[:Path] dict
    return dict
end

function load_data(N, t_max; path = PATH_TO_FILE)
    data1 = readdlm(PATH_TO_FILE)
    x = reshape(data1[:,2],N,:)'
    y = reshape(data1[:,3],N,:)'
    z = reshape(data1[:,4],N,:)'
    vx = reshape(data1[:,5],N,:)'
    vy = reshape(data1[:,6],N,:)'
    vz = reshape(data1[:,7],N,:)'
    # samp = Int(floor(t_max/t_sample))
    # d = [(x[r,:],y[r,:],z[r,:],vx[r,:],vy[r,:],vz[r,:]) for r in Iterators.partition(1:t_max,t_sample)][1:samp]
    @info "Data loaded"
    # t_max = t_max == 0 ? end : t_max
    d = t_max != 0 ? (x[1:t_max,:],y[1:t_max,:],z[1:t_max,:],vx[1:t_max,:],vy[1:t_max,:],vz[1:t_max,:]) : (x,y,z,vx,vy,vz)
end

function partition_data(d, t_sample)
    t_max = size(d[1],1)
    samp = Int(floor(t_max/t_sample))
    data = [(d[1][r,:],d[2][r,:],d[3][r,:],d[4][r,:],d[5][r,:],d[6][r,:]) for r in Iterators.partition(1:t_max,t_sample)][1:samp]
    @info "Data partitioned"
    data, samp * t_sample
end
#=
Tests
=#
function test_data(s, ;?? = 0.01, ??t = 0.1,
        ?? = 0.0, N=300, amp = 1.0, L = 1)
    tup = s*??t
    tr = [amp*cos(2??*??*t) + ??*randn()  for t in 0:??t:tup]
    # vel = [-??*amp*sin(??*t) + ??*randn()  for t in 0:??t:tup]
    x = hcat([i .+ tr for i in 1:N]...)
    y = hcat([i .+ tr for i in 1:N]...)
    z = hcat([i .+ tr for i in 1:N]...)

    xL = (x .- minimum(x)) ./ ((maximum(x) .- minimum(x))) .* L
    zL = (z .- minimum(z)) ./ ((maximum(z) .- minimum(z))) .* L
    yL = (y .- minimum(y)) ./ ((maximum(y) .- minimum(y))) .* L

    # vx = hcat([vel for i in 1:N]...)
    # vy = hcat([vel for i in 1:N]...)
    # vz = hcat([vel for i in 1:N]...)
    vx = (xL[2:end,:] .- xL[1:end-1,:])/??t
    vz = (zL[2:end,:] .- zL[1:end-1,:])/??t
    vy = (yL[2:end,:] .- yL[1:end-1,:])/??t

    xL[1:end-1,:],yL[1:end-1,:],zL[1:end-1,:],vx,vy,vz
end
#=
#Correlation Functions
=#

function gen_Jays(data, N; kmin = 1, kmax = 10, L = 1, m=1)
    #=
    Generates J???(k,t) = m/N * ?? v??????(t) exp(-???? k ??? r???(t))
    As a matter of sanity check, if data = rand(tmax, 6N+1) where
    mean(rand()) = 1/2, then J???(0,t) = m/2
    =#
    x, y, z, vx, vy, vz = data
    #Transversal to z
    Jxkz = hcat([m/???N*sum(vx .* cos.(2?? * k .* z / L), dims=2) for k in kmin:kmax]...)
    JxkzIm = hcat([m/???N*sum(vx .* sin.(2?? * k .* z / L), dims=2) for k in kmin:kmax]...)

    Jykz = hcat([m/???N*sum(vy .* cos.(2?? * k .* z / L), dims=2) for k in kmin:kmax]...)
    JykzIm = hcat([m/???N*sum(vy .* sin.(2?? * k .* z / L), dims=2) for k in kmin:kmax]...)
    #Transversal to x
    Jzkx = hcat([m/???N*sum(vz .* cos.(2?? * k .* x / L), dims=2) for k in kmin:kmax]...)
    JzkxIm = hcat([m/???N*sum(vz .* sin.(2?? * k .* x / L), dims=2) for k in kmin:kmax]...)

    Jykx = hcat([m/???N*sum(vy .* cos.(2?? * k .* x / L), dims=2) for k in kmin:kmax]...)
    JykxIm = hcat([m/???N*sum(vy .* sin.(2?? * k .* x / L), dims=2) for k in kmin:kmax]...)
    #Transversal to y
    Jxky = hcat([m/???N*sum(vx .* cos.(2?? * k .* y / L), dims=2) for k in kmin:kmax]...)
    JxkyIm = hcat([m/???N*sum(vx .* sin.(2?? * k .* y / L), dims=2) for k in kmin:kmax]...)

    Jzky = hcat([m/???N*sum(vz .* cos.(2?? * k .* y / L), dims=2) for k in kmin:kmax]...)
    JzkyIm = hcat([m/???N*sum(vz .* sin.(2?? * k .* y / L), dims=2) for k in kmin:kmax]...)

    Jxkz, Jykz, Jxky, Jzky, Jykx, Jzkx, JxkzIm, JykzIm, JxkyIm, JzkyIm, JykxIm, JzkxIm
end

function gen_JiJi(data, N, tmax; kmin = 1, kmax = 10, L = 1, m=1)
    #=
    Generates ??? JxJx ???(k,??) =  ????? J???(k,t+??)J???(k,t)/Z where Z = tmax-??
    As a matter of sanity check, if data = rand(tmax, 6N+1) where
    mean(rand()) = 1/2, then ??? JxJx ???(k,??) = m??/4
    =#
    # Jxkz, Jykz, Jxky, Jzky, Jykx, Jzkx = gen_Jays(data, N; kmin = kmin, kmax = kmax, L = L, m=m)
    Jxkz, Jykz, Jxky, Jzky, Jykx, Jzkx, JxkzIm, JykzIm, JxkyIm, JzkyIm,
        JykxIm, JzkxIm = gen_Jays(data, N; kmin = kmin, kmax = kmax, L = L, m=m)

    JxJxkz = hcat([sum([Jxkz[??+t,:] .* Jxkz[t,:] .+ JxkzIm[??+t,:] .* JxkzIm[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)
    JxJxky = hcat([sum([Jxky[??+t,:] .* Jxky[t,:] .+ JxkyIm[??+t,:] .* JxkyIm[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)

    JxJxkzIm = hcat([sum([Jxkz[??+t,:] .* JxkzIm[t,:] .+ JxkzIm[??+t,:] .* Jxkz[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)
    JxJxkyIm = hcat([sum([Jxky[??+t,:] .* JxkyIm[t,:] .+ JxkyIm[??+t,:] .* Jxky[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)

    JyJykz = hcat([sum([Jykz[??+t,:] .* Jykz[t,:] .+ JykzIm[??+t,:] .* JykzIm[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)
    JyJykx = hcat([sum([Jykx[??+t,:] .* Jykx[t,:] .+ JykxIm[??+t,:] .* JykxIm[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)

    JyJykzIm = hcat([sum([Jykz[??+t,:] .* JykzIm[t,:] .+ JykzIm[??+t,:] .* Jykz[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)
    JyJykxIm = hcat([sum([Jykx[??+t,:] .* JykxIm[t,:] .+ JykxIm[??+t,:] .* Jykx[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)

    JzJzkx = hcat([sum([Jzkx[??+t,:] .* Jzkx[t,:] .+ JzkxIm[??+t,:] .* JzkxIm[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)
    JzJzky = hcat([sum([Jzky[??+t,:] .* Jzky[t,:] .+ JzkyIm[??+t,:] .* JzkyIm[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)

    JzJzkxIm = hcat([sum([Jzkx[??+t,:] .* JzkxIm[t,:] .+ JzkxIm[??+t,:] .* Jzkx[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)
    JzJzkyIm = hcat([sum([Jzky[??+t,:] .* JzkyIm[t,:] .+ JzkyIm[??+t,:] .* Jzky[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)

    JxJxkz, JxJxky, JyJykz, JyJykx, JzJzkx, JzJzky, JxJxkzIm, JxJxkyIm, JyJykzIm, JyJykxIm, JzJzkxIm, JzJzkyIm
end

function gen_JiJi(data, N, tmax, ??::Int; kmin = 1, kmax = 10, L = 1, m=1)
    #=
    Generates ??? JxJx ???(k,??) =  ????? J???(k,t+??)J???(k,t)/Z where Z = tmax-??
    As a matter of sanity check, if data = rand(tmax, 6N+1) where
    mean(rand()) = 1/2, then ??? JxJx ???(k,??) = m??/4
    =#
    # Jxkz, Jykz, Jxky, Jzky, Jykx, Jzkx = gen_Jays(data, N; kmin = kmin, kmax = kmax, L = L, m=m)
    Jxkz, Jykz, Jxky, Jzky, Jykx, Jzkx, JxkzIm, JykzIm, JxkyIm, JzkyIm,
        JykxIm, JzkxIm = gen_Jays(data, N; kmin = kmin, kmax = kmax, L = L, m=m)

	JxJxkz = hcat([diag((Jxkz' * sparse_with_diag(i,tmax) * Jxkz .+ JxkzIm' * sparse_with_diag(i,tmax) * JxkzIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JxJxkz = hcat([sum([Jxkz[??+t,:] .* Jxkz[t,:] .+ JxkzIm[??+t,:] .* JxkzIm[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)
	JxJxky = hcat([diag((Jxky' * sparse_with_diag(i,tmax) * Jxky .+ JxkyIm' * sparse_with_diag(i,tmax) * JxkyIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JxJxky = hcat([sum([Jxky[??+t,:] .* Jxky[t,:] .+ JxkyIm[??+t,:] .* JxkyIm[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)

	JxJxkzIm = hcat([diag((Jxkz' * sparse_with_diag(i,tmax) * JxkzIm .+ JxkzIm' * sparse_with_diag(i,tmax) * Jxkz)/(tmax-i)) for i in 0:tmax-1]...)
    # JxJxkzIm = hcat([sum([Jxkz[??+t,:] .* JxkzIm[t,:] .+ JxkzIm[??+t,:] .* Jxkz[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)
	JxJxkyIm = hcat([diag((Jxky' * sparse_with_diag(i,tmax) * JxkyIm .+ JxkyIm' * sparse_with_diag(i,tmax) * Jxky)/(tmax-i)) for i in 0:tmax-1]...)
    # JxJxkyIm = hcat([sum([Jxky[??+t,:] .* JxkyIm[t,:] .+ JxkyIm[??+t,:] .* Jxky[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)

	JyJykz = hcat([diag((Jykz' * sparse_with_diag(i,tmax) * Jykz .+ JykzIm' * sparse_with_diag(i,tmax) * JykzIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JyJykz = hcat([sum([Jykz[??+t,:] .* Jykz[t,:] .+ JykzIm[??+t,:] .* JykzIm[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)
	JyJykx = hcat([diag((Jykx' * sparse_with_diag(i,tmax) * Jykx .+ JykxIm' * sparse_with_diag(i,tmax) * JykxIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JyJykx = hcat([sum([Jykx[??+t,:] .* Jykx[t,:] .+ JykxIm[??+t,:] .* JykxIm[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)

	JyJykzIm = hcat([diag((Jykz' * sparse_with_diag(i,tmax) * JykzIm .+ JykzIm' * sparse_with_diag(i,tmax) * Jykz)/(tmax-i)) for i in 0:tmax-1]...)
    # JyJykzIm = hcat([sum([Jykz[??+t,:] .* JykzIm[t,:] .+ JykzIm[??+t,:] .* Jykz[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)
	JyJykxIm = hcat([diag((Jykx' * sparse_with_diag(i,tmax) * JykxIm .+ JykxIm' * sparse_with_diag(i,tmax) * Jykx)/(tmax-i)) for i in 0:tmax-1]...)
    # JyJykxIm = hcat([sum([Jykx[??+t,:] .* JykxIm[t,:] .+ JykxIm[??+t,:] .* Jykx[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)

	JzJzkx = hcat([diag((Jzkx' * sparse_with_diag(i,tmax) * Jzkx .+ JzkxIm' * sparse_with_diag(i,tmax) * JzkxIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JzJzkx = hcat([sum([Jzkx[??+t,:] .* Jzkx[t,:] .+ JzkxIm[??+t,:] .* JzkxIm[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)
	JzJzky = hcat([diag((Jzky' * sparse_with_diag(i,tmax) * Jzky .+ JzkyIm' * sparse_with_diag(i,tmax) * JzkyIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JzJzky = hcat([sum([Jzky[??+t,:] .* Jzky[t,:] .+ JzkyIm[??+t,:] .* JzkyIm[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)

	JzJzkxIm = hcat([diag((Jzkx' * sparse_with_diag(i,tmax) * JzkxIm .+ JzkxIm' * sparse_with_diag(i,tmax) * Jzkx)/(tmax-i)) for i in 0:tmax-1]...)
    # JzJzkxIm = hcat([sum([Jzkx[??+t,:] .* JzkxIm[t,:] .+ JzkxIm[??+t,:] .* Jzkx[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)
	JzJzkyIm = hcat([diag((Jzky' * sparse_with_diag(i,tmax) * JzkyIm .+ JzkyIm' * sparse_with_diag(i,tmax) * Jzky)/(tmax-i)) for i in 0:tmax-1]...)
    # JzJzkyIm = hcat([sum([Jzky[??+t,:] .* JzkyIm[t,:] .+ JzkyIm[??+t,:] .* Jzky[t,:] for t in 1:tmax-??])/(tmax-??) for ?? in 0:tmax-1]...)

    JxJxkz, JxJxky, JyJykz, JyJykx, JzJzkx, JzJzky, JxJxkzIm, JxJxkyIm, JyJykzIm, JyJykxIm, JzJzkxIm, JzJzkyIm
end

function sparse_with_diag(idx, tmax)
    A = blockdiag(sparse([i for i in 1+idx:tmax],[i for i in 1:tmax-idx],[1 for i in 1:tmax-idx]), sparse(0I, 0, idx))
    A
end

function get_Corr(data, N, tmax; kmin = 1, kmax = 10, L = 1, m=1)
	if use_sparse == 0
    	CT = gen_JiJi(data, N, tmax; kmin = kmin, kmax = kmax, L = L, m=m)
		return mean(CT[1:6]), mean(CT[7:12])
	elseif use_sparse == 1
		CT = gen_JiJi(data, N, tmax,1; kmin = kmin, kmax = kmax, L = L, m=m)
		return  mean(CT[1:6]), mean(CT[7:12])
	end
end

function get_Corr_over_all_samples(t_samp, N, t_max, d; kmin = 1, kmax = 10,
		L = 1, m=1, in_Fourier=false)
    tmax = t_samp
    samp = Int(floor(t_max/t_samp))
    # d = load_data(N,t_samp,t_max)
    data = d[1]
    CT, CTIm = get_Corr(data, N, tmax; kmin = kmin, kmax = kmax, L = L, m=m)
    CT_av, CTIm_av = reshape(CT,1,kmax-kmin+1,tmax), reshape(CTIm,1,kmax-kmin+1,tmax)
    for i in 2:samp
        @info "Sample $i"
        data = d[i]
        CT, CTIm = get_Corr(data, N, tmax; kmin = kmin, kmax = kmax, L = L, m=m)

        CT_av = vcat(CT_av, reshape(CT,1,kmax-kmin+1,tmax))
        CTIm_av = vcat(CTIm_av, reshape(CTIm,1,kmax-kmin+1,tmax))
    end

    if in_Fourier
        CT_av, CTIm_av = reshape(mean(CT_av,dims=1),kmax-kmin+1,tmax), reshape(mean(CTIm_av,dims=1),kmax-kmin+1,tmax)
        CT_FT2 = abs.(hcat(fft.([CT_av[i,:] for i in 1:kmax-kmin+1])...))
        CTIm_FT = abs.(hcat(fft.([CTIm_av[i,:] for i in 1:kmax-kmin+1])...))
        return CT_FT2, CTIm_FT
    else
        return reshape(mean(CT_av,dims=1),kmax-kmin+1,tmax), reshape(mean(CTIm_av,dims=1),kmax-kmin+1,tmax)
    end
end

function get_Corr_over_all_samples_in_fourier(t_samp, N, t_max, d; kmin = 1, kmax = 10, L = 1, m=1)
    tmax = t_samp
    samp = Int(floor(t_max/t_samp))
    # d = load_data(N,t_samp,t_max)
    data = d[1]
    CT, CTIm = get_Corr(data, N, tmax; kmin = kmin, kmax = kmax, L = L, m=m)
    CT_av, CTIm_av = reshape(CT,1,kmax-kmin+1,tmax), reshape(CTIm,1,kmax-kmin+1,tmax)
    for i in 2:samp
        @info "Sample $i"
        data = d[i]
        CT, CTIm = get_Corr(data, N, tmax; kmin = kmin, kmax = kmax, L = L, m=m)

        CT_av = vcat(CT_av, reshape(CT,1,kmax-kmin+1,tmax))
        CTIm_av = vcat(CTIm_av, reshape(CTIm,1,kmax-kmin+1,tmax))
    end
	apply_fourier(CT_av, samp, kmax-kmin+1), apply_fourier(CTIm_av, samp, kmax-kmin+1)
end

function apply_fourier(corr, samp, knum)
    ff = fft.([corr[i,j,:] for i in 1:samp, j in 1:knum])
    ff_ar = cat([hcat(ff[i,:]...) for i in 1:samp]..., dims=3)
    ff_ar_m = reshape(mean(ff_ar, dims=3),:,knum)
    fp = abs.(ff_ar_m)

    fp
end

function get_??_sample(N, t_max, data, t_samp_list; kmin = 1, kmax = 10, L = 1,
        m=1, ??s=10) #??lim = 30 data_samp_freq=10
	data_samp_period=dict[:dt]
    tsamp = Int(floor(t_max/??s))
    t_samp_list = t_samp_list == 0 ? [i*tsamp for i in 1:??s] : t_samp_list
    n_samp = Int.(floor.(t_max ./ t_samp_list))
    ??_list = zeros(size(t_samp_list,1),kmax-kmin + 1)
	CT_arr = zeros(maximum(t_samp_list),kmax-kmin + 1,2,size(t_samp_list,1))

    for j in 1:size(t_samp_list,1)
        ts = t_samp_list[j]
        d_part, t_m = partition_data(data, ts)
        @info "$j : t_samp set to $ts"
        if ts * n_samp[j] != t_m
            @warn "t_sample x n_sample does NOT equal t_m"
            break
        end
        ??lim = Int(floor(ts/2))
        CT, _ = get_Corr_over_all_samples_in_fourier(ts, N, t_max, d_part; kmin = kmin, kmax = kmax, L = L, m=m)
        ??_list[j,:] = vcat([findall(x->x == maximum(CT[1:??lim,i]), CT[1:??lim,i]) for i in 1:size(CT,2)]...)

		freq = hcat([[l for l in 0:size(CT,1)-1] ./ (ts * data_samp_period) for i in 1:size(CT,2)]...)
		CT_arr[1:size(CT,1),:,:,j] = cat(CT,freq, dims=3)
    end
    freq_list = hcat([(??_list[i,:] .- 1) ./(t_samp_list[i] * data_samp_period) for i in 1:size(t_samp_list,1)]...)
    ??_list, freq_list, arrange_CT_arr(CT_arr; len=Int(floor(minimum(t_samp_list)/2)), ??k = kmax-kmin + 1)
end

function get_??_sample(N, t_max, data, t_samp_list, c::Bool; kmin = 1, kmax = 10, L = 1,
        m=1, ??s=10) #??lim = 30 data_samp_period=10
	data_samp_period=dict[:dt]
    tsamp = Int(floor(t_max/??s))
    t_samp_list = t_samp_list == 0 ? [i*tsamp for i in 1:??s] : t_samp_list
    n_samp = Int.(floor.(t_max ./ t_samp_list))
    ??_list = zeros(size(t_samp_list,1),kmax-kmin + 1)
	CT_arr = zeros(maximum(t_samp_list),kmax-kmin + 1,2,size(t_samp_list,1))

    for j in 1:size(t_samp_list,1)
        ts = t_samp_list[j]
        d_part, t_m = partition_data(data, ts)
        @info "$j : t_samp set to $ts"
        if ts * n_samp[j] != t_m
            @warn "t_sample x n_sample does NOT equal t_m"
            break
        end
        ??lim = Int(floor(ts/2))
        CT, _ = get_Corr_over_all_samples(ts, N, t_max, d_part; kmin = kmin, kmax = kmax, L = L, m=m, in_Fourier = c)
        ??_list[j,:] = vcat([findall(x->x == maximum(CT[1:??lim,i]), CT[1:??lim,i]) for i in 1:size(CT,2)]...)

		freq = hcat([[l for l in 0:size(CT,1)-1] ./ (ts * data_samp_period) for i in 1:size(CT,2)]...)
		CT_arr[1:size(CT,1),:,:,j] = cat(CT,freq, dims=3)
    end
    freq_list = hcat([(??_list[i,:] .- 1) ./(t_samp_list[i] * data_samp_period) for i in 1:size(t_samp_list,1)]...)
    ??_list, freq_list, arrange_CT_arr(CT_arr; len=Int(floor(minimum(t_samp_list)/2)), ??k = kmax-kmin + 1)
end

function arrange_CT_arr(CT_arr; len=30, ??k=10)
        t_len=size(CT_arr,4)
        l = [i*len for i in 1:t_len]
        new_CT = zeros(??k,sum(l),2)
        for k in 1:??k
                x = vcat([CT_arr[1:i*len,k,2,i] for i in 1:t_len]...)
                y = vcat([CT_arr[1:i*len,k,1,i] for i in 1:t_len]...)
                idx = sortperm(x)
                new_CT[k,:,:] = hcat([x[idx],y[idx]]...)
        end
        new_CT
end

function get_freq_max_2(CT_arr)
        freq_max = zeros(10,2)
        for k in 1:10
                idx = sortperm(CT_arr[k,:,2])[end]
                freq_max[k,:] = [k, CT_arr[k,idx,1]]
        end
        freq_max
end

function get_freqs(dict; test=false, ?? = 0.0002, ??t = 10.0, ?? = 0.5)
    #dict for t_MAX, L_char,, num_part
    t_MAX = dict[:t_max]
    L_char = dict[:L_char]
    num_part = dict[:num_part]

    if test
        d = test_data(t_MAX; ?? = ??, ??t = ??t, ?? = ??, L = L_char)
    else
        d = load_data(num_part, t_MAX)
    end

    get_??_sample(num_part, t_MAX, d, 0, true; L=L_char)
	# get_??_sample(num_part, t_MAX, d, [500,1000], true; L=L_char)
end

function main()
	dim = dict[:dim]
	??_list, freq_list, CT_arr = get_freqs(dict)
	m = reshape(mean(freq_list, dims=2),:)
	st = reshape(std(freq_list, dims=2),:)
	fig = plot(1:10, m, ribbon=st, fillalpha=0.2, label="mean", lw=4,
	    	frame=:box, xlabel="k (wave number)", ylabel="freq (1/$dim)", legend=:topleft,
	    	title=dict[:title])
	fig_name = split(filename,".")[1]
	savefig(fig, dict[:PathOut] * "/" * fig_name * ".png")
	writedlm(dict[:PathOut] * "/" * fig_name * "_modes.dat", ??_list')
	writedlm(dict[:PathOut] * "/" * fig_name * "_freq.dat", freq_list)
	# writedlm(dict[:PathOut] * "/" * fig_name * "_freq_method2.dat",
	# 		get_freq_max_2(CT_arr))
end
