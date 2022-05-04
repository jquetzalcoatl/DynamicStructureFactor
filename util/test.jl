using Plots, Random, Interact, Distributions, LinearAlgebra, DelimitedFiles

# N = 30 #number of osc
# plot([rand(Distributions.Binomial(30,0.1)) for i in 1:100], st=:histogram)
# rand([0,-rand(Distributions.Binomial(30,1/150))-0.0001*rand()])
function randomHessian(N; dims=2, p=0.0)
    p = p == 0.0 ? 1/(5*N) : p
    m = zeros(dims,N,N)
    for k in 1:dims
        # m[k,:,:] = [rand([0,-rand(1:N)-0.1*rand()]) for j in 1:N, i in 1:N]
        m[k,:,:] = [rand([0,-rand(Distributions.Binomial(N,p))-0.0001*rand()]) for j in 1:N, i in 1:N]
        m[k,:,:] = (m[k,:,:] + m[k,:,:]')/2
        for i in 1:N
            m[k,i,i] = -sum(m[k,i,:])+m[k,i,i]
        end
    end
    m ./ maximum(m, dims=3)
end

function expHessian(N; dims=2)
    m = zeros(dims,N,N)
    for k in 1:dims
        m[k,:,:] = [-exp(-(i-j)^2) for i in 1:N, j in 1:N]
        m[k,:,:] = (m[k,:,:] + m[k,:,:]')/2
        for i in 1:N
            m[k,i,i] = -sum(m[k,i,:])+m[k,i,i]
        end
    end
    m
end

function nNeighHessian(N; dims=2, n=2)
    m = zeros(dims,N,N)
    for k in 1:dims
        m[k,:,:] = [ abs(i-j) < n ? -1 : 0 for i in 1:N, j in 1:N]
        m[k,:,:] = (m[k,:,:] + m[k,:,:]')/2
        for i in 1:N
            m[k,i,i] = -sum(m[k,i,:])+m[k,i,i]
        end
    end
    m
end

function genHessian(N, type; dims=2, n=2)
    if type == "random"
        return randomHessian(N; dims)
    elseif type == "exp"
        return expHessian(N; dims)
    elseif type == "nNeigh"
        return nNeighHessian(N; dims,n)
    end
end

function gen_Cryst(N; L=[5,5,5])
    init_pos = zeros(3,N)
    for i in 0:N-1
        init_pos[1,i+1] = i % L[1]
        init_pos[2,i+1] = (i ÷ L[1]) % L[2]
        init_pos[3,i+1] = i ÷ (L[1]*L[2])
    end
    init_pos
end

function generateData(N; dims=2, steps=500, Δt=0.01, arrange = "Crystal",
            typeHessian="random", amp = 0.4, nModes = 0, idxStart = 1)
    if arrange == "Crystal"
        L = [Int(floor(N^(1/dims))) for i in 1:dims]
        if dims==1
            append!(L,1)
        end
        init_pos = gen_Cryst(N; L)
    end

    M = genHessian(N, typeHessian; dims)
    r = zeros(dims, N, steps)
    vel = zeros(dims, N, steps)
    nModes = nModes == 0 ? N : nModes
    for k in 1:dims
        d,v = eigen(M[k,:,:])
        # r[k,:,:] = hcat([sum([amp * d[i]^2 .* v[:,i] .* sin(2π*t*d[i]) for i in idxStart:nModes]) .+ init_pos[k,:] for t in 0:Δt:(steps-1)*Δt]...)
        r[k,:,:] = hcat([sum([amp .* v[:,i] .* sin(2π*t*d[i]) for i in idxStart:nModes]) .+ init_pos[k,:] for t in 0:Δt:(steps-1)*Δt]...)
        vel[k,:,:] = hcat([sum([amp .* v[:,i] .* cos(2π*t*d[i]) * 2π * d[i] for i in idxStart:nModes]) for t in 0:Δt:(steps-1)*Δt]...)
    end
    permutedims(r,(1,3,2)), permutedims(vel,(1,3,2)), M
end
