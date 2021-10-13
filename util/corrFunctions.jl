using Statistics, SparseArrays, LinearAlgebra

# mutable struct structureFactor
# 	fields
# end
#=
#Correlation Functions
=#
function longitudinalDensityCurrent(data, N; kmin = 1, kmax = 10, L = 1, m=1)
    #=
    Generates Jₓ(k,t) = m/N * Σ vₓⁱ(t) exp(-𝚤 k ⋅ rⁱ(t))
    As a matter of sanity check, if data = rand(tmax, 6N+1) where
    mean(rand()) = 1/2, then Jₓ(0,t) = m/2
    =#
    x, y, z, vx, vy, vz = data
    #Longidutinal to x
    Jxkx = hcat([m/√N*sum(vx .* cos.(2π * k .* x / L), dims=2) for k in kmin:kmax]...)
    JxkxIm = hcat([m/√N*sum(vx .* sin.(2π * k .* x / L), dims=2) for k in kmin:kmax]...)

    #Transversal to y
    Jyky = hcat([m/√N*sum(vy .* cos.(2π * k .* y / L), dims=2) for k in kmin:kmax]...)
    JykyIm = hcat([m/√N*sum(vy .* sin.(2π * k .* y / L), dims=2) for k in kmin:kmax]...)

    #Transversal to z
    Jzkz = hcat([m/√N*sum(vz .* cos.(2π * k .* z / L), dims=2) for k in kmin:kmax]...)
    JzkzIm = hcat([m/√N*sum(vz .* sin.(2π * k .* z / L), dims=2) for k in kmin:kmax]...)

    Jxkx, Jyky, Jzkz, JxkxIm, JykyIm, JzkzIm
end

function transversalDensityCurrent(data, N; kmin = 1, kmax = 10, L = 1, m=1)
    #=
    Generates Jₓ(k,t) = m/N * Σ vₓⁱ(t) exp(-𝚤 k ⋅ rⁱ(t))
    As a matter of sanity check, if data = rand(tmax, 6N+1) where
    mean(rand()) = 1/2, then Jₓ(0,t) = m/2
    =#
    x, y, z, vx, vy, vz = data
    #Transversal to z
    Jxkz = hcat([m/√N*sum(vx .* cos.(2π * k .* z / L), dims=2) for k in kmin:kmax]...)
    JxkzIm = hcat([m/√N*sum(vx .* sin.(2π * k .* z / L), dims=2) for k in kmin:kmax]...)

    Jykz = hcat([m/√N*sum(vy .* cos.(2π * k .* z / L), dims=2) for k in kmin:kmax]...)
    JykzIm = hcat([m/√N*sum(vy .* sin.(2π * k .* z / L), dims=2) for k in kmin:kmax]...)
    #Transversal to x
    Jzkx = hcat([m/√N*sum(vz .* cos.(2π * k .* x / L), dims=2) for k in kmin:kmax]...)
    JzkxIm = hcat([m/√N*sum(vz .* sin.(2π * k .* x / L), dims=2) for k in kmin:kmax]...)

    Jykx = hcat([m/√N*sum(vy .* cos.(2π * k .* x / L), dims=2) for k in kmin:kmax]...)
    JykxIm = hcat([m/√N*sum(vy .* sin.(2π * k .* x / L), dims=2) for k in kmin:kmax]...)
    #Transversal to y
    Jxky = hcat([m/√N*sum(vx .* cos.(2π * k .* y / L), dims=2) for k in kmin:kmax]...)
    JxkyIm = hcat([m/√N*sum(vx .* sin.(2π * k .* y / L), dims=2) for k in kmin:kmax]...)

    Jzky = hcat([m/√N*sum(vz .* cos.(2π * k .* y / L), dims=2) for k in kmin:kmax]...)
    JzkyIm = hcat([m/√N*sum(vz .* sin.(2π * k .* y / L), dims=2) for k in kmin:kmax]...)

    Jxkz, Jykz, Jxky, Jzky, Jykx, Jzkx, JxkzIm, JykzIm, JxkyIm, JzkyIm, JykxIm, JzkxIm
end

function longitudinalDensityCorrelation(data, N, tmax; kmin = 1, kmax = 10, L = 1, m=1)
    #=
    Generates ⟨ JxJx ⟩(k,τ) =  Σₜ Jₓ(k,t+τ)Jₓ(k,t)/Z where Z = tmax-τ
    As a matter of sanity check, if data = rand(tmax, 6N+1) where
    mean(rand()) = 1/2, then ⟨ JxJx ⟩(k,τ) = m²/4
    =#

    Jxkx, Jyky, Jzkz, JxkxIm, JykyIm, JzkzIm = longitudinalDensityCurrent(data, N; kmin = kmin, kmax = kmax, L = L, m=m)

    JxJxkx = hcat([sum([Jxkx[τ+t,:] .* Jxkx[t,:] .+ JxkxIm[τ+t,:] .* JxkxIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
    # JxJxky = hcat([sum([Jxky[τ+t,:] .* Jxky[t,:] .+ JxkyIm[τ+t,:] .* JxkyIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JxJxkxIm = hcat([sum([Jxkx[τ+t,:] .* JxkxIm[t,:] .+ JxkxIm[τ+t,:] .* Jxkx[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
    # JxJxkyIm = hcat([sum([Jxky[τ+t,:] .* JxkyIm[t,:] .+ JxkyIm[τ+t,:] .* Jxky[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JyJyky = hcat([sum([Jyky[τ+t,:] .* Jyky[t,:] .+ JykyIm[τ+t,:] .* JykyIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
    # JyJykx = hcat([sum([Jykx[τ+t,:] .* Jykx[t,:] .+ JykxIm[τ+t,:] .* JykxIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JyJykyIm = hcat([sum([Jyky[τ+t,:] .* JykyIm[t,:] .+ JykyIm[τ+t,:] .* Jyky[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
    # JyJykxIm = hcat([sum([Jykx[τ+t,:] .* JykxIm[t,:] .+ JykxIm[τ+t,:] .* Jykx[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JzJzkz = hcat([sum([Jzkz[τ+t,:] .* Jzkz[t,:] .+ JzkzIm[τ+t,:] .* JzkzIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
    # JzJzky = hcat([sum([Jzky[τ+t,:] .* Jzky[t,:] .+ JzkyIm[τ+t,:] .* JzkyIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JzJzkzIm = hcat([sum([Jzkz[τ+t,:] .* JzkzIm[t,:] .+ JzkzIm[τ+t,:] .* Jzkz[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
    # JzJzkyIm = hcat([sum([Jzky[τ+t,:] .* JzkyIm[t,:] .+ JzkyIm[τ+t,:] .* Jzky[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JxJxkx, JyJyky, JzJzkz, JxJxkxIm, JyJykyIm, JzJzkzIm
end

function longitudinalDensityCorrelation(data, N, tmax, β::Int; kmin = 1, kmax = 10, L = 1, m=1)
    #=
    Generates ⟨ JxJx ⟩(k,τ) =  Σₜ Jₓ(k,t+τ)Jₓ(k,t)/Z where Z = tmax-τ
    As a matter of sanity check, if data = rand(tmax, 6N+1) where
    mean(rand()) = 1/2, then ⟨ JxJx ⟩(k,τ) = m²/4
    =#

    Jxkx, Jyky, Jzkz, JxkxIm, JykyIm, JzkzIm = longitudinalDensityCurrent(data, N; kmin = kmin, kmax = kmax, L = L, m=m)

	JxJxkx = hcat([diag((Jxkx' * sparse_with_diag(i,tmax) * Jxkx .+ JxkxIm' * sparse_with_diag(i,tmax) * JxkxIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JxJxkz = hcat([sum([Jxkz[τ+t,:] .* Jxkz[t,:] .+ JxkzIm[τ+t,:] .* JxkzIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
	# JxJxky = hcat([diag((Jxky' * sparse_with_diag(i,tmax) * Jxky .+ JxkyIm' * sparse_with_diag(i,tmax) * JxkyIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JxJxky = hcat([sum([Jxky[τ+t,:] .* Jxky[t,:] .+ JxkyIm[τ+t,:] .* JxkyIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

	JxJxkxIm = hcat([diag((Jxkx' * sparse_with_diag(i,tmax) * JxkxIm .+ JxkxIm' * sparse_with_diag(i,tmax) * Jxkx)/(tmax-i)) for i in 0:tmax-1]...)
    # JxJxkzIm = hcat([sum([Jxkz[τ+t,:] .* JxkzIm[t,:] .+ JxkzIm[τ+t,:] .* Jxkz[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
	# JxJxkyIm = hcat([diag((Jxky' * sparse_with_diag(i,tmax) * JxkyIm .+ JxkyIm' * sparse_with_diag(i,tmax) * Jxky)/(tmax-i)) for i in 0:tmax-1]...)
    # JxJxkyIm = hcat([sum([Jxky[τ+t,:] .* JxkyIm[t,:] .+ JxkyIm[τ+t,:] .* Jxky[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

	JyJyky = hcat([diag((Jyky' * sparse_with_diag(i,tmax) * Jyky .+ JykyIm' * sparse_with_diag(i,tmax) * JykyIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JyJykz = hcat([sum([Jykz[τ+t,:] .* Jykz[t,:] .+ JykzIm[τ+t,:] .* JykzIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
	# JyJykx = hcat([diag((Jykx' * sparse_with_diag(i,tmax) * Jykx .+ JykxIm' * sparse_with_diag(i,tmax) * JykxIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JyJykx = hcat([sum([Jykx[τ+t,:] .* Jykx[t,:] .+ JykxIm[τ+t,:] .* JykxIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

	JyJykyIm = hcat([diag((Jyky' * sparse_with_diag(i,tmax) * JykyIm .+ JykyIm' * sparse_with_diag(i,tmax) * Jyky)/(tmax-i)) for i in 0:tmax-1]...)
    # JyJykzIm = hcat([sum([Jykz[τ+t,:] .* JykzIm[t,:] .+ JykzIm[τ+t,:] .* Jykz[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
	# JyJykxIm = hcat([diag((Jykx' * sparse_with_diag(i,tmax) * JykxIm .+ JykxIm' * sparse_with_diag(i,tmax) * Jykx)/(tmax-i)) for i in 0:tmax-1]...)
    # JyJykxIm = hcat([sum([Jykx[τ+t,:] .* JykxIm[t,:] .+ JykxIm[τ+t,:] .* Jykx[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

	JzJzkz = hcat([diag((Jzkz' * sparse_with_diag(i,tmax) * Jzkz .+ JzkzIm' * sparse_with_diag(i,tmax) * JzkzIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JzJzkx = hcat([sum([Jzkx[τ+t,:] .* Jzkx[t,:] .+ JzkxIm[τ+t,:] .* JzkxIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
	# JzJzky = hcat([diag((Jzky' * sparse_with_diag(i,tmax) * Jzky .+ JzkyIm' * sparse_with_diag(i,tmax) * JzkyIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JzJzky = hcat([sum([Jzky[τ+t,:] .* Jzky[t,:] .+ JzkyIm[τ+t,:] .* JzkyIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

	JzJzkzIm = hcat([diag((Jzkz' * sparse_with_diag(i,tmax) * JzkzIm .+ JzkzIm' * sparse_with_diag(i,tmax) * Jzkz)/(tmax-i)) for i in 0:tmax-1]...)
    # JzJzkxIm = hcat([sum([Jzkx[τ+t,:] .* JzkxIm[t,:] .+ JzkxIm[τ+t,:] .* Jzkx[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
	# JzJzkyIm = hcat([diag((Jzky' * sparse_with_diag(i,tmax) * JzkyIm .+ JzkyIm' * sparse_with_diag(i,tmax) * Jzky)/(tmax-i)) for i in 0:tmax-1]...)
    # JzJzkyIm = hcat([sum([Jzky[τ+t,:] .* JzkyIm[t,:] .+ JzkyIm[τ+t,:] .* Jzky[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JxJxkx, JyJyky, JzJzkz, JxJxkxIm, JyJykyIm, JzJzkzIm
end

function transversalDensityCorrelation(data, N, tmax; kmin = 1, kmax = 10, L = 1, m=1)
    #=
    Generates ⟨ JxJx ⟩(k,τ) =  Σₜ Jₓ(k,t+τ)Jₓ(k,t)/Z where Z = tmax-τ
    As a matter of sanity check, if data = rand(tmax, 6N+1) where
    mean(rand()) = 1/2, then ⟨ JxJx ⟩(k,τ) = m²/4
    =#

    Jxkz, Jykz, Jxky, Jzky, Jykx, Jzkx, JxkzIm, JykzIm, JxkyIm, JzkyIm,
        JykxIm, JzkxIm = transversalDensityCurrent(data, N; kmin = kmin, kmax = kmax, L = L, m=m)

    JxJxkz = hcat([sum([Jxkz[τ+t,:] .* Jxkz[t,:] .+ JxkzIm[τ+t,:] .* JxkzIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
    JxJxky = hcat([sum([Jxky[τ+t,:] .* Jxky[t,:] .+ JxkyIm[τ+t,:] .* JxkyIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JxJxkzIm = hcat([sum([Jxkz[τ+t,:] .* JxkzIm[t,:] .+ JxkzIm[τ+t,:] .* Jxkz[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
    JxJxkyIm = hcat([sum([Jxky[τ+t,:] .* JxkyIm[t,:] .+ JxkyIm[τ+t,:] .* Jxky[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JyJykz = hcat([sum([Jykz[τ+t,:] .* Jykz[t,:] .+ JykzIm[τ+t,:] .* JykzIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
    JyJykx = hcat([sum([Jykx[τ+t,:] .* Jykx[t,:] .+ JykxIm[τ+t,:] .* JykxIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JyJykzIm = hcat([sum([Jykz[τ+t,:] .* JykzIm[t,:] .+ JykzIm[τ+t,:] .* Jykz[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
    JyJykxIm = hcat([sum([Jykx[τ+t,:] .* JykxIm[t,:] .+ JykxIm[τ+t,:] .* Jykx[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JzJzkx = hcat([sum([Jzkx[τ+t,:] .* Jzkx[t,:] .+ JzkxIm[τ+t,:] .* JzkxIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
    JzJzky = hcat([sum([Jzky[τ+t,:] .* Jzky[t,:] .+ JzkyIm[τ+t,:] .* JzkyIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JzJzkxIm = hcat([sum([Jzkx[τ+t,:] .* JzkxIm[t,:] .+ JzkxIm[τ+t,:] .* Jzkx[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
    JzJzkyIm = hcat([sum([Jzky[τ+t,:] .* JzkyIm[t,:] .+ JzkyIm[τ+t,:] .* Jzky[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JxJxkz, JxJxky, JyJykz, JyJykx, JzJzkx, JzJzky, JxJxkzIm, JxJxkyIm, JyJykzIm, JyJykxIm, JzJzkxIm, JzJzkyIm
end

function transversalDensityCorrelation(data, N, tmax, β::Int; kmin = 1, kmax = 10, L = 1, m=1)
    #=
    Generates ⟨ JxJx ⟩(k,τ) =  Σₜ Jₓ(k,t+τ)Jₓ(k,t)/Z where Z = tmax-τ
    As a matter of sanity check, if data = rand(tmax, 6N+1) where
    mean(rand()) = 1/2, then ⟨ JxJx ⟩(k,τ) = m²/4
    =#

    Jxkz, Jykz, Jxky, Jzky, Jykx, Jzkx, JxkzIm, JykzIm, JxkyIm, JzkyIm,
        JykxIm, JzkxIm = transversalDensityCurrent(data, N; kmin = kmin, kmax = kmax, L = L, m=m)

	JxJxkz = hcat([diag((Jxkz' * sparse_with_diag(i,tmax) * Jxkz .+ JxkzIm' * sparse_with_diag(i,tmax) * JxkzIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JxJxkz = hcat([sum([Jxkz[τ+t,:] .* Jxkz[t,:] .+ JxkzIm[τ+t,:] .* JxkzIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
	JxJxky = hcat([diag((Jxky' * sparse_with_diag(i,tmax) * Jxky .+ JxkyIm' * sparse_with_diag(i,tmax) * JxkyIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JxJxky = hcat([sum([Jxky[τ+t,:] .* Jxky[t,:] .+ JxkyIm[τ+t,:] .* JxkyIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

	JxJxkzIm = hcat([diag((Jxkz' * sparse_with_diag(i,tmax) * JxkzIm .+ JxkzIm' * sparse_with_diag(i,tmax) * Jxkz)/(tmax-i)) for i in 0:tmax-1]...)
    # JxJxkzIm = hcat([sum([Jxkz[τ+t,:] .* JxkzIm[t,:] .+ JxkzIm[τ+t,:] .* Jxkz[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
	JxJxkyIm = hcat([diag((Jxky' * sparse_with_diag(i,tmax) * JxkyIm .+ JxkyIm' * sparse_with_diag(i,tmax) * Jxky)/(tmax-i)) for i in 0:tmax-1]...)
    # JxJxkyIm = hcat([sum([Jxky[τ+t,:] .* JxkyIm[t,:] .+ JxkyIm[τ+t,:] .* Jxky[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

	JyJykz = hcat([diag((Jykz' * sparse_with_diag(i,tmax) * Jykz .+ JykzIm' * sparse_with_diag(i,tmax) * JykzIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JyJykz = hcat([sum([Jykz[τ+t,:] .* Jykz[t,:] .+ JykzIm[τ+t,:] .* JykzIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
	JyJykx = hcat([diag((Jykx' * sparse_with_diag(i,tmax) * Jykx .+ JykxIm' * sparse_with_diag(i,tmax) * JykxIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JyJykx = hcat([sum([Jykx[τ+t,:] .* Jykx[t,:] .+ JykxIm[τ+t,:] .* JykxIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

	JyJykzIm = hcat([diag((Jykz' * sparse_with_diag(i,tmax) * JykzIm .+ JykzIm' * sparse_with_diag(i,tmax) * Jykz)/(tmax-i)) for i in 0:tmax-1]...)
    # JyJykzIm = hcat([sum([Jykz[τ+t,:] .* JykzIm[t,:] .+ JykzIm[τ+t,:] .* Jykz[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
	JyJykxIm = hcat([diag((Jykx' * sparse_with_diag(i,tmax) * JykxIm .+ JykxIm' * sparse_with_diag(i,tmax) * Jykx)/(tmax-i)) for i in 0:tmax-1]...)
    # JyJykxIm = hcat([sum([Jykx[τ+t,:] .* JykxIm[t,:] .+ JykxIm[τ+t,:] .* Jykx[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

	JzJzkx = hcat([diag((Jzkx' * sparse_with_diag(i,tmax) * Jzkx .+ JzkxIm' * sparse_with_diag(i,tmax) * JzkxIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JzJzkx = hcat([sum([Jzkx[τ+t,:] .* Jzkx[t,:] .+ JzkxIm[τ+t,:] .* JzkxIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
	JzJzky = hcat([diag((Jzky' * sparse_with_diag(i,tmax) * Jzky .+ JzkyIm' * sparse_with_diag(i,tmax) * JzkyIm)/(tmax-i)) for i in 0:tmax-1]...)
    # JzJzky = hcat([sum([Jzky[τ+t,:] .* Jzky[t,:] .+ JzkyIm[τ+t,:] .* JzkyIm[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

	JzJzkxIm = hcat([diag((Jzkx' * sparse_with_diag(i,tmax) * JzkxIm .+ JzkxIm' * sparse_with_diag(i,tmax) * Jzkx)/(tmax-i)) for i in 0:tmax-1]...)
    # JzJzkxIm = hcat([sum([Jzkx[τ+t,:] .* JzkxIm[t,:] .+ JzkxIm[τ+t,:] .* Jzkx[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)
	JzJzkyIm = hcat([diag((Jzky' * sparse_with_diag(i,tmax) * JzkyIm .+ JzkyIm' * sparse_with_diag(i,tmax) * Jzky)/(tmax-i)) for i in 0:tmax-1]...)
    # JzJzkyIm = hcat([sum([Jzky[τ+t,:] .* JzkyIm[t,:] .+ JzkyIm[τ+t,:] .* Jzky[t,:] for t in 1:tmax-τ])/(tmax-τ) for τ in 0:tmax-1]...)

    JxJxkz, JxJxky, JyJykz, JyJykx, JzJzkx, JzJzky, JxJxkzIm, JxJxkyIm, JyJykzIm, JyJykxIm, JzJzkxIm, JzJzkyIm
end

function sparse_with_diag(idx, tmax)
    A = blockdiag(sparse([i for i in 1+idx:tmax],[i for i in 1:tmax-idx],[1 for i in 1:tmax-idx]), sparse(0I, 0, idx))
    A
end

function getTransversalCorrelationFunction(data, N, tmax; kmin = 1, kmax = 10, L = 1, m=1)
	use_sparse = 1
	if use_sparse == 0
    	CT = transversalDensityCorrelation(data, N, tmax; kmin = kmin, kmax = kmax, L = L, m=m)
		return mean(CT[1:6]), mean(CT[7:12])
	elseif use_sparse == 1
		CT = transversalDensityCorrelation(data, N, tmax,1; kmin = kmin, kmax = kmax, L = L, m=m)
		return  mean(CT[1:6]), mean(CT[7:12])
	end
end

function getLongitudinalCorrelationFunction(data, N, tmax; kmin = 1, kmax = 10, L = 1, m=1)
	use_sparse = 1
	if use_sparse == 0
    	CL = longitudinalDensityCorrelation(data, N, tmax; kmin = kmin, kmax = kmax, L = L, m=m)
		return mean(CL[1:3]), mean(CL[4:6])
	elseif use_sparse == 1
		CL = longitudinalDensityCorrelation(data, N, tmax,1; kmin = kmin, kmax = kmax, L = L, m=m)
		return  mean(CL[1:3]), mean(CL[4:6])
	end
end
