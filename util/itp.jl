using Interpolations

function getSpline(data; Δx=10)
    x = data[:,1]
    y = data[:,2]
    stepR = x[1]:(x[end] - x[1])/(size(x,1)-1):x[end]
    itp = interpolate(y, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, stepR)

    ∇_list = collect(x[1]:(x[end] - x[end-1])/Δx:x[end])
    # ∇ = vcat([Interpolations.gradient(sitp, x) for x in ∇_list]...)
    sitp#, ∇_list, ∇
end

function plotSpline(corrType, dict; xmax = 20, mixWindows=true)
    suffix = mixWindows ? "_CorrelationDataTimeWindowsMixed/" : "_CorrelationData/"
    pathout = dict[:PathOut] * split(dict[:title], ".")[1] * suffix
    # files = readdir("/Users/javier/Desktop/Julia/Correlation/Ge00Sb00Te100T823K/Ge00Sb00Te100T823K_CorrelationDataTimeWindowsMixed/")
    files = readdir(pathout)
    hline(0)
    for file in files
        if split(file,"_")[1] == corrType
            dd = readdlm(pathout * file)
            stepR = dd[1,1]:(dd[end,1] - dd[end-1,1])/10:dd[xmax,1]
            sitp = getSpline(dd)
            plot!(stepR, x->sitp(x), label="BSpline $(file)", frame=:box, lw=3)
            plot!(dd[1:xmax,1], dd[1:xmax,2], label="Data $(file)", seriestype=:scatter, markerstrokewidth=0, ms=7)
        end
    end
    plot!(size=(500,700), xlabel="w", ylabel=corrType, margin = 5Plots.mm)
end

function getSplines(corrType, dict; xmax = 20, mixWindows=true)
    suffix = mixWindows ? "_CorrelationDataTimeWindowsMixed/" : "_CorrelationData/"
    pathout = dict[:PathOut] * split(dict[:title], ".")[1] * suffix
    # files = readdir("/Users/javier/Desktop/Julia/Correlation/Ge00Sb00Te100T823K/Ge00Sb00Te100T823K_CorrelationDataTimeWindowsMixed/")
    files = readdir(pathout)
    splineList = []
    # hline(0)
    for file in files
        if split(file,"_")[1] == corrType
            dd = readdlm(pathout * file)
            stepR = dd[1,1]:(dd[end,1] - dd[end-1,1])/10:dd[end,1]
            sitp = getSpline(dd)
            append!(splineList, (stepR, sitp))
            # plot!(stepR, x->sitp(x), label="BSpline $(file)", frame=:box, lw=3)
            # plot!(dd[1:xmax,1], dd[1:xmax,2], label="Data $(file)", seriestype=:scatter, markerstrokewidth=0, ms=7)
        end
    end
    # plot!(size=(500,700), xlabel="w", ylabel=corrType, margin = 5Plots.mm)
    splineList
end
