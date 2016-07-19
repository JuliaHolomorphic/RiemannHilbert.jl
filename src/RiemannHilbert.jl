
module RiemannHilbert
    using Base, ApproxFun, SingularIntegralEquations
    import SingularIntegralEquations:cauchyforward,cauchybackward

export cauchymatrix


# represent a + b*log(ε)
immutable LogNumber
    constant::Complex128
    log::Complex128
end


function fpcauchy(f::Fun,z::Dual)
    x = mobius(domain(f),z)
    if isinf(mobius(domain(f),Inf))
        error("Not implemented")
    end
    cfs = coefficients(f,Chebyshev)
    if realpart(x) ≈ 1
        c = (log(dualpart(x))-log(2))/(2π*im) * sum(cfs)
        r = 0.0
        for k=2:2:length(cfs)-1
            r += 1/(k-1)
            c += r*2/(π*im)*cfs[k+1]
        end
        r = 1.0
        for k=1:2:length(cfs)-1
            r += 1/(k-2)
            c += (r+1/(2k))*2/(π*im)*cfs[k+1]
        end
        c
    elseif realpart(x) ≈ -1
        v = (log(-dualpart(x))-log(2))/(2π*im)
        if !isempty(cfs)
            c = -v*cfs[1]
        end
        r = 0.0
        for k=2:2:length(cfs)-1
            r += 1/(k-1)
            c += -r*2/(π*im)*cfs[k+1]
            c += -v*cfs[k+1]
        end
        r = 1.0
        for k=1:2:length(cfs)-1
            r += 1/(k-2)
            c += (r+1/(2k))*2/(π*im)*cfs[k+1]
            c += v*cfs[k+1]
        end
        c
    else
        error("Not implemented")
    end
end



function cauchymatrix(s::Bool,space,pts::Vector)
    n=length(pts)
    C=Array(Complex128,n,n)
    for k=1:n
         C[k,:]=cauchyforward(s,space,n,pts[k])
    end
    C
end

function cauchymatrix(space,pts::Vector)
    n=length(pts)
    C=zeros(Complex128,n,n)
    for k=1:n
        cfs=cauchybackward(space,pts[k])
        C[k,1:min(length(cfs),n)]=cfs
    end

    C
end


cauchymatrix(s::Bool,space,n::Integer)=cauchymatrix(s,space,points(space,n))
cauchymatrix(space,space2,n::Integer)=cauchymatrix(s,space,points(space2,n))

end #module
