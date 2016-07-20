
module RiemannHilbert
using Base, ApproxFun, SingularIntegralEquations, DualNumbers
    import SingularIntegralEquations: stieltjesforward, stieltjesbackward
    import ApproxFun: mobius

export cauchymatrix


# represent a + b*log(ε)
immutable LogNumber
    constant::Complex128
    log::Complex128
end


function fpstieltjes(f::Fun,z::Dual)
    x = mobius(domain(f),z)
    if !isinf(mobius(domain(f),Inf))
        error("Not implemented")
    end
    cfs = coefficients(f,Chebyshev)
    if realpart(x) ≈ 1
        c = -(log(dualpart(x))-log(2)) * sum(cfs)
        r = 0.0
        for k=2:2:length(cfs)-1
            r += 1/(k-1)
            c += -r*4*cfs[k+1]
        end
        r = 1.0
        for k=1:2:length(cfs)-1
            r += 1/(k-2)
            c += -(r+1/(2k))*4*cfs[k+1]
        end
        c
    elseif realpart(x) ≈ -1
        v = -(log(-dualpart(x))-log(2))
        if !isempty(cfs)
            c = -v*cfs[1]
        end
        r = 0.0
        for k=2:2:length(cfs)-1
            r += 1/(k-1)
            c += r*4*cfs[k+1]
            c += -v*cfs[k+1]
        end
        r = 1.0
        for k=1:2:length(cfs)-1
            r += 1/(k-2)
            c += -(r+1/(2k))*4*cfs[k+1]
            c += v*cfs[k+1]
        end
        c
    else
        error("Not implemented")
    end
end

fpcauchy(x...) = fpstieltjes(x...)/(-2π*im)



function stieltjesmatrix(space,pts::Vector,s::Bool)
    n=length(pts)
    C=Array(Complex128,n,n)
    for k=1:n
         C[k,:]=stieltjesforward(s,space,n,pts[k])
    end
    C
end

function stieltjesmatrix(space,pts::Vector)
    n=length(pts)
    C=zeros(Complex128,n,n)
    for k=1:n
        cfs=stieltjesbackward(space,pts[k])
        C[k,1:min(length(cfs),n)]=cfs
    end

    C
end


stieltjesmatrix(space,n::Integer,s::Bool)=stieltjesmatrix(space,points(space,n),s)
stieltjesmatrix(space,space2,n::Integer)=stieltjesmatrix(space,points(space2,n))

cauchymatrix(x...) = stieltjesmatrix(x...)/(-2π*im)

end #module
