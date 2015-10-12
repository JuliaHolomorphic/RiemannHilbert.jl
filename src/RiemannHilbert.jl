
module RiemannHilbert
    using Base, ApproxFun, SingularIntegralEquations
    import SingularIntegralEquations:cauchyforward,cauchybackward

export cauchymatrix

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
