# A flexible right-preconditioned GMRES pressure solve. The existing multigrid
# V-cycle is useful even when it is not a positive-definite preconditioner, which
# breaks the scalar divisions in flexible CG on this refined-grid scene.
# Algorithm: Saad 1993; see also PETSc KSPFGMRES (right preconditioning).

const SPARSE_FGMRES_WORK = WeakKeyDict{Any,Any}()

function sparse_fgmres_workspace(s, restart)
    get!(SPARSE_FGMRES_WORK,s) do
        field()=Field(KA.zeros(s.backend,Float32,length(s.φ.fine)),KA.zeros(s.backend,Float32,size(s.φ.coarse)...))
        (;V=[field() for _ in 1:restart+1],Z=[field() for _ in 1:restart])
    end
end

function sparse_diagonal_precondition!(s,z,r)
    zero!(s,s.tmp)
    L,t=level0(s),s.topo
    jacobifine_kernel!(s.backend,256)(z.fine,s.tmp.fine,r.fine,s.tmp.coarse,
        L.cfx,L.cfy,L.cfz,L.dDf,L.activef,t,L.v;ndrange=finecells(s))
    jacobicoarse_kernel!(s.backend,WG3)(z.coarse,s.tmp.coarse,r.coarse,s.tmp.fine,
        L.cfx,L.cfy,L.cfz,L.ccx,L.ccy,L.ccz,L.dDc,L.activec,t,L.v;ndrange=size(z.coarse))
    return z
end

"Pressure with a checked true residual, without assuming a definite V-cycle."
function pressure_fgmres!(s;tol=5f-4,maxiter=240,restart=8,sweeps=1,boost=1f0)
    hierarchy!(s)
    work=sparse_fgmres_workspace(s,restart)
    length(work.Z)==restart || error("Pressure restart differs from the existing workspace")
    bnorm=sqrt(inner(s,s.b,s.b))
    isfinite(bnorm) || error("Nonfinite pressure RHS at t=$(s.t)")
    bnorm==0f0 && (zero!(s,s.P);return 0)
    target=tol*bnorm
    actual_residual!()=begin
        apply!(s,s.q,s.P)
        copyfield!(s,s.r,s.b)
        axpy!(s,s.r,-1f0,s.q)
        sqrt(inner(s,s.r,s.r))
    end
    residual=actual_residual!()
    iterations=0
    while residual>target && iterations<maxiter
        isfinite(residual) || error("Nonfinite pressure residual at t=$(s.t)")
        H=zeros(Float64,restart+1,restart)
        g=zeros(Float64,restart+1);g[1]=residual
        cs=zeros(Float64,restart);sn=zeros(Float64,restart)
        zero!(s,work.V[1]);axpy!(s,work.V[1],1f0/residual,s.r)
        used=0
        for j in 1:min(restart,maxiter-iterations)
            vcycle!(s,s.z,work.V[j];sweeps,boost)
            # The diagonal component covers residual directions that this
            # approximate V-cycle handles poorly at refinement interfaces.
            sparse_diagonal_precondition!(s,work.Z[j],work.V[j])
            axpy!(s,work.Z[j],1f0,s.z)
            apply!(s,s.q,work.Z[j])
            # Reorthogonalization keeps single-precision basis vectors useful;
            # the small least-squares system is f64. The default true residual
            # tolerance is above the measured Float32 residual stagnation floor.
            for _ in 1:2, i in 1:j
                h=inner(s,s.q,work.V[i])
                H[i,j]+=h
                axpy!(s,s.q,-h,work.V[i])
            end
            hnext=sqrt(inner(s,s.q,s.q))
            isfinite(hnext) || error("Nonfinite pressure Arnoldi vector at t=$(s.t)")
            H[j+1,j]=hnext
            if hnext>0f0
                zero!(s,work.V[j+1]);axpy!(s,work.V[j+1],1f0/hnext,s.q)
            end
            for i in 1:j-1
                a,b=H[i,j],H[i+1,j]
                H[i,j]=cs[i]*a+sn[i]*b
                H[i+1,j]=-sn[i]*a+cs[i]*b
            end
            a,b=H[j,j],H[j+1,j]
            hyp=hypot(a,b)
            hyp>0 || error("Pressure Arnoldi breakdown at t=$(s.t)")
            cs[j]=a/hyp;sn[j]=b/hyp
            H[j,j]=hyp;H[j+1,j]=0
            g[j+1]=-sn[j]*g[j];g[j]=cs[j]*g[j]
            used=j;iterations+=1
            (abs(g[j+1])<=target || hnext==0f0) && break
        end
        y=UpperTriangular(H[1:used,1:used])\g[1:used]
        for j in 1:used
            axpy!(s,s.P,Float32(y[j]),work.Z[j])
        end
        residual=actual_residual!()
    end
    residual<=target || error("Pressure did not converge at t=$(s.t): $(residual/bnorm) after $iterations iterations")
    return iterations
end

# Accept the fast CG solve only when its actual residual meets the same tolerance;
# flexible GMRES handles cases where the approximate V-cycle defeats CG.
function pressure!(s::SparseBasin{B,S};tol=5f-4,boost=1.5f0,maxiter=240,sweeps=1) where {B,S<:SurfaceStir}
    iterations=0
    try
        iterations=invoke(pressure!,Tuple{SparseBasin},s;tol,boost,sweeps,maxiter=min(80,maxiter),warn_on_nonconvergence=false)
    catch err
        err isa ErrorException && (startswith(err.msg,"Nonfinite pressure") || startswith(err.msg,"Pressure CG ")) || rethrow()
    end
    apply!(s,s.q,s.P)
    copyfield!(s,s.r,s.b);axpy!(s,s.r,-1f0,s.q)
    bnorm=sqrt(inner(s,s.b,s.b))
    rnorm=sqrt(inner(s,s.r,s.r))
    rnorm<=tol*bnorm && return iterations
    (!isfinite(rnorm) || rnorm>2f0*bnorm) && zero!(s,s.P)
    return iterations+pressure_fgmres!(s;tol,boost,maxiter,sweeps)
end
