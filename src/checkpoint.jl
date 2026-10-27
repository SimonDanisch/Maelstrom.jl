# Portable physical state at a completed frame, with no Vulkan handles or scratch
# buffers. A checkpoint can be restored into a larger block pool without rebaking.

host_stir(s::Stir)=Stir(s.basin,Array(s.waves),s.origin,s.offset,s.race)
host_stir(s::SurfaceStir)=SurfaceStir(host_stir(s.base),Array(s.heights),s.radius,s.ramp)

function save_sparse_checkpoint(s,w,path;fps=24)
    isapprox(s.t*fps,round(s.t*fps);atol=1e-6) || error("Save at a completed frame")
    w.frames==round(Int,s.t*fps) || error("Water and whitewater times differ")
    fields=Dict(name=>(;fine=Array(getfield(s,name).fine),coarse=Array(getfield(s,name).coarse))
        for name in (:φ,:types,:u,:v,:w,:fu,:fv,:fw,:P))
    particle_names=(:x,:vx,:vy,:vz,:cx,:cy,:cz,:lag)
    particles=Dict(name=>Array(view(getfield(s.particles,name),1:s.particles.n)) for name in particle_names)
    white=(;x=Array(w.x),v=Array(w.v),life=Array(w.life),kind=Array(w.kind),head=Array(w.head),
        rules=w.rules,lo=w.lo,hi=w.hi,h=w.h,floor=w.floor,g=w.g,frames=w.frames,stream=w.stream)
    data=(;format=1,julia=string(VERSION),fps,basin=s.basin,grid=s.grid,stir=host_stir(s.stir),
        band=s.band,nmin=s.nmin,cfl=s.cfl,rings=s.rings,vary=s.vary,reflect=s.reflect,
        capacity=capacity(s.particles),slots=Array(s.topo.slot),wheres=Array(s.topo.where),
        free=copy(s.free),nslots=s.nslots,time=s.t,steps=s.steps,volume0=s.volume0,iterations=copy(s.iterations),
        fields,particles,white)
    mkpath(dirname(path))
    if endswith(path,".zst")
        open(pipeline(`zstd -T2 -3 -q`,stdout=path*".tmp"),"w") do io
            serialize(io,data)
        end
    else
        serialize(path*".tmp",data)
    end
    mv(path*".tmp",path;force=true)
    return path
end

function load_sparse_checkpoint(path,backend;blocks=nothing,capacity_override=nothing)
    d=endswith(path,".zst") ? open(deserialize,`zstd -d -q -c $path`) : deserialize(path)
    d.format==1 && d.julia==string(VERSION) || error("Checkpoint format or Julia version differs")
    nblocks=blocks===nothing ? length(d.wheres) : max(blocks,length(d.wheres))
    cap=capacity_override===nothing ? d.capacity : capacity_override
    cap>=length(d.particles[:x]) || error("Particle capacity is too small")
    s=SparseBasin(d.basin,backend;dx=d.grid.dx,stir=d.stir,band=d.band,nmin=d.nmin,cfl=d.cfl,
        rings=d.rings,vary=d.vary,reflect=d.reflect,blocks=nblocks,capacity=cap)
    s.grid==d.grid || error("Checkpoint grid differs")
    s.slots=copy(d.slots);s.wheres[1:length(d.wheres)].=d.wheres
    s.nslots=d.nslots;s.free=copy(d.free)
    copyto!(s.topo.slot,s.slots);copyto!(s.topo.where,s.wheres)
    for (name,data) in d.fields
        f=getfield(s,name)
        copyto!(view(f.fine,1:length(data.fine)),data.fine)
        copyto!(f.coarse,data.coarse)
    end
    copyfield!(s,s.φ0,s.φ)
    for (name,old) in ((:u0,:u),(:v0,:v),(:w0,:w))
        copyfield!(s,getfield(s,name),getfield(s,old))
    end
    for (name,data) in d.particles
        copyto!(view(getfield(s.particles,name),1:length(data)),data)
    end
    s.particles.n=length(d.particles[:x])
    s.volume0=d.volume0;s.t=d.time;s.steps=d.steps;s.iterations=copy(d.iterations)
    a=d.white
    w=Whitewater(a.lo,a.hi,a.h,backend;floor=a.floor,g=a.g,rules=a.rules,stream=a.stream,capacity=length(a.life))
    for name in (:x,:v,:life,:kind,:head)
        copyto!(getfield(w,name),getfield(a,name))
    end
    w.frames=a.frames
    return (;simulation=s,whitewater=w)
end
