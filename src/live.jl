# Whitewater made live while a FLIP basin runs: the Ihmsen model (see `Whitewater`)
# fed every film frame from the solver's own Float32 particles and level set, with no
# intermediate particle files or half-precision conversion.

"The live secondary state of the liquid `s`, its whitewater starting empty now."
function prepare_live_whitewater(s;rules,spacing=.1f0,capacity=8_000_000)
    b=Basin(; (f=>(f==:spacing ? Float32(spacing) : getfield(s.basin,f)) for f in fieldnames(Basin))...)
    white=Whitewater(b,s.backend;rules,capacity)
    white.frames=round(Int,s.t*24)
    return prepare_live_whitewater(s,white;spacing,initial_time=s.t)
end

"""
    prepare_live_whitewater(s, white::Whitewater; spacing = 0.1, initial_time)

The live secondary state continuing `white` with the liquid `s`, both at the same
completed frame (as `load_sparse_checkpoint` restores them); its whitewater has evolved
since `initial_time`.
"""
function prepare_live_whitewater(s,white::Whitewater;spacing=.1f0,initial_time)
    white.frames==round(Int,s.t*24) || error("Liquid and whitewater are at different frames")
    sampler=BandSampler(white.lo,white.hi,spacing,ceil(Int,(spacing/(s.grid.dx/2))^3/2))
    dims=Int.(s.grid.nb .* (BLOCK÷Int32(2)))
    phi=KA.allocate(s.backend,UInt8,dims...)
    cell=2f0*s.grid.dx;level_range=Float32((s.band+2)*s.grid.dx)
    return (;white,sampler,phi,cell,level_range,initial_time,last_time=Ref(s.t))
end

function update_live_whitewater!(state,w,s,u,v,dt)
    w===state.white || error("Live secondary state and output pool differ")
    isapprox(s.t,state.last_time[]+dt;atol=1e-6) || error("Live secondary model requires consecutive liquid frames")
    isapprox(dt,1/24;atol=1e-8) || error("Live secondary model currently records at24fps")
    sparsequantize_kernel!(s.backend,WG3)(state.phi,s.φ,s.topo,state.cell,state.level_range;
        ndrange=size(state.phi))
    ps=s.particles;n=ps.n
    positions=Array(view(ps.x,1:n));lag=Array(view(ps.lag,1:n))
    velocities=hcat(Array(view(ps.vx,1:n)),Array(view(ps.vy,1:n)),Array(view(ps.vz,1:n)))
    x=Matrix{Float32}(undef,3,n);velocity=Matrix{Float32}(undef,3,n)
    for i in 1:n,d in 1:3
        velocity[d,i]=velocities[i,d]
        x[d,i]=positions[i][d]+lag[i]*velocity[d,i]
    end
    fluid=BandFrame(state.sampler,LatticeWater(state.phi,s.grid.lo,state.cell,state.level_range))
    whitewater!(w,fluid,x,velocity,dt)
    emitted=only(Array(w.head))
    emitted<=length(w.life) || error("Live secondary capacity exhausted")
    state.last_time[]=s.t
    return w
end

function write_live_whitewater_provenance(state,s,prefix)
    rules=state.white.rules
    extensions=(".basin.jls",".particles.jls",".particles",".velocities",".levelset.jls",".levelset")
    provenance=(;source=abspath(prefix),time=s.t,fps=24,initial_time=state.initial_time,
        initial_whitewater="empty; continuously evolved from the recorded live initial time",
        model="Ihmsen secondary particles from live liquid motion, not a resolved air-water simulation",
        liquid_particle_spacing=s.grid.dx/2,representative_spacing=state.sampler.spacing,
        representative_minimum_particles=state.sampler.least,secondary_kernel_radius=state.white.h,
        input_precision="synchronized Float32 particle positions and velocities",
        rules=(;(f=>(f==:within && isinf(rules.within) ? "none" : getfield(rules,f))
            for f in fieldnames(WhitewaterRules))...),
        radius_limit=isinf(rules.within) ? "none" : rules.within,
        markers=only(deserialize(prefix*".whitewater.jls").counts),
        source_sha256=Dict(ext=>open(io->bytes2hex(SHA.sha256(io)),prefix*ext) for ext in extensions),
        emitted_total=only(Array(state.white.head)),capacity=length(state.white.life))
    open(io->JSON.print(io,provenance,2),prefix*".replay.json.tmp","w")
    mv(prefix*".replay.json.tmp",prefix*".replay.json";force=true)
    return provenance
end
