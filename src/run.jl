# Running a sparse FLIP basin through film frames: advance it, keep its whitewater
# live, log its health and save the frames asked for (`advance_wild!`, `snapshot_wild`).

"The same basin metadata with the actual marker spacing of this simulation."
wild_drawn_basin(s) = Basin(; (f => (f==:spacing ? s.grid.dx/2 : getfield(s.basin,f)) for f in fieldnames(Basin))...)

"Save the current water and diffuse particles, without advancing either state."
function snapshot_wild(s, white, path)
    mkpath(dirname(path))
    b, be = wild_drawn_basin(s), s.backend
    particles = BandParticles(path,b,be;keep=b.depth+1f0,throat=b.radius)
    levels = LevelSets(path,s)
    foam = WhitewaterFrames(white,open(path*".whitewater","w"),open(path*".whitewater.vel","w"),
        String(path),Int[],24.0,KA.allocate(be,Float16,4,2_000_000),
        KA.allocate(be,Float16,3,2_000_000),KA.zeros(be,Int32,1))
    try
        record!(particles,s,nothing,nothing)
        record!(levels,s,nothing,nothing)
        alive = packed!((foam.io=>foam.out,foam.vio=>foam.vel),foam.count,length(white.life)) do first,n
            packwhite_kernel!(be,256)(foam.out,foam.vel,foam.count,white.x,white.v,white.life,white.kind,
                Int32(first),Int32(n);ndrange=n)
        end
        push!(foam.counts,alive)
    finally
        close(particles); close(levels); close(foam)
    end
    serialize(path*".basin.jls",(;basin=b,fps=24.0,from=s.t,frames=1,waves=Array(s.stir.base.waves),
        viscosity="flip",solver=:flip,dx=s.grid.dx,band=Float32(s.band*s.grid.dx),kernel=2.5f0,
        race=s.stir.base.race,surface_relative_stirring=true,cfl=s.cfl,steps=s.steps))
    return path
end

"Advance a live simulation, keep diagnostics, and save only requested simulation times."
function advance_wild!(s, white, root; tend, snapshots=(6,8,10,12), fps=24,on_snapshot=nothing,stopfile=nothing,
                       whitewater_update=whitewater!)
    mkpath(root)
    rows = []
    clock = time()
    firstframe = floor(Int,s.t*fps+1e-6)+1
    for frame in firstframe:round(Int,tend*fps)
        target = frame/fps
        while s.t < target-1e-9
            left = target-s.t
            dt = left/ceil(left/(s.cfl*s.grid.dx/max(maxspeed(s),1f-3)))
            step!(s,dt)
        end
        s.t = target
        whitewater_update(white,s,nothing,nothing,1/fps)
        if frame%fps==0
            row=(;time=s.t,steps=s.steps,particles=s.particles.n,blocks=s.nslots,
                speed=maxspeed(s),volume=volume(s)/s.volume0,iterations=last(s.iterations),seconds=time()-clock)
            push!(rows,row)
            println(row); flush(stdout)
            open(io->JSON.print(io,rows,2),joinpath(root,"simulation_progress.json"),"w")
            isfinite(row.speed) && .95<row.volume<1.05 || error("Simulation health check failed: $row")
        end
        if any(t->isapprox(target,t;atol=1e-6),snapshots)
            path=joinpath(root,"frame$(lpad(round(Int,target*fps),4,'0'))","whirlpool")
            snapshot_wild(s,white,path)
            println((snapshot=path,time=target)); flush(stdout)
            on_snapshot===nothing || on_snapshot(path,s,white)
        end
        stopfile===nothing || !isfile(stopfile) || break
    end
    return rows
end
