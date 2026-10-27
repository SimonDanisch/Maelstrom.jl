# ─── whitewater: spray, foam and bubbles ─────────────────────────────────────
#
# Ihmsen, Akinci, Akinci & Teschner 2012, "Unified spray, foam and air bubbles for
# particle-based fluids". Every fluid particle may make diffuse particles, as many as
# the air it traps (neighbours moving apart across its line to them) and the crest it
# sits on (a convex surface moving outwards) say, times how fast it moves. A diffuse
# particle is spray, foam or a bubble by how much water is round it: spray flies under
# gravity, foam rides the water's surface and fades, a bubble rises and is dragged
# along by the water.

"""
    WhitewaterRules(; ...)

Ihmsen et al.'s parameters, in this basin's units (metres, seconds, velocities in m/s):
the ranges the trapped-air (`air`), wave-crest (`crest`) and kinetic-energy (`energy`,
per unit mass) potentials are clamped and scaled to [0, 1] over; how many diffuse
particles a fluid particle at full potential makes per second (`kair`, `kcrest`); the
fluid neighbour counts below which a diffuse particle is spray and above which it is a
bubble; the lifetime range of foam and bubbles (a bubble the water holds under rises
or dissolves in seconds; held down in a vortex's downwelling, ageless ones piled up
by the million); the bubbles' buoyancy and drag; `within`, how far from the vortex's
axis any is made or kept (none where the frames are not drawn); and `kick`, the speed
(m/s, at full kinetic-energy potential, varied by up to half either way) the particles a
wave crest makes are thrown off it with, along its normal and from just outside it, as
VFX whitewater launches spray, and `tossed`, the share of them thrown (the rest stay foam
on the crest). At `kick = 0` every new particle keeps the water's velocity inside it,
and is spray only where the water itself comes apart.
"""
Base.@kwdef struct WhitewaterRules
    air::NTuple{2,Float32} = (5f0, 20f0)
    crest::NTuple{2,Float32} = (2f0, 8f0)
    energy::NTuple{2,Float32} = (2f0, 20f0)
    kair::Float32 = 40f0
    kcrest::Float32 = 80f0
    spray::Int32 = 6
    bubble::Int32 = 20
    life::NTuple{2,Float32} = (2f0, 8f0)
    buoyancy::Float32 = 2f0
    drag::Float32 = 0.5f0
    kick::Float32 = 0f0
    tossed::Float32 = 1f0
    within::Float32 = Inf32
end

"Ihmsen et al.'s clamp: `I` scaled to [0, 1] over `(lo, hi)`."
@inline clampfn(I, r) = (min(I, r[2]) - min(I, r[1])) / (r[2] - r[1])

"""
    FluidGrid

The fluid particles sorted into cells `h` wide (the reach of every neighbour sum
below), with their velocities in the same order; built on the host every frame.
"""
struct FluidGrid{P,V,I}
    x::P
    v::V
    cellstart::I
    lo::Point3f
    dims::Vec{3,Int32}
    h::Float32
end
Adapt.@adapt_structure FluidGrid

function FluidGrid(u::Matrix{Float32}, v::Matrix{Float32}, h::Float32, lo::Point3f, hi::Point3f, backend)
    dims = Vec{3,Int32}(ceil.(Int32, (hi .- lo) ./ h))
    n = size(u, 2)
    cell = Vector{Int}(undef, n)
    for i in 1:n
        c = cellcoords(lo, h, dims, Point3f(u[1, i], u[2, i], u[3, i]))
        cell[i] = gridcell(dims, c)
    end
    order = sortperm(cell)
    ncells = Int(dims[1]) * Int(dims[2]) * Int(dims[3])
    counts = zeros(Int32, ncells + 1)
    for c in cell
        counts[c + 1] += Int32(1)
    end
    start = cumsum(counts) .+ Int32(1)
    xs = [Point3f(u[1, i], u[2, i], u[3, i]) for i in order]
    vs = [Vec3f(v[1, i], v[2, i], v[3, i]) for i in order]
    dev(a) = (d = KA.allocate(backend, eltype(a), length(a)); copyto!(d, a); d)
    return FluidGrid(dev(xs), dev(vs), dev(start), lo, dims, h)
end

"The cell (coordinates from 0) of `g` the point `x` is in, clamped to the grid."
@inline fluidcell(g::FluidGrid, x) = cellcoords(g.lo, g.h, g.dims, x)

"The fluid particles in cell `n` (coordinates from 0) of `g`, or none if `n` is off the grid."
@inline function cellrange(g::FluidGrid, n)
    (Int32(0) <= n[1] < g.dims[1] && Int32(0) <= n[2] < g.dims[2] && Int32(0) <= n[3] < g.dims[3]) ||
        return Int32(1):Int32(0)
    cell = gridcell(g.dims, n)
    return @inbounds(g.cellstart[cell]):(@inbounds(g.cellstart[cell + 1]) - Int32(1))
end

"""
Each fluid particle's surface normal (the colour field's gradient, pointing out of the
water; zero deep inside, where the neighbours balance) and how many fluid particles are
within reach.
"""
@kernel function normals_kernel!(normal, count, g)
    i = @index(Global)
    x = @inbounds g.x[i]
    c = fluidcell(g, x)
    s = Vec3f(0); k = Int32(0)
    for dk in Int32(-1):Int32(1), dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
        for q in cellrange(g, c + Vec{3,Int32}(di, dj, dk))
            xq = @inbounds g.x[q]
            d = sqrt(sum(abs2, x - xq))
            if d < g.h
                s += (1f0 - d / g.h) * Vec3f(x - xq)
                k += Int32(1)
            end
        end
    end
    l = sqrt(sum(abs2, s))
    @inbounds normal[i] = l > 0.3f0 * g.h ? s / l : Vec3f(0)
    @inbounds count[i] = k
end

"""
The diffuse particles each fluid particle makes this frame (`dt` long), appended at
the ring buffer's head: trapped air and wave crests, scaled by kinetic energy. Only
at the free surface (where the particle has a normal): inside, the vortex's layers
slide past each other, which the trapped-air potential counts as much as water
closing over air, but traps none. None within two reaches of the floor at height
`bed` either: its particles, with no water under them, look like a surface. Crests and
kinetic energy count relative to the `stream` (see [`carried`](@ref)): a wave breaks
running over the current it rides on, and water turning fast round the vortex but
smoothly makes no whitewater.
"""
@kernel function emit_kernel!(wx, wv, wlife, head, g, @Const(normal), dt::Float32, r::WhitewaterRules, seed::UInt32,
                              bed::Float32, stream)
    i = @index(Global)
    x = @inbounds g.x[i]; v = @inbounds g.v[i]; ni = @inbounds normal[i]
    (x[3] < bed + 2f0 * g.h || sum(abs2, ni) == 0f0 || x[1] * x[1] + x[2] * x[2] > r.within * r.within) && return
    vr = v - carried(stream, x)
    relative_speed = sqrt(sum(abs2, vr))
    energy = clampfn(0.5f0 * relative_speed * relative_speed, r.energy)
    emission_draw = hash01(i % Int32, seed % Int32)
    # Exact rejection before the neighbour search: each clamped potential is <=1.
    # This is the same random draw used below, so rejected particles would emit
    # nothing in the original algorithm too. Fine FLIP markers have rates <1/frame.
    upper = energy * r.kair * dt + energy * r.kcrest * dt
    if 0f0 <= upper < 1f0
        # Round conservatively past possible contraction/reassociation differences.
        upper = reinterpret(Float32,reinterpret(UInt32,upper)+UInt32(8))
        emission_draw >= upper && return
    end
    c = fluidcell(g, x)
    air = 0f0; crest = 0f0
    for dk in Int32(-1):Int32(1), dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
        for q in cellrange(g, c + Vec{3,Int32}(di, dj, dk))
            xq = @inbounds g.x[q]
            d = sqrt(sum(abs2, x - xq))
            if 0f0 < d < g.h
                w = 1f0 - d / g.h
                vij = v - @inbounds(g.v[q]); xij = Vec3f(x - xq) / d
                lv = sqrt(sum(abs2, vij))
                lv > 0f0 && (air += lv * (1f0 - (vij / lv) ⋅ xij) * w)
                nq = @inbounds normal[q]
                # Convex: the neighbour lies behind the surface's tangent plane here.
                if sum(abs2, ni) > 0f0 && sum(abs2, nq) > 0f0 && (-xij) ⋅ ni < 0f0
                    crest += (1f0 - ni ⋅ nq) * w
                end
            end
        end
    end
    facing = relative_speed > 0f0 && sum(abs2, ni) > 0f0 && (vr / relative_speed) ⋅ ni >= 0.6f0
    aired = energy * r.kair * clampfn(air, r.air) * dt
    crested = energy * r.kcrest * (facing ? clampfn(crest, r.crest) : 0f0) * dt
    # Whole particles, the fraction by chance.
    whole = unsafe_trunc(Int32, floor(aired + crested))
    whole += emission_draw < aired + crested - Float32(whole) ? Int32(1) : Int32(0)
    # Of them, the crest's share: with a `kick` they are spray, thrown off the crest.
    n̂ = ni / sqrt(sum(abs2, ni))
    cap = Int32(length(wlife))
    for m in Int32(1):whole
        slot = mod(KA.@atomic(:monotonic, head[1] += Int32(1)), cap) + Int32(1)
        a = hash01(i % Int32 + m * Int32(7919), seed % Int32 + Int32(31))
        bb = hash01(i % Int32 - m * Int32(104729), seed % Int32 + Int32(77))
        cc = hash01(m, i % Int32 + seed % Int32)
        jitter = Vec3f(a - 0.5f0, bb - 0.5f0, cc - 0.5f0) * (0.5f0 * g.h)
        u = hash01(i % Int32 + m * Int32(613), seed % Int32 + Int32(5))
        thrown = r.kick > 0f0 && hash01(i % Int32 - m * Int32(331), seed % Int32 + Int32(9)) * (aired + crested) < r.tossed * crested
        if thrown
            # Out of the water, a reach beyond the crest, so it is spray from its first step
            # (in the water it would be foam, and foam takes the water's velocity at once),
            # leaving along the crest's normal as fast as the water is energetic.
            @inbounds wx[slot] = x + g.h * n̂ + 0.5f0 * jitter
            @inbounds wv[slot] = v + (r.kick * energy * (0.5f0 + u)) * n̂
        else
            @inbounds wx[slot] = x + jitter
            @inbounds wv[slot] = v
        end
        @inbounds wlife[slot] = r.life[1] + (r.life[2] - r.life[1]) * hash01(seed % Int32 - m, i % Int32)
    end
end

"""
Every diffuse particle a step `dt` on, by what it is: spray (few fluid neighbours)
falls freely, foam is carried by the water and ages, a bubble (many) rises, is
dragged towards the water's velocity and ages too. Gone when its life is spent, when it
leaves the basin's water downwards, or past the rules' `within` of the axis.
"""
@kernel function diffuse_kernel!(wx, wv, wlife, wkind, g, dt::Float32, r::WhitewaterRules, gravity::Float32, floor::Float32)
    i = @index(Global)
    life = @inbounds wlife[i]
    if life > 0f0
        x = @inbounds wx[i]; v = @inbounds wv[i]
        c = fluidcell(g, x)
        sv = Vec3f(0); sw = 0f0; k = Int32(0)
        for dk in Int32(-1):Int32(1), dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
            for q in cellrange(g, c + Vec{3,Int32}(di, dj, dk))
                d = sqrt(sum(abs2, x - @inbounds(g.x[q])))
                if d < g.h
                    w = 1f0 - d / g.h
                    sv += w * @inbounds(g.v[q]); sw += w; k += Int32(1)
                end
            end
        end
        vf = sw > 0f0 ? sv / sw : Vec3f(0)
        if k < r.spray
            v = v + Vec3f(0, 0, -gravity * dt)
            @inbounds wkind[i] = SPRAY
        elseif k > r.bubble
            v = v + dt * Vec3f(0, 0, r.buoyancy * gravity) + r.drag * (vf - v)
            life -= dt
            @inbounds wkind[i] = BUBBLE
        else
            v = vf
            life -= dt
            @inbounds wkind[i] = FOAM
        end
        x = x + dt * v
        (x[3] < floor || x[1] * x[1] + x[2] * x[2] > r.within * r.within) && (life = 0f0)
        @inbounds wx[i] = x; @inbounds wv[i] = v; @inbounds wlife[i] = life
    end
end

"""
    Whitewater(lo, hi, h, backend; floor = lo[3], g = 9.81, capacity = 4_000_000, rules = WhitewaterRules())

The diffuse particles of a fluid in the box `lo`–`hi`, whose particles are found within
`h` of each other (the reach of every sum over neighbours), a ring buffer of `capacity`
on `backend`. A diffuse particle that falls below `floor` is gone.
"""
mutable struct Whitewater{P,V,F,K,H,S}
    x::P
    v::V
    life::F
    kind::K
    head::H
    rules::WhitewaterRules
    lo::Point3f
    hi::Point3f
    h::Float32
    floor::Float32
    g::Float32
    frames::Int
    stream::S
end

function Whitewater(lo, hi, h, backend; floor = lo[3], g = 9.81f0, capacity = 4_000_000, rules = WhitewaterRules(),
                    stream = nothing)
    z(T) = fill!(KA.allocate(backend, T, capacity), zero(T))
    head = KA.zeros(backend, Int32, 1)
    return Whitewater(z(Point3f), z(Vec3f), z(Float32), z(UInt8), head, rules, Point3f(lo), Point3f(hi), Float32(h),
                      Float32(floor), Float32(g), 0, stream)
end

"The basin's whitewater: its box, reach twice the particles' spacing, gone below the floor, riding the vortex."
function Whitewater(b::Basin, backend; kw...)
    h = 2f0 * b.spacing
    L = b.layers * b.spacing
    lo = Point3f(-b.radius - L - h, -b.radius - L - h, -b.depth - L - h)
    hi = Point3f(b.radius + L + h, b.radius + L + h, b.headroom)
    return Whitewater(lo, hi, h, backend; floor = -b.depth, g = b.g, stream = b, kw...)
end

"""
    carried(stream, x) -> Vec3f

The steady flow at `x` the water's waves ride on: none, or a basin's vortex, its swirl.
"""
@inline carried(::Nothing, x) = Vec3f(0)
@inline function carried(b::Basin, x)
    r = sqrt(x[1] * x[1] + x[2] * x[2])
    turn = r > 1f-4 ? swirl(b, r) / r : 0f0
    return Vec3f(-turn * x[2], turn * x[1], 0f0)
end

"""
    whitewater!(w, u, v, dt; substeps = 4) -> w

Make the diffuse particles the fluid at `u`, `v` (device, `3 × N` / `≥3 × N`) makes over
the frame `dt`, and move all of them through it.
"""
function whitewater!(w::Whitewater, u, v, dt; substeps = 4)
    backend = KA.get_backend(w.life)
    g = FluidGrid(Array(u), Array(view(v, 1:3, :)), w.h, w.lo, w.hi, backend)
    n = length(g.x)
    normal = KA.allocate(backend, Vec3f, n); count = KA.allocate(backend, Int32, n)
    normals_kernel!(backend)(normal, count, g; ndrange = n)
    w.frames += 1
    emit_kernel!(backend)(w.x, w.v, w.life, w.head, g, normal, Float32(dt), w.rules, UInt32(w.frames), w.floor, w.stream;
                          ndrange = n)
    for _ in 1:substeps
        diffuse_kernel!(backend)(w.x, w.v, w.life, w.kind, g, Float32(dt / substeps), w.rules, w.g, w.floor;
                                 ndrange = length(w.life))
    end
    return w
end

"The whitewater a frame of the fluid `s` makes: from its particles `u`, `v` alone, unless the fluid is one with more to go by."
whitewater!(w::Whitewater, s, u, v, dt) = whitewater!(w, u, v, dt)

"""
    WhitewaterFrames(path, b, backend; fps = 24, kw...)

The sink that runs the basin's whitewater ([`Whitewater`](@ref)) along with the water,
a frame at a time, and writes each frame's live diffuse particles to `path.whitewater`
as Float16 `(x, y, z, kind)`, `4 × N` one frame after the other, their velocities to
`path.whitewater.vel` (`3 × N`, the spray's droplets are drawn streaked along them), and
how many each frame has to `path.whitewater.jls` when closed.
"""
struct WhitewaterFrames{W<:Whitewater,H<:AbstractMatrix{Float16},C<:AbstractVector{Int32}}
    w::W
    io::IOStream
    vio::IOStream
    path::String
    counts::Vector{Int}
    fps::Float64
    out::H      # a chunk of the live particles, packed on the device (see `packwhite_kernel!`)
    vel::H
    count::C
end
WhitewaterFrames(path::AbstractString, b::Basin, backend; fps = 24, chunk = 2_000_000, kw...) =
    WhitewaterFrames(Whitewater(b, backend; kw...), open(path * ".whitewater", "w"), open(path * ".whitewater.vel", "w"),
                     String(path), Int[], Float64(fps), KA.allocate(backend, Float16, 4, chunk),
                     KA.allocate(backend, Float16, 3, chunk), KA.zeros(backend, Int32, 1))

"""
    packed!(pack!, streams, count, n) -> how many

Write the chosen ones of `n` elements, packed on the device a chunk at a time:
`pack!(first, len)` packs those among `first:first+len-1` into the device matrices of
`streams` (`io => matrix`, as many columns as a chunk holds), counting them in `count`;
the packed columns go to the host and into the streams. Only what is written crosses
to the host.
"""
function packed!(pack!, streams, count, n)
    chunk = size(last(first(streams)), 2)
    total = 0
    for start in 1:chunk:n
        len = min(chunk, n - start + 1)
        fill!(count, Int32(0))
        pack!(start, len)
        m = Int(Array(count)[1])
        m > 0 && foreach(((io, M),) -> write(io, Array(view(M, :, 1:m))), streams)
        total += m
    end
    return total
end

"The live diffuse particles among `first:first+n-1`, each at the next free column: `(x, y, z, kind)` into `out`, its velocity into `vel`, as Float16."
@kernel inbounds=true function packwhite_kernel!(out, vel, count, @Const(x), @Const(v), @Const(life), @Const(kind), first::Int32,
                                                 n::Int32)
    t = @index(Global)
    if t <= n
        i = Int(first) + t - 1
        if life[i] > 0f0
            m = KA.@atomic :monotonic count[1] += Int32(1)
            p = x[i]; u = v[i]
            out[1, m] = Float16(p[1]); out[2, m] = Float16(p[2]); out[3, m] = Float16(p[3]); out[4, m] = Float16(kind[i])
            vel[1, m] = Float16(u[1]); vel[2, m] = Float16(u[2]); vel[3, m] = Float16(u[3])
        end
    end
end

function record!(k::WhitewaterFrames, s, u, v)
    whitewater!(k.w, s, u, v, 1 / k.fps)
    w = k.w
    backend = KA.get_backend(w.life)
    alive = packed!((k.io => k.out, k.vio => k.vel), k.count, length(w.life)) do first, n
        packwhite_kernel!(backend, 256)(k.out, k.vel, k.count, w.x, w.v, w.life, w.kind, Int32(first), Int32(n); ndrange = n)
    end
    push!(k.counts, alive)
    return k
end

function Base.close(k::WhitewaterFrames)
    close(k.io); close(k.vio)
    serialize(k.path * ".whitewater.jls", (; counts = k.counts, velocities = true))
end

"""
    rewhite(path, b, backend; fps = 24, rules = WhitewaterRules(), kw...) -> path

Make a bake's whitewater again, with other `rules`, from its particles' positions and
velocities (a [`Particles`](@ref) sink with `velocities = true`): Ihmsen et al.'s
diffuse particles do not act on the water, so they can be run after it. Rewrites
`path.whitewater`; `kw` go to [`Whitewater`](@ref) (`capacity`, say).
"""
function rewhite(path::AbstractString, b::Basin, backend; fps = 24, rules = WhitewaterRules(), kw...)
    pf = ParticleFrames(path)
    sink = WhitewaterFrames(path, b, backend; fps, rules, kw...)
    for k in 1:length(pf)
        record!(sink, nothing, Float32.(frame(pf, k)), Float32.(velocity(pf, k)))
    end
    close(sink)
    return path
end

"""
Ihmsen et al.'s three potentials of every fluid particle, as `emit_kernel!` computes
them, for setting the ranges they are clamped over: trapped air, wave crest (zero
where the particle does not move along its surface normal) and kinetic energy per
unit mass.
"""
@kernel function potentials_kernel!(ta, wc, ek, g, @Const(normal), stream)
    i = @index(Global)
    x = @inbounds g.x[i]; v = @inbounds g.v[i]; ni = @inbounds normal[i]
    c = fluidcell(g, x)
    air = 0f0; crest = 0f0
    for dk in Int32(-1):Int32(1), dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
        for q in cellrange(g, c + Vec{3,Int32}(di, dj, dk))
            xq = @inbounds g.x[q]
            d = sqrt(sum(abs2, x - xq))
            if 0f0 < d < g.h
                w = 1f0 - d / g.h
                vij = v - @inbounds(g.v[q]); xij = Vec3f(x - xq) / d
                lv = sqrt(sum(abs2, vij))
                lv > 0f0 && (air += lv * (1f0 - (vij / lv) ⋅ xij) * w)
                nq = @inbounds normal[q]
                if sum(abs2, ni) > 0f0 && sum(abs2, nq) > 0f0 && (-xij) ⋅ ni < 0f0
                    crest += (1f0 - ni ⋅ nq) * w
                end
            end
        end
    end
    vr = v - carried(stream, x)
    lv = sqrt(sum(abs2, vr))
    facing = lv > 0f0 && sum(abs2, ni) > 0f0 && (vr / lv) ⋅ ni >= 0.6f0
    @inbounds ta[i] = air; @inbounds wc[i] = facing ? crest : 0f0; @inbounds ek[i] = 0.5f0 * lv * lv
end

"""
    potentials(u, v, h, lo, hi, backend; stream = nothing) -> (; air, crest, energy, count, surface)

The three potentials of the fluid at `u`, `v` (host, `3 × N`), each particle's
neighbour count and whether it is at the free surface, on the host.
"""
function potentials(u::Matrix{Float32}, v::Matrix{Float32}, h, lo, hi, backend; stream = nothing)
    g = FluidGrid(u, v, Float32(h), Point3f(lo), Point3f(hi), backend)
    n = length(g.x)
    normal = KA.allocate(backend, Vec3f, n); count = KA.allocate(backend, Int32, n)
    normals_kernel!(backend)(normal, count, g; ndrange = n)
    ta = KA.allocate(backend, Float32, n); wc = similar(ta); ek = similar(ta)
    potentials_kernel!(backend)(ta, wc, ek, g, normal, stream; ndrange = n)
    return (; air = Array(ta), crest = Array(wc), energy = Array(ek), count = Array(count),
            surface = map(n -> sum(abs2, n) > 0f0, Array(normal)))
end

@inline foam_seed(::Nothing,p)=p
@inline foam_direction(::Nothing,p)=Vec3f(0)
