# A basin (see `Basin`) as narrow-band FLIP on the GPU through Mantle: the water near its
# surface carried by APIC particles, the deep water by a MAC grid, the pressure solved
# by a multigrid-preconditioned conjugate gradient. The same basin, vortex, stirring
# race, whitewater and bake files as an SPH bake of it (in TrixiParticles), simulated the way
# film water is: particles where the surface is, a grid where it is not, and time steps
# the flow sets, not the speed of sound.
#
# Ferstl, Ando, Wojtan, Westermann & Thuerey 2016, "Narrow band FLIP for liquid
# simulations": particles only within `band` cells of the surface, their velocities used
# within `band - 1`; the grid velocity and the level set advected semi-Lagrangian
# elsewhere; φ = min(φ_grid + Δx, φ_particles); particles added and removed only between
# one cell and `band` cells below the surface, 8 to 16 a cell.
# Jiang, Schroeder, Selle, Teran & Stomakhin 2015, "The affine particle-in-cell method":
# quadratic B-spline transfers carrying each particle's affine velocity, so the vortex
# keeps its angular momentum without FLIP's noise.
# McAdams, Sifakis & Teran 2010, "A parallel multigrid Poisson solver for fluids
# simulation on large grids": cell-centred pressure, voxelized Dirichlet (air) and
# Neumann (solid) cells, coarse cells Dirichlet if any child is, a symmetric V-cycle as
# the preconditioner of CG. The free surface is the ghost-fluid one (Enright & Fedkiw
# 2002): pressure zero where φ is, between the cell centres.
#
# Everything a step does runs on the device.

const SOLID = 0x00
const LIQUID = 0x01
const AIR = 0x02

# ─── the grid ────────────────────────────────────────────────────────────────

"""
    FlipGrid(b::Basin; dx, top = b.rimwall, pad = 2)

The basin on a MAC grid of cells `dx` wide: in x and y the cylinder and `pad` cells
round it, in z from one cell under the floor to at least `top`. The sides are a
multiple of 2^(levels-1), `levels` the multigrid's levels, chosen so the coarsest has
at most 1024 cells (one workgroup solves it). A cell is solid if its centre is outside
the cylinder or under the floor.
"""
struct FlipGrid
    lo::Point3f
    dx::Float32
    n::NTuple{3,Int32}
    radius::Float32
    floor::Float32
    levels::Int32
end

function FlipGrid(b::Basin; dx, top = b.rimwall, pad = 2)
    dx = Float32(dx)
    nxy = ceil(Int, 2 * (b.radius + pad * dx) / dx)
    loz = -b.depth - dx
    nz0 = ceil(Int, (top - loz) / dx)
    levels = 2
    while true
        m = 2^(levels - 1)
        nx = m * cld(nxy, m); nz = m * cld(nz0, m)
        (nx ÷ m)^2 * (nz ÷ m) <= 1024 && return FlipGrid(Point3f(-nx * dx / 2, -nx * dx / 2, loz), dx,
                                                         (Int32(nx), Int32(nx), Int32(nz)), b.radius, -b.depth,
                                                         Int32(levels))
        levels += 1
    end
end

@inline cellcentre(g::FlipGrid, i, j, k) = g.lo + Vec3f(i - 0.5f0, j - 0.5f0, k - 0.5f0) * g.dx
@inline issolid(g::FlipGrid, p) = p[1] * p[1] + p[2] * p[2] > g.radius * g.radius || p[3] < g.floor
@inline issolidcell(g::FlipGrid, i, j, k) = issolid(g, cellcentre(g, i, j, k))

# Where a component's samples sit within a cell, in cells: u on the x faces, v on the y
# faces, w on the z faces, scalars at the centres. Node (a, b, c) of an array sits at
# `lo + ((a, b, c) - 1 + offset) dx`.
const UOFF = Vec3f(0f0, 0.5f0, 0.5f0)
const VOFF = Vec3f(0.5f0, 0f0, 0.5f0)
const WOFF = Vec3f(0.5f0, 0.5f0, 0f0)
const COFF = Vec3f(0.5f0, 0.5f0, 0.5f0)

@inline nodepos(g::FlipGrid, i, j, k, off::Vec3f) = g.lo + (Vec3f(i - 1, j - 1, k - 1) + off) * g.dx
@inline indexspace(g::FlipGrid, p, off::Vec3f) = Vec3f((p - g.lo) / g.dx) - off

"The grid's velocity at `p` (world)."
@inline velocity(u, v, w, g::FlipGrid, p) =
    Vec3f(trilinear(u, indexspace(g, p, UOFF)), trilinear(v, indexspace(g, p, VOFF)), trilinear(w, indexspace(g, p, WOFF)))

# The six neighbours of a cell, `m` in 1:6.
@inline function neighbour(i, j, k, m)
    o = oneunit(i)
    m == 1 && return (i + o, j, k)
    m == 2 && return (i - o, j, k)
    m == 3 && return (i, j + o, k)
    m == 4 && return (i, j - o, k)
    m == 5 && return (i, j, k + o)
    return (i, j, k - o)
end
@inline inrange(A, i, j, k) = 1 <= i <= size(A, 1) && 1 <= j <= size(A, 2) && 1 <= k <= size(A, 3)
@inline typeat(types, i, j, k) = inrange(types, i, j, k) ? @inbounds(types[i, j, k]) : SOLID

# The cell before face (i, j, k) along `axis` (the cell after is (i, j, k) itself).
@inline before(i, j, k, axis) = axis == 1 ? (i - 1, j, k) : axis == 2 ? (i, j - 1, k) : (i, j, k - 1)

"Quadratic B-spline weights at index-space coordinate `q`: the first node (0-based) and its three weights."
@inline function bspline(q::Float32)
    b = floor(q - 0.5f0)
    f = q - b
    return unsafe_trunc(Int, b), 0.5f0 * (1.5f0 - f)^2, 0.75f0 - (f - 1f0)^2, 0.5f0 * (f - 0.5f0)^2
end
@inline pick(m, w0, w1, w2) = m == 0 ? w0 : (m == 1 ? w1 : w2)

# Floats in an order integer atomics keep: the level set the particles splat is a
# minimum taken by `atomic min` on these.
@inline orderkey(f::Float32) = (b = reinterpret(Int32, f); b >= Int32(0) ? b : b ⊻ Int32(0x7fffffff))
@inline orderfloat(k::Int32) = reinterpret(Float32, k >= Int32(0) ? k : k ⊻ Int32(0x7fffffff))

"The free surface's position between a liquid cell (`φi < 0`) and an air one: the fraction of the way across."
@inline theta(φi, φj) = clamp(φi / (φi - φj), 0.01f0, 1f0)

# ─── the state ───────────────────────────────────────────────────────────────

"One coarse level of the multigrid: its cells' types, the correction, the right-hand side and the residual."
struct MGLevel{T,F}
    types::T
    x::F
    b::F
    r::F
end

"""
The particles of the narrow band, as arrays of their position, their velocity's three
components, each component's affine velocity (APIC) and how far the particle's own time
lags behind the step's (`lag`, seconds; see `advectparticles_kernel!`), twice: the
second set is what resampling writes into before the two swap.
"""
mutable struct FlipParticles{P,F,C}
    x::P; vx::F; vy::F; vz::F; cx::C; cy::C; cz::C; lag::F
    x2::P; vx2::F; vy2::F; vz2::F; cx2::C; cy2::C; cz2::C; lag2::F
    n::Int
end

function FlipParticles(backend, capacity)
    P() = KA.allocate(backend, Point3f, capacity)
    F() = KA.zeros(backend, Float32, capacity)
    C() = KA.allocate(backend, Vec3f, capacity)
    return FlipParticles(P(), F(), F(), F(), C(), C(), C(), F(), P(), F(), F(), F(), C(), C(), C(), F(), 0)
end
capacity(p::FlipParticles) = length(p.x)

"Swap the particles and the arrays resampling writes into."
function swap!(p::FlipParticles)
    p.x, p.x2 = p.x2, p.x; p.vx, p.vx2 = p.vx2, p.vx; p.vy, p.vy2 = p.vy2, p.vy; p.vz, p.vz2 = p.vz2, p.vz
    p.cx, p.cx2 = p.cx2, p.cx; p.cy, p.cy2 = p.cy2, p.cy; p.cz, p.cz2 = p.cz2, p.cz
    p.lag, p.lag2 = p.lag2, p.lag
    return p
end

"""
    FlipBasin(b::Basin, backend; dx, stir = Stir(b), band = 4, nmin = 8, cfl = 2, capacity)

The basin as narrow-band FLIP on `backend`: the grid (see [`FlipGrid`](@ref)), its
velocities, level set and pressure, the multigrid's levels, and the particles within
`band` cells of the surface (`nmin` to `2nmin` a cell). Stirred by `stir` (see `Stir`);
time steps move the fastest water `cfl` cells.
"""
mutable struct FlipBasin{B,S,F3,I3,T3,U3,L,P,I1,F1}
    basin::Basin
    grid::FlipGrid
    backend::B
    stir::S
    u::F3; v::F3; w::F3
    u0::F3; v0::F3; w0::F3
    mu::F3; mv::F3; mw::F3
    wu::F3; wv::F3; ww::F3
    φ::F3; φ0::F3; φp::I3
    types::T3; reach::T3
    P::F3; b::F3; r::F3; z::F3; d::F3; q::F3; diag::F3; rv::F3
    fu::U3; fv::U3; fw::U3; fu2::U3; fv2::U3; fw2::U3
    levels::L
    particles::P
    counts::I1; totals::I1; offsets::I1; cellstart::I1; changed::I1
    key::I1; rank::I1
    partials::F1
    cellpartials::F1
    band::Int32
    nmin::Int32
    cfl::Float32
    volume0::Float32
    t::Float64
    steps::Int
    iterations::Vector{Int}
end

function FlipBasin(b::Basin, backend; dx, stir = Stir(b), band = 4, nmin = 8, cfl = 2f0, capacity = 12_000_000)
    g = FlipGrid(b; dx)
    nx, ny, nz = Int.(g.n)
    F(dims...) = KA.zeros(backend, Float32, dims...)
    U(dims...) = KA.zeros(backend, UInt8, dims...)
    ncell = nx * ny * nz
    levels = MGLevel[]
    m = 1
    for l in 2:g.levels
        m *= 2
        dims = (nx ÷ m, ny ÷ m, nz ÷ m)
        push!(levels, MGLevel(U(dims...), F(dims...), F(dims...), F(dims...)))
    end
    levels = [l for l in levels]   # concretely typed
    # The stir's waves as a device array: `Stir` is adapted to the device per launch.
    devstir = Stir(stir.basin, Mantle.devicearray(backend, collect(stir.waves)), stir.origin, stir.offset, stir.race)
    s = FlipBasin(b, g, backend, devstir,
                  F(nx + 1, ny, nz), F(nx, ny + 1, nz), F(nx, ny, nz + 1),
                  F(nx + 1, ny, nz), F(nx, ny + 1, nz), F(nx, ny, nz + 1),
                  F(nx + 1, ny, nz), F(nx, ny + 1, nz), F(nx, ny, nz + 1),
                  F(nx + 1, ny, nz), F(nx, ny + 1, nz), F(nx, ny, nz + 1),
                  F(nx, ny, nz), F(nx, ny, nz), KA.zeros(backend, Int32, nx, ny, nz),
                  U(nx, ny, nz), U(nx, ny, nz),
                  F(nx, ny, nz), F(nx, ny, nz), F(nx, ny, nz), F(nx, ny, nz), F(nx, ny, nz), F(nx, ny, nz),
                  F(nx, ny, nz), F(nx, ny, nz),
                  U(nx + 1, ny, nz), U(nx, ny + 1, nz), U(nx, ny, nz + 1),
                  U(nx + 1, ny, nz), U(nx, ny + 1, nz), U(nx, ny, nz + 1),
                  levels, FlipParticles(backend, capacity),
                  KA.zeros(backend, Int32, ncell), KA.zeros(backend, Int32, ncell), KA.zeros(backend, Int32, ncell),
                  KA.zeros(backend, Int32, ncell + 1), KA.zeros(backend, Int32, 1),
                  KA.zeros(backend, Int32, capacity), KA.zeros(backend, Int32, capacity),
                  KA.zeros(backend, Float32, cld(maximum(prod, ((nx + 1, ny, nz), (nx, ny + 1, nz), (nx, ny, nz + 1))), 256)),
                  KA.zeros(backend, Float32, cellgroups((nx, ny, nz))),
                  Int32(band), Int32(nmin), Float32(cfl), 0f0, 0.0, 0, Int[])
    return s
end

cells(s::FlipBasin) = size(s.φ)
spacing(s::FlipBasin) = s.grid.dx
const WG3 = (8, 8, 4)

# ─── reductions ──────────────────────────────────────────────────────────────

@kernel function dot_kernel!(partials, @Const(a), @Const(b))
    i = @index(Global); li = @index(Local); gi = @index(Group)
    s = @localmem Float32 (256,)
    @inbounds s[li] = i <= length(a) ? a[i] * b[i] : 0f0
    @synchronize
    step = 128
    while step >= 1
        li <= step && (@inbounds s[li] += s[li + step])
        @synchronize
        step ÷= 2
    end
    li == 1 && (@inbounds partials[gi] = s[1])
end

@kernel function maxabs_kernel!(partials, @Const(a))
    i = @index(Global); li = @index(Local); gi = @index(Group)
    s = @localmem Float32 (256,)
    @inbounds s[li] = i <= length(a) ? abs(a[i]) : 0f0
    @synchronize
    step = 128
    while step >= 1
        li <= step && (@inbounds s[li] = max(s[li], s[li + step]))
        @synchronize
        step ÷= 2
    end
    li == 1 && (@inbounds partials[gi] = s[1])
end

# The water's volume: each non-solid cell counts by how much of it is under the surface.
@kernel function volume_kernel!(partials, @Const(φ), g::FlipGrid)
    i = @index(Global); li = @index(Local); gi = @index(Group)
    s = @localmem Float32 (256,)
    v = 0f0
    if i <= length(φ)
        n1, n2 = size(φ, 1), size(φ, 2)
        a = (i - 1) % n1 + 1; b = ((i - 1) ÷ n1) % n2 + 1; c = (i - 1) ÷ (n1 * n2) + 1
        issolidcell(g, a, b, c) || (v = clamp(0.5f0 - @inbounds(φ[i]) / g.dx, 0f0, 1f0))
    end
    @inbounds s[li] = v
    @synchronize
    step = 128
    while step >= 1
        li <= step && (@inbounds s[li] += s[li + step])
        @synchronize
        step ÷= 2
    end
    li == 1 && (@inbounds partials[gi] = s[1])
end

# Every reduction runs over all of `partials` (sized for the largest array): groups
# past the array's end write zero, so the sum or maximum is one call on the whole.
reduction(kernel!, s::FlipBasin, args...) =
    kernel!(s.backend, 256)(s.partials, args...; ndrange = 256 * length(s.partials))

"The dot product of two of the grid's arrays."
inner(s::FlipBasin, a, b) = (reduction(dot_kernel!, s, a, b); sum(s.partials))

"The largest magnitude in one of the grid's arrays."
maxabs(s::FlipBasin, a) = (reduction(maxabs_kernel!, s, a); maximum(s.partials))

"The water's volume, in cubic metres."
volume(s::FlipBasin) = (reduction(volume_kernel!, s, s.φ, s.grid); sum(s.partials) * s.grid.dx^3)

# ─── the start: the balanced vortex ──────────────────────────────────────────

@kernel inbounds=true function initphi_kernel!(φ, @Const(η), g::FlipGrid, rmax::Float32)
    i, j, k = @index(Global, NTuple)
    p = cellcentre(g, i, j, k)
    r = sqrt(p[1] * p[1] + p[2] * p[2])
    t = clamp(r / rmax, 0f0, 1f0) * Float32(length(η) - 1)
    a = min(unsafe_trunc(Int, t), length(η) - 2)
    f = t - a
    h = η[a + 1] * (1f0 - f) + η[a + 2] * f
    φ[i, j, k] = p[3] - h
end

@kernel inbounds=true function initvelocity_kernel!(u, g::FlipGrid, Γ::Float32, core::Float32, off::Vec3f, axis::Int32)
    i, j, k = @index(Global, NTuple)
    p = nodepos(g, i, j, k, off)
    r = sqrt(p[1] * p[1] + p[2] * p[2])
    turn = r > 1f-4 ? oseen(Γ, core, r) / r : 0f0
    u[i, j, k] = axis == 1 ? -turn * p[2] : (axis == 2 ? turn * p[1] : 0f0)
end

"""
    initialize!(s::FlipBasin)

The basin's water at rest in its vortex: the balanced surface as the level set, the
swirl on every face, and the narrow band filled with particles carrying it.
"""
function initialize!(s::FlipBasin)
    b, g, be = s.basin, s.grid, s.backend
    rmax = hypot(g.lo[1], g.lo[2])
    η = Float32[surface(b, r) for r in range(0f0, rmax; length = 2048)]
    initphi_kernel!(be, WG3)(s.φ, Mantle.devicearray(be, η), g, Float32(rmax); ndrange = cells(s))
    for (a, off, arr) in ((1, UOFF, s.u), (2, VOFF, s.v), (3, WOFF, s.w))
        initvelocity_kernel!(be, WG3)(arr, g, b.Γ, b.core, off, Int32(a); ndrange = size(arr))
    end
    redistance!(s; iterations = 2s.band + 4)
    classify!(s)
    solidify!(s)
    # The vortex's velocity is known everywhere: every face counts as reached.
    foreach(f -> fill!(f, 0x01), (s.fu, s.fv, s.fw))
    s.particles.n = 0
    resample!(s; seeding = true)
    # Again, which drops the seeds the level set put in the air.
    resample!(s)
    # The level set and velocity a step keeps: the particles' where they are, their
    # volume the one to keep.
    copyto!(s.φ0, s.φ); copyto!(s.u0, s.u); copyto!(s.v0, s.v); copyto!(s.w0, s.w)
    particles2grid!(s, 1f0)
    redistance!(s)
    classify!(s)
    s.volume0 = volume(s)
    s.t = 0.0
    return s
end

# ─── advection ───────────────────────────────────────────────────────────────

"""
Semi-Lagrangian advection of the array `src` (samples at `off`) through the grid's
velocity: traced back in RK2 sub-steps of at most a cell, so a step may move the water
many cells (Braun et al. 2026 sub-step the same way).
"""
@kernel inbounds=true function advect_kernel!(out, @Const(src), @Const(u), @Const(v), @Const(w), g::FlipGrid, off::Vec3f,
                                              dt::Float32)
    i, j, k = @index(Global, NTuple)
    p = nodepos(g, i, j, k, off)
    m = substeps(sqrt(sum(abs2, velocity(u, v, w, g, p))) * dt / g.dx)
    h = dt / m
    for _ in 1:m
        v1 = velocity(u, v, w, g, p)
        v2 = velocity(u, v, w, g, p - 0.5f0 * h * v1)
        p = p - h * v2
    end
    out[i, j, k] = trilinear(src, indexspace(g, p, off))
end

"How many sub-steps of at most a cell a move of `cells` cells takes (at most 32)."
@inline substeps(cells) = clamp(unsafe_trunc(Int32, ceil(cells)), Int32(1), Int32(32))

"Whether the grid's velocity was carried out to `p`: every face the trilinear sample at `p` reads was projected or extrapolated."
@inline reached(fu, fv, fw, g::FlipGrid, p) =
    min(trilinear(fu, indexspace(g, p, UOFF)), trilinear(fv, indexspace(g, p, VOFF)), trilinear(fw, indexspace(g, p, WOFF))) >
    0.999f0

"`p` moved back inside the basin's wall and above its floor, a quarter cell in, if it left them."
@inline function inwalls(g::FlipGrid, p)
    m = 0.25f0 * g.dx
    r = sqrt(p[1] * p[1] + p[2] * p[2])
    r > g.radius - m && (p = Point3f(p[1] * (g.radius - m) / r, p[2] * (g.radius - m) / r, p[3]))
    p[3] < g.floor + m && (p = Point3f(p[1], p[2], g.floor + m))
    return p
end

"""
The velocity the particles' kernels read: the grid (a `FlipGrid`, or a sparse grid's
`Topology`), its face velocities, and the flags of the faces the velocity was carried
out to (projected or extrapolated).
"""
struct Flow{G,U,F}
    g::G
    u::U; v::U; w::U
    fu::F; fv::F; fw::F
end
Adapt.@adapt_structure Flow

@inline faceoff(a) = a == 1 ? UOFF : (a == 2 ? VOFF : WOFF)
@inline spacing(g::FlipGrid) = g.dx
@inline origin(g::FlipGrid) = g.lo
@inline velocity(f::Flow{FlipGrid}, p) = velocity(f.u, f.v, f.w, f.g, p)
@inline reached(f::Flow{FlipGrid}, p) = reached(f.fu, f.fv, f.fw, f.g, p)
@inline inwalls(f::Flow, p) = inwalls(f.g, p)
@inline component(f::Flow, a) = a == 1 ? f.u : (a == 2 ? f.v : f.w)
@inline flag(f::Flow, a) = a == 1 ? f.fu : (a == 2 ? f.fv : f.fw)

"APIC's gather of component `a` at `xp`: the velocity, its affine part, and whether every face it read was reached."
@inline function gathered(f::Flow{FlipGrid}, a, xp)
    g = f.g
    vel, aff = gather(component(f, a), xp, g.lo, g.dx, faceoff(a))
    return vel, aff, gather(flag(f, a), xp, g.lo, g.dx, faceoff(a))[1] > 0.999f0
end

"Linear index of node `(I, J, K)` (from 0) of the dense array `A`, or 0 if it is off the array."
@inline function nodeindex(::FlipGrid, A, a, I, J, K)
    n1, n2, n3 = size(A)
    (0 <= I < n1 && 0 <= J < n2 && 0 <= K < n3) || return 0
    return I + n1 * (J + n2 * K) + 1
end
@inline cellindex(g::FlipGrid, A, I, J, K) = nodeindex(g, A, 0, I, J, K)
"A cell's centre from its coordinates from 0."
@inline centre0(g::FlipGrid, I, J, K) = g.lo + (Vec3f(I, J, K) + COFF) * g.dx

"""
One step `dt` of the particles, as spatiotemporal FLIP (Braun et al. 2026, "ST-FLIP")
advances them: each by its own time, `dt` plus the `lag` it carries plus a jitter
`γ ξ dt` (`ξ` uniform in ±½, `γ` the particle's own Courant number eased to 1, so calm
water is not jittered), at most `2dt`; what it did not advance is its new lag. The
particles then sample the whole step in time, which keeps large steps from aliasing the
surface. Along the grid's velocity in RK2 sub-steps of at most a cell; where the
velocity was not carried out to (air past the extrapolation), on its own velocity under
gravity, as spray flies.
"""
@kernel inbounds=true function advectparticles_kernel!(x, lag, @Const(vx), @Const(vy), @Const(vz), flow, dt::Float32,
                                                        gravity::Float32, seed::Int32, n::Int32)
    p = @index(Global)
    if p <= n
        y = x[p]
        vb = Vec3f(vx[p], vy[p], vz[p])
        h0 = spacing(flow.g)
        speed = sqrt(sum(abs2, vb))
        γ = strength(speed, dt, h0)
        ξ = hash01(p % Int32, seed) - 0.5f0
        τ = lag[p]
        act = clamp(dt + τ + γ * ξ * dt, 0f0, 2f0 * dt)
        lag[p] = dt + τ - act
        m = substeps(max(speed, sqrt(sum(abs2, velocity(flow, y)))) * act / h0)
        h = act / m
        for _ in 1:m
            if reached(flow, y)
                v1 = velocity(flow, y)
                vb = velocity(flow, y + 0.5f0 * h * v1)
                y = y + h * vb
            else
                y = y + h * (vb - Vec3f(0f0, 0f0, 0.5f0 * gravity * h))
                vb = vb - Vec3f(0f0, 0f0, gravity * h)
            end
            y = inwalls(flow, y)
        end
        x[p] = y
    end
end

"ST-FLIP's jitter strength of a particle at `speed`: its Courant number on a grid `h` apart, eased to 1 (Braun et al. 2026, Algorithm 1, line 25)."
@inline function strength(speed, dt, h)
    c = clamp(speed * dt / h, 0f0, 1f0)
    return c * c * (3f0 - 2f0 * c)
end

"""
    ahead(γ) -> steps

The mean time, past the step's end, that a particle of jitter strength `γ` deposits
for: its lag is spread evenly over ±γ/2 steps and weighted by `temporal`, which favours
the latest samples. That is 0.227 steps at full jitter and γ²/3 for small γ. The
momentum on the grid stands for this time, so the surface the pressure solve sees has
to as well. A surface at the step's end trails the momentum, and at large steps the
solve pumped flow-aligned ridges into it. Braun et al. get this consistency from their
phase field, which comes from the same weighted deposits.
"""
@inline function ahead(γ)
    h2 = 0.25f0 * γ * γ
    return h2 * (9f0 / 16 - h2 * (7f0 / 10 - h2 * 3f0 / 7)) / (27f0 / 64 + h2 * (3f0 / 16 - h2 * (3f0 / 20 + h2 / 7f0)))
end

Flow(s::FlipBasin) = Flow(s.grid, s.u, s.v, s.w, s.fu, s.fv, s.fw)

function advect!(s::FlipBasin, dt)
    g, be, ps = s.grid, s.backend, s.particles
    n = ps.n
    n > 0 && advectparticles_kernel!(be, 256)(ps.x, ps.lag, ps.vx, ps.vy, ps.vz, Flow(s), Float32(dt), s.basin.g,
                                              Int32(s.steps), Int32(n); ndrange = n)
    for (src, out, off) in ((s.u, s.u0, UOFF), (s.v, s.v0, VOFF), (s.w, s.w0, WOFF), (s.φ, s.φ0, COFF))
        advect_kernel!(be, WG3)(out, src, s.u, s.v, s.w, g, off, Float32(dt); ndrange = size(out))
    end
    return s
end

# ─── particles to grid ───────────────────────────────────────────────────────

"""
ST-FLIP's temporal weight (Braun et al. 2026, eq. 19) of a particle `lag` behind a step
`dt` long: the past half of a poly6 kernel centred on the step's end, so the latest
samples weigh most, 1 on average over the step.
"""
@inline function temporal(lag, dt)
    r = clamp(-lag / dt, -0.5f0, 0.5f0) - 0.5f0
    return (35f0 / 16f0) * (1f0 - r * r)^3
end

"""
APIC: each particle's momentum along component `a`, and its weight, onto that
component's faces on the grid `g` (a `FlipGrid`, or a sparse grid's `Topology`, whose
fine faces are the ones particles reach), weighted in time too (see `temporal`).
"""
@kernel inbounds=true function p2g_kernel!(mom, wt, @Const(x), @Const(va), @Const(ca), @Const(lag), g, a::Int32, dt::Float32,
                                           n::Int32)
    p = @index(Global)
    if p <= n
        xp = x[p]; vp = va[p]; cp = ca[p]
        wT = temporal(lag[p], dt)
        dx = spacing(g)
        q = Vec3f((xp - origin(g)) / dx) - faceoff(a)
        bx, x0, x1, x2 = bspline(q[1]); by, y0, y1, y2 = bspline(q[2]); bz, z0, z1, z2 = bspline(q[3])
        for mz in Int32(0):Int32(2), my in Int32(0):Int32(2), mx in Int32(0):Int32(2)
            idx = nodeindex(g, mom, a, Int32(bx) + mx, Int32(by) + my, Int32(bz) + mz)
            if idx > 0
                wgt = wT * pick(mx, x0, x1, x2) * pick(my, y0, y1, y2) * pick(mz, z0, z1, z2)
                d = Vec3f(bx + mx - q[1], by + my - q[2], bz + mz - q[3]) * dx
                KA.@atomic :monotonic mom[idx] += wgt * (vp + cp ⋅ d)
                KA.@atomic :monotonic wt[idx] += wgt
            end
        end
    end
end

"""
The particles' level set: each cell within reach of a particle at most its distance to
it less `radius`. Each particle counts where it is at the time its deposited momentum
stands for: moved on by its lag, then `ahead` of the step's end.
"""
@kernel inbounds=true function splat_kernel!(φp, @Const(x), @Const(lag), @Const(vx), @Const(vy), @Const(vz), g,
                                             radius::Float32, dt::Float32, n::Int32)
    p = @index(Global)
    if p <= n
        vp = Vec3f(vx[p], vy[p], vz[p])
        xp = x[p] + (lag[p] + ahead(strength(sqrt(sum(abs2, vp)), dt, spacing(g))) * dt) * vp
        c = Vec3f((xp - origin(g)) / spacing(g))
        ci = unsafe_trunc(Int32, floor(c[1])); cj = unsafe_trunc(Int32, floor(c[2])); ck = unsafe_trunc(Int32, floor(c[3]))
        for dk in Int32(-1):Int32(1), dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
            idx = cellindex(g, φp, ci + di, cj + dj, ck + dk)
            if idx > 0
                d = sqrt(sum(abs2, centre0(g, ci + di, cj + dj, ck + dk) - xp)) - radius
                KA.@atomic :monotonic φp[idx] min orderkey(d)
            end
        end
    end
end

@kernel inbounds=true function union_kernel!(φ, @Const(φ0), @Const(φp), dx::Float32)
    i = @index(Global)
    φ[i] = min(φ0[i] + dx, orderfloat(φp[i]))
end

"""
Ferstl et al.'s eq. 3: on faces within `r` of the surface the particles' velocity,
elsewhere the grid's. A face the particles cover only partly (a weight under `full`,
the weight a face full of them gets) takes the grid's velocity for the weight they
lack. Taken alone, the few particles that reach such a face set it to their velocity
extrapolated by their affine part as far as a cell and a half: at the band's edges, on
a shear layer, that overshot step after step and drove the water to 45 m/s.
"""
@kernel inbounds=true function combine_kernel!(u, @Const(u0), @Const(mom), @Const(wt), @Const(φ), r::Float32, full::Float32,
                                               axis::Int32)
    i, j, k = @index(Global, NTuple)
    a, b, c = before(i, j, k, axis)
    n1, n2, n3 = size(φ)
    fa = inrange(φ, a, b, c) ? φ[a, b, c] : φ[min(i, n1), min(j, n2), min(k, n3)]
    fb = inrange(φ, i, j, k) ? φ[i, j, k] : fa
    φf = 0.5f0 * (fa + fb)
    wgt = wt[i, j, k]
    u[i, j, k] = φf >= -r ? (mom[i, j, k] + max(full - wgt, 0f0) * u0[i, j, k]) / max(full, wgt) : u0[i, j, k]
end

function particles2grid!(s::FlipBasin, dt)
    g, be, ps = s.grid, s.backend, s.particles
    n = ps.n
    for arr in (s.mu, s.mv, s.mw, s.wu, s.wv, s.ww)
        fill!(arr, 0f0)
    end
    fill!(s.φp, orderkey(1f30))
    if n > 0
        for (mom, wt, va, ca, a) in ((s.mu, s.wu, ps.vx, ps.cx, 1), (s.mv, s.wv, ps.vy, ps.cy, 2), (s.mw, s.ww, ps.vz, ps.cz, 3))
            p2g_kernel!(be, 256)(mom, wt, ps.x, va, ca, ps.lag, g, Int32(a), Float32(dt), Int32(n); ndrange = n)
        end
        splat_kernel!(be, 256)(s.φp, ps.x, ps.lag, ps.vx, ps.vy, ps.vz, g, 0.5f0 * g.dx, Float32(dt), Int32(n); ndrange = n)
    end
    union_kernel!(be, 256)(s.φ, s.φ0, s.φp, g.dx; ndrange = length(s.φ))
    r = Float32((s.band - 1) * g.dx)
    for (u, u0, mom, wt, a) in ((s.u, s.u0, s.mu, s.wu, 1), (s.v, s.v0, s.mv, s.wv, 2), (s.w, s.w0, s.mw, s.ww, 3))
        combine_kernel!(be, WG3)(u, u0, mom, wt, s.φ, r, Float32(s.nmin), Int32(a); ndrange = size(u))
    end
    return s
end

# ─── the level set ───────────────────────────────────────────────────────────

"The distance at which |∇φ| = 1 from the smallest neighbour distances along each axis (Godunov upwind)."
@inline function eikonal(a, b, c, h)
    a, b = min(a, b), max(a, b); b, c = min(b, c), max(b, c); a, b = min(a, b), max(a, b)
    d = a + h
    if d > b
        d = 0.5f0 * (a + b + sqrt(max(2f0 * h * h - (a - b)^2, 0f0)))
        if d > c
            t = a + b + c
            d = (t + sqrt(max(t * t - 3f0 * (a * a + b * b + c * c - h * h), 0f0))) / 3f0
        end
    end
    return d
end

"""
One Jacobi sweep of redistancing: cells next to the surface (a neighbour of the other
sign) keep their value, the others take the upwind distance from their neighbours, at
most `φmax`. A solid cell takes the lowest value of the water next to it, so the walls
are wet where the water touches them.
"""
@kernel inbounds=true function reinit_kernel!(out, @Const(φ), g::FlipGrid, φmax::Float32)
    i, j, k = @index(Global, NTuple)
    c = φ[i, j, k]
    h = g.dx
    if issolidcell(g, i, j, k)
        lowest = c
        found = false
        for m in 1:6
            a, b, cc = neighbour(i, j, k, m)
            if inrange(φ, a, b, cc) && !issolidcell(g, a, b, cc)
                lowest = found ? min(lowest, φ[a, b, cc]) : φ[a, b, cc]
                found = true
            end
        end
        out[i, j, k] = found ? lowest : clamp(c, -φmax, φmax)
    else
        inside = c < 0f0
        frozen = false
        ax = φmax; ay = φmax; az = φmax
        for m in 1:6
            a, b, cc = neighbour(i, j, k, m)
            if inrange(φ, a, b, cc) && !issolidcell(g, a, b, cc)
                f = φ[a, b, cc]
                (f < 0f0) != inside && (frozen = true)
                d = abs(f)
                m <= 2 ? (ax = min(ax, d)) : (m <= 4 ? (ay = min(ay, d)) : (az = min(az, d)))
            end
        end
        if frozen
            out[i, j, k] = c
        else
            d = min(eikonal(ax, ay, az, h), φmax)
            out[i, j, k] = inside ? -d : d
        end
    end
end

"""
    redistance!(s; iterations)

Make φ a distance again near the surface, `iterations` cells deep (beyond, its sign
and at least that distance), by Jacobi sweeps through `φ0`.
"""
function redistance!(s::FlipBasin; iterations = s.band + 3)
    φmax = Float32((iterations + 1) * s.grid.dx)
    for _ in 1:iterations
        reinit_kernel!(s.backend, WG3)(s.φ0, s.φ, s.grid, φmax; ndrange = cells(s))
        s.φ, s.φ0 = s.φ0, s.φ
    end
    return s
end

@kernel inbounds=true function classify_kernel!(types, @Const(φ), g::FlipGrid)
    i, j, k = @index(Global, NTuple)
    types[i, j, k] = issolidcell(g, i, j, k) ? SOLID : (φ[i, j, k] < 0f0 ? LIQUID : AIR)
end
classify!(s::FlipBasin) = (classify_kernel!(s.backend, WG3)(s.types, s.φ, s.grid; ndrange = cells(s)); s)

# ─── trapped air ─────────────────────────────────────────────────────────────
#
# The free surface holds the pressure at zero, the atmosphere's. Air the water has
# closed round is not at it: a pocket three metres down is at the water's pressure
# there, and held at zero it drew the water round it in at 50 m/s. The level set's
# shrinking by a cell a step (Ferstl et al.'s eq. 4) also opens pockets under the
# band, where no particle holds the surface, which then grow a cell a step. So air not
# joined to the atmosphere becomes water; the whitewater's bubbles stand for the air
# the water takes in, as they do in the SPH bake.

@kernel inbounds=true function seedair_kernel!(reach, @Const(types))
    i, j, k = @index(Global, NTuple)
    reach[i, j, k] = (k == size(types, 3) && types[i, j, k] == AIR) ? 0x01 : 0x00
end

"""
Air reached from the atmosphere along one axis: each thread walks its line of cells
there and back, passing reach on through runs of air, and counts the cells it reached.
"""
@kernel inbounds=true function sweepair_kernel!(reach, changed, @Const(types), axis::Int32)
    a, b = @index(Global, NTuple)
    n = size(types, Int(axis))
    count = Int32(0)
    for pass in 0:1
        carry = false
        for m in 1:n
            l = pass == 0 ? m : n + 1 - m
            i, j, k = axis == 1 ? (l, a, b) : (axis == 2 ? (a, l, b) : (a, b, l))
            if types[i, j, k] == AIR
                if reach[i, j, k] != 0x00
                    carry = true
                elseif carry
                    reach[i, j, k] = 0x01
                    count += Int32(1)
                end
            else
                carry = false
            end
        end
    end
    count > Int32(0) && KA.@atomic :monotonic changed[1] += count
end

"Air the atmosphere does not reach becomes water, at least two cells deep in the level set, so its shrinking does not open it again."
@kernel inbounds=true function fillair_kernel!(types, φ, @Const(reach), depth::Float32)
    i = @index(Global)
    if types[i] == AIR && reach[i] == 0x00
        types[i] = LIQUID
        φ[i] = min(φ[i], -depth)
    end
end

"""
    filltrapped!(s; maxrounds = 32) -> s

Fill the air the water has closed round (see above): reach the air from the top of the
grid, sweeping along x, y and z until a round reaches no more, and make what is left
water.
"""
function filltrapped!(s::FlipBasin; maxrounds = 32)
    be = s.backend
    n1, n2, n3 = cells(s)
    seedair_kernel!(be, WG3)(s.reach, s.types; ndrange = cells(s))
    for _ in 1:maxrounds
        fill!(s.changed, Int32(0))
        sweepair_kernel!(be, (16, 16))(s.reach, s.changed, s.types, Int32(3); ndrange = (n1, n2))
        sweepair_kernel!(be, (16, 16))(s.reach, s.changed, s.types, Int32(1); ndrange = (n2, n3))
        sweepair_kernel!(be, (16, 16))(s.reach, s.changed, s.types, Int32(2); ndrange = (n1, n3))
        Array(s.changed)[1] == 0 && break
    end
    fillair_kernel!(be, 256)(s.types, s.φ, s.reach, 2f0 * s.grid.dx; ndrange = length(s.types))
    return s
end

# ─── forces and walls ────────────────────────────────────────────────────────

"Gravity and the race's stirring (`Stir`, the SPH bake's forcing, at the face) on one component."
@kernel inbounds=true function forces_kernel!(ua, @Const(u), @Const(v), @Const(w), stir, g::FlipGrid, off::Vec3f,
                                              axis::Int32, dt::Float32, t::Float32, ρ0::Float32, gravity::Float32)
    i, j, k = @index(Global, NTuple)
    p = nodepos(g, i, j, k, off)
    if !issolid(g, p)
        vel = velocity(u, v, w, g, p)
        acc = stir(p, vel, ρ0, 0f0, t)
        a = axis == 1 ? acc[1] : (axis == 2 ? acc[2] : acc[3] - gravity)
        ua[i, j, k] += dt * a
    end
end

"Whether face (i, j, k) along `axis` has a wall on either side (or is the grid's own edge)."
@inline function walled(g::FlipGrid, i, j, k, axis)
    a, b, c = before(i, j, k, axis)
    n1, n2, n3 = g.n
    (1 <= a && 1 <= b && 1 <= c && i <= n1 && j <= n2 && k <= n3) || return true
    return issolidcell(g, a, b, c) || issolidcell(g, i, j, k)
end

"No flow through the walls: zero on every walled face."
@kernel inbounds=true function solid_kernel!(u, g::FlipGrid, axis::Int32)
    i, j, k = @index(Global, NTuple)
    walled(g, i, j, k, axis) && (u[i, j, k] = 0f0)
end

function forces!(s::FlipBasin, dt)
    g, be, b = s.grid, s.backend, s.basin
    # The forces read the velocity they change: the stir's guide relaxes it. Read a copy.
    copyto!(s.u0, s.u); copyto!(s.v0, s.v); copyto!(s.w0, s.w)
    for (arr, off, a) in ((s.u, UOFF, 1), (s.v, VOFF, 2), (s.w, WOFF, 3))
        forces_kernel!(be, WG3)(arr, s.u0, s.v0, s.w0, s.stir, g, off, Int32(a), Float32(dt), Float32(s.t), b.ρ0, b.g;
                                ndrange = size(arr))
    end
    return solidify!(s)
end

function solidify!(s::FlipBasin)
    for (arr, a) in ((s.u, 1), (s.v, 2), (s.w, 3))
        solid_kernel!(s.backend, WG3)(arr, s.grid, Int32(a); ndrange = size(arr))
    end
    return s
end

# ─── the pressure ────────────────────────────────────────────────────────────
#
# Unknown: P = p Δt/ρ, so u ← u - ∇P. Equation per liquid cell, scaled by -Δx²:
# Σ_neighbours (P_i - P_n) c_n = -Δx div u + Δx² γ, c_n = 1 for water, 1/θ for air
# (ghost fluid, the air cell's pressure the extrapolation that is zero at the surface),
# no term for a wall. `γ` is a small divergence that gives back volume the surface
# tracking lost.

@kernel inbounds=true function rhs_kernel!(b, @Const(u), @Const(v), @Const(w), @Const(types), dx::Float32, γ::Float32)
    i, j, k = @index(Global, NTuple)
    b[i, j, k] = types[i, j, k] == LIQUID ?
                 -dx * (u[i + 1, j, k] - u[i, j, k] + v[i, j + 1, k] - v[i, j, k] + w[i, j, k + 1] - w[i, j, k]) + dx * dx * γ :
                 0f0
end

@kernel inbounds=true function diag_kernel!(diag, @Const(types), @Const(φ))
    i, j, k = @index(Global, NTuple)
    d = 0f0
    if types[i, j, k] == LIQUID
        for m in 1:6
            a, b, c = neighbour(i, j, k, m)
            t = typeat(types, a, b, c)
            t == LIQUID && (d += 1f0)
            t == AIR && (d += 1f0 / theta(φ[i, j, k], φ[a, b, c]))
        end
    end
    diag[i, j, k] = d
end

# The diagonal: the fine level's, with the ghost fluid's, stored; a coarse level's, its
# open faces, counted (an air neighbour is a Dirichlet one at its centre).
@inline celldiag(diag::AbstractArray, types, i, j, k) = @inbounds diag[i, j, k]
@inline function celldiag(::Nothing, types, i, j, k)
    d = 0f0
    for m in 1:6
        a, b, c = neighbour(i, j, k, m)
        typeat(types, a, b, c) != SOLID && (d += 1f0)
    end
    return d
end

@inline function liquidsum(x, types, i, j, k)
    s = 0f0
    for m in 1:6
        a, b, c = neighbour(i, j, k, m)
        typeat(types, a, b, c) == LIQUID && (s += @inbounds x[a, b, c])
    end
    return s
end

@kernel inbounds=true function residual_kernel!(r, @Const(x), @Const(b), @Const(types), diag)
    i, j, k = @index(Global, NTuple)
    r[i, j, k] = types[i, j, k] == LIQUID ?
                 b[i, j, k] - (celldiag(diag, types, i, j, k) * x[i, j, k] - liquidsum(x, types, i, j, k)) : 0f0
end

"""
One colour of red-black Gauss-Seidel: the cells with (i + j + k) of parity `colour`,
one thread each (launched over half the first axis: thread `h` is cell `2h` or `2h - 1`,
whichever has the colour).
"""
@kernel inbounds=true function rbgs_kernel!(x, @Const(b), @Const(types), diag, colour::Int32)
    h, j, k = @index(Global, NTuple)
    i = 2h - ((colour + j + k) & 1)
    if types[i, j, k] == LIQUID
        d = celldiag(diag, types, i, j, k)
        x[i, j, k] = d > 0f0 ? (b[i, j, k] + liquidsum(x, types, i, j, k)) / d : 0f0
    end
end

"A coarse cell is air if any of its eight is, else water if any is, else wall."
@kernel inbounds=true function coarsen_kernel!(tc, @Const(tf))
    I, J, K = @index(Global, NTuple)
    air = false; liquid = false
    for c in 0:1, b in 0:1, a in 0:1
        t = typeat(tf, 2I - 1 + a, 2J - 1 + b, 2K - 1 + c)
        t == AIR && (air = true)
        t == LIQUID && (liquid = true)
    end
    tc[I, J, K] = air ? AIR : (liquid ? LIQUID : SOLID)
end

@inline weightB(a) = (a == 1 || a == 2) ? 0.375f0 : 0.125f0

"McAdams et al.'s restriction (the 4-point stencil (1, 3, 3, 1)/8 per axis) of the fine residual into the coarse water cells, times 4 for the coarser spacing."
@kernel inbounds=true function restrict_kernel!(bc, @Const(rf), @Const(tf), @Const(tc))
    I, J, K = @index(Global, NTuple)
    s = 0f0
    if tc[I, J, K] == LIQUID
        for c in 0:3, b in 0:3, a in 0:3
            i = 2I - 2 + a; j = 2J - 2 + b; k = 2K - 2 + c
            if typeat(tf, i, j, k) == LIQUID
                s += weightB(a) * weightB(b) * weightB(c) * rf[i, j, k]
            end
        end
    end
    bc[I, J, K] = 4f0 * s
end

@inline function prolongweights(i)
    I = (i + 1) ÷ 2
    return isodd(i) ? (I - 1, 0.25f0, I, 0.75f0) : (I, 0.75f0, I + 1, 0.25f0)
end

"The coarse correction, trilinearly interpolated, added into the fine water cells (8× the restriction's transpose)."
@kernel inbounds=true function prolong_kernel!(xf, @Const(xc), @Const(tf), @Const(tc))
    i, j, k = @index(Global, NTuple)
    if tf[i, j, k] == LIQUID
        a0, wa0, a1, wa1 = prolongweights(i)
        b0, wb0, b1, wb1 = prolongweights(j)
        c0, wc0, c1, wc1 = prolongweights(k)
        s = 0f0
        for m in 0:7
            a = (m & 1) == 0 ? a0 : a1; wa = (m & 1) == 0 ? wa0 : wa1
            b = (m & 2) == 0 ? b0 : b1; wb = (m & 2) == 0 ? wb0 : wb1
            c = (m & 4) == 0 ? c0 : c1; wc = (m & 4) == 0 ? wc0 : wc1
            typeat(tc, a, b, c) == LIQUID && (s += wa * wb * wc * xc[a, b, c])
        end
        xf[i, j, k] += s
    end
end

"""
The coarsest level solved in one workgroup: `sweeps` red-black Gauss-Seidel sweeps
from zero, then as many in the reverse colour order, so the V-cycle stays symmetric.
"""
@kernel function coarse_kernel!(x, @Const(b), @Const(types), n1::Int32, n2::Int32, n3::Int32, sweeps::Int32)
    t = @index(Local)
    xs = @localmem Float32 (1024,)
    N = n1 * n2 * n3
    active = t <= N
    i = (t - 1) % n1 + 1; j = ((t - 1) ÷ n1) % n2 + 1; k = (t - 1) ÷ (n1 * n2) + 1
    ti = active ? @inbounds(types[i, j, k]) : SOLID
    bi = active ? @inbounds(b[i, j, k]) : 0f0
    @inbounds xs[t] = 0f0
    @synchronize
    for s in 1:2sweeps
        for half in 0:1
            colour = s <= sweeps ? half : 1 - half
            if active && ti == LIQUID && (i + j + k) % 2 == colour
                d = 0f0; acc = bi
                for m in 1:6
                    a, bb, c = neighbour(i, j, k, m)
                    tn = typeat(types, a, bb, c)
                    tn != SOLID && (d += 1f0)
                    tn == LIQUID && (acc += @inbounds xs[(c - 1) * n1 * n2 + (bb - 1) * n1 + a])
                end
                @inbounds xs[t] = d > 0f0 ? acc / d : 0f0
            end
            @synchronize
        end
    end
    active && (@inbounds x[i, j, k] = xs[t])
end

"A level's smoothing from zero, its first sweep: the first colour's water cells get `b / diagonal` (their neighbours are zero), every other cell zero."
@kernel inbounds=true function start_kernel!(x, @Const(b), @Const(types), diag)
    i, j, k = @index(Global, NTuple)
    v = 0f0
    if (i + j + k) % 2 == 0 && types[i, j, k] == LIQUID
        d = celldiag(diag, types, i, j, k)
        d > 0f0 && (v = b[i, j, k] / d)
    end
    x[i, j, k] = v
end

# Level `l`'s arrays: the fine level is the CG's own.
leveltypes(s::FlipBasin, l) = l == 1 ? s.types : s.levels[l - 1].types
leveldiag(s::FlipBasin, l) = l == 1 ? s.diag : nothing

function smooth!(s::FlipBasin, l, x, b, colours)
    for c in colours
        n1, n2, n3 = size(x)
        rbgs_kernel!(s.backend, WG3)(x, b, leveltypes(s, l), leveldiag(s, l), Int32(c); ndrange = (n1 ÷ 2, n2, n3))
    end
end

"""
    vcycle!(s, z, r; sweeps = 2)

`z = M⁻¹ r`: one V-cycle from zero, `sweeps` red-black sweeps down and the reverse up,
the coarsest level solved in one workgroup.
"""
function vcycle!(s::FlipBasin, z, r; sweeps = 2, coarse = 40)
    be = s.backend
    L = Int(s.grid.levels)
    # The first colour's first sweep from zero is the start kernel's.
    down = repeat([0, 1], sweeps)[2:end]; up = repeat([1, 0], sweeps)
    xs = (z, (lv.x for lv in s.levels)...)
    bs = (r, (lv.b for lv in s.levels)...)
    rs = (s.rv, (lv.r for lv in s.levels)...)
    for l in 1:L-1
        start_kernel!(be, WG3)(xs[l], bs[l], leveltypes(s, l), leveldiag(s, l); ndrange = size(xs[l]))
        smooth!(s, l, xs[l], bs[l], down)
        residual_kernel!(be, WG3)(rs[l], xs[l], bs[l], leveltypes(s, l), leveldiag(s, l); ndrange = size(xs[l]))
        restrict_kernel!(be, WG3)(bs[l + 1], rs[l], leveltypes(s, l), leveltypes(s, l + 1); ndrange = size(bs[l + 1]))
    end
    n1, n2, n3 = size(xs[L])
    coarse_kernel!(be, 1024)(xs[L], bs[L], leveltypes(s, L), Int32(n1), Int32(n2), Int32(n3), Int32(coarse);
                             ndrange = 1024)
    for l in L-1:-1:1
        prolong_kernel!(be, WG3)(xs[l], xs[l + 1], leveltypes(s, l), leveltypes(s, l + 1); ndrange = size(xs[l]))
        smooth!(s, l, xs[l], bs[l], up)
    end
    return z
end

# Conjugate gradients with the vector updates folded into the passes that read the
# vectors anyway: the search direction `p = z + β d` is made where it is used, each
# pass sums its dot product per group into `partials`.

"Each workgroup's sum of `v` (one value a thread) into `partials`."
@inline function groupsum!(partials, sm, v, li, gi)
    @inbounds sm[li] = v
    @synchronize
    step = 128
    while step >= 1
        li <= step && (@inbounds sm[li] += sm[li + step])
        @synchronize
        step ÷= 2
    end
    li == 1 && (@inbounds partials[gi] = sm[1])
end

"""
`q = A p` for the search direction `p = z + β d`, and each workgroup's share of
`p · q`: launched over the cells in workgroups of `WG3` (256 threads), one partial sum
each (see `cellgroups`).
"""
@kernel function search_kernel!(partials, q, @Const(z), @Const(d), @Const(types), @Const(diag), β::Float32)
    i, j, k = @index(Global, NTuple); li = @index(Local); gi = @index(Group)
    sm = @localmem Float32 (256,)
    qv = 0f0; v = 0f0
    @inbounds if types[i, j, k] == LIQUID
        p = z[i, j, k] + β * d[i, j, k]
        acc = 0f0
        for m in 1:6
            a, b, c = neighbour(i, j, k, m)
            typeat(types, a, b, c) == LIQUID && (acc += z[a, b, c] + β * d[a, b, c])
        end
        qv = diag[i, j, k] * p - acc
        v = p * qv
    end
    @inbounds q[i, j, k] = qv
    groupsum!(partials, sm, v, li, gi)
end

"How many `WG3` workgroups cover the cells: the length of the partial sums a cell kernel writes."
cellgroups(dims) = prod(cld.(dims, WG3))

"The step along the search direction: `d = z + β d`, `x += α d`, `r -= α q`, and each group's share of `r · r`."
@kernel function advance_kernel!(partials, x, r, d, @Const(z), @Const(q), α::Float32, β::Float32)
    l = @index(Global); li = @index(Local); gi = @index(Group)
    sm = @localmem Float32 (256,)
    v = 0f0
    if l <= length(x)
        @inbounds p = z[l] + β * d[l]
        @inbounds d[l] = p
        @inbounds x[l] += α * p
        @inbounds rl = r[l] - α * q[l]
        @inbounds r[l] = rl
        v = rl * rl
    end
    groupsum!(partials, sm, v, li, gi)
end

"""
    pressure!(s; tol = 1e-4, maxiter = 60, sweeps = 2) -> iterations

Solve for the pressure by conjugate gradients with the multigrid V-cycle as the
preconditioner (McAdams et al.'s algorithm 3), from the last step's pressure, until
the residual is `tol` of the right-hand side's.
"""
function pressure!(s::FlipBasin; tol = 1f-4, maxiter = 60, sweeps = 2)
    be = s.backend
    for l in 2:s.grid.levels
        coarsen_kernel!(be, WG3)(leveltypes(s, l), leveltypes(s, l - 1); ndrange = size(leveltypes(s, l)))
    end
    diag_kernel!(be, WG3)(s.diag, s.types, s.φ; ndrange = cells(s))
    # Start from last step's pressure, wherever it is still water.
    residual_kernel!(be, WG3)(s.r, s.P, s.b, s.types, s.diag; ndrange = cells(s))
    bnorm = sqrt(inner(s, s.b, s.b))
    bnorm == 0f0 && return 0
    target = tol * bnorm
    rnorm = sqrt(inner(s, s.r, s.r))
    rnorm <= target && return 0
    vcycle!(s, s.z, s.r; sweeps)
    ρ = inner(s, s.z, s.r)
    β = 0f0
    for it in 1:maxiter
        search_kernel!(be, WG3)(s.cellpartials, s.q, s.z, s.d, s.types, s.diag, Float32(β); ndrange = cells(s))
        α = ρ / sum(s.cellpartials)
        reduction(advance_kernel!, s, s.P, s.r, s.d, s.z, s.q, Float32(α), Float32(β))
        rnorm = sqrt(sum(s.partials))
        rnorm <= target && return it
        vcycle!(s, s.z, s.r; sweeps)
        ρnew = inner(s, s.z, s.r)
        β = ρnew / ρ
        ρ = ρnew
    end
    @warn "pressure solve did not converge" residual = rnorm / bnorm maxiter
    return maxiter
end

"u ← u - ∇P on every face with water on a side and no wall; marks those faces valid for the extrapolation."
@kernel inbounds=true function project_kernel!(u, valid, @Const(P), @Const(types), @Const(φ), dx::Float32, axis::Int32)
    i, j, k = @index(Global, NTuple)
    a, b, c = before(i, j, k, axis)
    t0 = typeat(types, a, b, c); t1 = typeat(types, i, j, k)
    ok = t0 != SOLID && t1 != SOLID && (t0 == LIQUID || t1 == LIQUID)
    if ok
        Δ = if t0 == LIQUID && t1 == LIQUID
            P[i, j, k] - P[a, b, c]
        elseif t0 == LIQUID
            -P[a, b, c] / theta(φ[a, b, c], φ[i, j, k])
        else
            P[i, j, k] / theta(φ[i, j, k], φ[a, b, c])
        end
        u[i, j, k] -= Δ / dx
    end
    valid[i, j, k] = ok ? 0x01 : 0x00
end

@kernel inbounds=true function clearair_kernel!(P, @Const(types))
    i = @index(Global)
    types[i] == LIQUID || (P[i] = 0f0)
end

"""
    project!(s, dt; recover = 0.5)

Make the velocity divergence-free, but for a uniform divergence that gives back over
`recover` seconds the volume the surface tracking gained or lost since the start (Kim
et al. 2007's volume control, its proportional part), at most a percent a step.
"""
function project!(s::FlipBasin, dt; recover = 0.5, sweeps = 2)
    be, g = s.backend, s.grid
    loss = (s.volume0 - volume(s)) / s.volume0
    γ = Float32(clamp(loss / recover, -0.01 / dt, 0.01 / dt))
    rhs_kernel!(be, WG3)(s.b, s.u, s.v, s.w, s.types, g.dx, γ; ndrange = cells(s))
    clearair_kernel!(be, 256)(s.P, s.types; ndrange = length(s.P))
    its = pressure!(s; sweeps)
    push!(s.iterations, its)
    for (arr, f, a) in ((s.u, s.fu, 1), (s.v, s.fv, 2), (s.w, s.fw, 3))
        project_kernel!(be, WG3)(arr, f, s.P, s.types, s.φ, g.dx, Int32(a); ndrange = size(arr))
    end
    return s
end

# ─── extrapolation ───────────────────────────────────────────────────────────

"One layer of velocity extrapolation: an invalid face takes the mean of its valid neighbours of the same component."
@kernel inbounds=true function extrapolate_kernel!(out, fout, @Const(src), @Const(fin), g::FlipGrid, axis::Int32)
    i, j, k = @index(Global, NTuple)
    if fin[i, j, k] != 0x00
        out[i, j, k] = src[i, j, k]; fout[i, j, k] = 0x01
    else
        s = 0f0; m = 0
        if !walled(g, i, j, k, axis)
            for q in 1:6
                x, y, z = neighbour(i, j, k, q)
                if inrange(src, x, y, z) && fin[x, y, z] != 0x00
                    s += src[x, y, z]; m += 1
                end
            end
        end
        out[i, j, k] = m > 0 ? s / m : src[i, j, k]
        fout[i, j, k] = m > 0 ? 0x01 : 0x00
    end
end

"Zero on every face the extrapolation did not reach: air far from the water, where gravity would otherwise pile up step after step."
@kernel inbounds=true function unreached_kernel!(u, @Const(reached))
    i = @index(Global)
    reached[i] == 0x00 && (u[i] = 0f0)
end

"""
    extrapolate!(s; layers = 4)

Carry the water's velocity `layers` faces out into the air and the band's particles'
reach (see `extrapolate_kernel!`); zero beyond.
"""
function extrapolate!(s::FlipBasin; layers = 4)
    be = s.backend
    for (arr, tmp, f, f2, a) in ((s.u, s.u0, s.fu, s.fu2, 1), (s.v, s.v0, s.fv, s.fv2, 2), (s.w, s.w0, s.fw, s.fw2, 3))
        src, dst, fin, fout = arr, tmp, f, f2
        for _ in 1:layers
            extrapolate_kernel!(be, WG3)(dst, fout, src, fin, s.grid, Int32(a); ndrange = size(arr))
            src, dst = dst, src; fin, fout = fout, fin
        end
        src === arr || copyto!(arr, src)
        # The flags of what was reached stay in `f`: the particles' advection reads them.
        fin === f || copyto!(f, fin)
        unreached_kernel!(be, 256)(arr, f; ndrange = length(arr))
    end
    return solidify!(s)
end

# ─── grid to particles ───────────────────────────────────────────────────────

"APIC's gather of one component at `xp`: the velocity and its affine part (4/Δx² Σ w u (x_i - x_p))."
@inline function gather(u, xp, lo::Point3f, dx::Float32, off::Vec3f)
    q = Vec3f((xp - lo) / dx) - off
    bx, x0, x1, x2 = bspline(q[1]); by, y0, y1, y2 = bspline(q[2]); bz, z0, z1, z2 = bspline(q[3])
    n1, n2, n3 = size(u)
    vel = 0f0; aff = Vec3f(0)
    for mz in 0:2, my in 0:2, mx in 0:2
        i = clamp(bx + mx + 1, 1, n1); j = clamp(by + my + 1, 1, n2); k = clamp(bz + mz + 1, 1, n3)
        wgt = pick(mx, x0, x1, x2) * pick(my, y0, y1, y2) * pick(mz, z0, z1, z2)
        val = @inbounds u[i, j, k]
        vel += wgt * val
        aff += (wgt * val) * Vec3f(bx + mx - q[1], by + my - q[2], bz + mz - q[3]) * dx
    end
    return vel, aff * (4f0 / (dx * dx))
end

"""
Each particle's velocity and affine velocity from the grid (APIC's gather), where every
face the gather reads was reached by the projection or the extrapolation; elsewhere (air
past the extrapolation) its own velocity, a step `dt` of gravity on, without an affine part.
"""
@kernel inbounds=true function g2p_kernel!(vx, vy, vz, cx, cy, cz, @Const(x), flow, dt::Float32, gravity::Float32, n::Int32)
    p = @index(Global)
    if p <= n
        xp = x[p]
        a1, c1, k1 = gathered(flow, 1, xp); a2, c2, k2 = gathered(flow, 2, xp); a3, c3, k3 = gathered(flow, 3, xp)
        if k1 && k2 && k3
            vx[p] = a1; vy[p] = a2; vz[p] = a3
            cx[p] = c1; cy[p] = c2; cz[p] = c3
        else
            vz[p] -= gravity * dt
            cx[p] = Vec3f(0); cy[p] = Vec3f(0); cz[p] = Vec3f(0)
        end
    end
end

function grid2particles!(s::FlipBasin, dt)
    ps, g = s.particles, s.grid
    ps.n == 0 && return s
    g2p_kernel!(s.backend, 256)(ps.vx, ps.vy, ps.vz, ps.cx, ps.cy, ps.cz, ps.x, Flow(s), Float32(dt), s.basin.g,
                                Int32(ps.n); ndrange = ps.n)
    return s
end

# ─── resampling ──────────────────────────────────────────────────────────────

@inline lincell(n1, n2, i, j, k) = (k - 1) * n1 * n2 + (j - 1) * n1 + i

"""
Each particle's cell, if it stays (inside the grid and the walls, within the band, not
far above the surface), and its rank among the cell's particles.
"""
@kernel inbounds=true function bin_kernel!(key, rank, counts, @Const(x), @Const(φ), g::FlipGrid, deep::Float32,
                                           above::Float32, n::Int32)
    p = @index(Global)
    if p <= n
        xp = x[p]
        c = Vec3f((xp - g.lo) / g.dx)
        keep = isfinite(c[1]) && isfinite(c[2]) && isfinite(c[3])
        i = 0; j = 0; k = 0
        if keep
            i = unsafe_trunc(Int, floor(c[1])) + 1; j = unsafe_trunc(Int, floor(c[2])) + 1
            k = unsafe_trunc(Int, floor(c[3])) + 1
            keep = 1 <= i <= g.n[1] && 1 <= j <= g.n[2] && 1 <= k <= g.n[3] && !issolid(g, xp)
        end
        if keep
            f = trilinear(φ, indexspace(g, xp, COFF))
            keep = -deep <= f < above
        end
        if keep
            l = lincell(g.n[1], g.n[2], i, j, k)
            key[p] = Int32(l)
            rank[p] = (KA.@atomic :monotonic counts[l] += Int32(1)) - Int32(1)
        else
            key[p] = Int32(0); rank[p] = Int32(-1)
        end
    end
end

# How many particles a cell gets: the ones it keeps (at most twice the minimum) and the
# ones added where it is in the resampling band (between one cell and `deep` below the
# surface; from the surface down when seeding) and has fewer than the minimum.
@inline function cellcounts(count, φc, solid, nmin, deep, dx, seeding)
    kept = min(count, Int32(2) * nmin)
    inband = !solid && -deep <= φc && φc <= (seeding ? 0.5f0 * dx : -dx)
    add = inband && kept < nmin ? nmin - kept : Int32(0)
    return kept, add
end

@kernel inbounds=true function totals_kernel!(totals, @Const(counts), @Const(φ), g::FlipGrid, nmin::Int32, deep::Float32,
                                              seeding::Bool)
    i, j, k = @index(Global, NTuple)
    l = lincell(g.n[1], g.n[2], i, j, k)
    kept, add = cellcounts(counts[l], φ[i, j, k], issolidcell(g, i, j, k), nmin, deep, g.dx, seeding)
    totals[l] = kept + add
end

@kernel inbounds=true function scatter_kernel!(x2, vx2, vy2, vz2, cx2, cy2, cz2, lag2, @Const(x), @Const(vx), @Const(vy),
                                               @Const(vz), @Const(cx), @Const(cy), @Const(cz), @Const(lag), @Const(key),
                                               @Const(rank), @Const(totals), @Const(offsets), nmax::Int32, n::Int32)
    p = @index(Global)
    if p <= n
        l = key[p]
        if l > Int32(0) && rank[p] < nmax
            dst = offsets[l] - totals[l] + rank[p] + 1
            x2[dst] = x[p]; vx2[dst] = vx[p]; vy2[dst] = vy[p]; vz2[dst] = vz[p]
            cx2[dst] = cx[p]; cy2[dst] = cy[p]; cz2[dst] = cz[p]; lag2[dst] = lag[p]
        end
    end
end

"New particles in the cells that need them: stratified over the cell's eight octants, jittered, the grid's velocity and its affine part."
@kernel inbounds=true function spawn_kernel!(x2, vx2, vy2, vz2, cx2, cy2, cz2, lag2, @Const(counts), @Const(totals),
                                             @Const(offsets), @Const(φ), @Const(u), @Const(v), @Const(w), g::FlipGrid,
                                             nmin::Int32, deep::Float32, seeding::Bool, seed::UInt32)
    i, j, k = @index(Global, NTuple)
    l = lincell(g.n[1], g.n[2], i, j, k)
    kept, add = cellcounts(counts[l], φ[i, j, k], issolidcell(g, i, j, k), nmin, deep, g.dx, seeding)
    if add > Int32(0)
        base = offsets[l] - totals[l] + kept
        corner = g.lo + Vec3f(i - 1, j - 1, k - 1) * g.dx
        for m in Int32(0):add-Int32(1)
            h = (l % Int32) * Int32(31) + m
            jx = hash01(h, (seed % Int32) + Int32(1)); jy = hash01(h, (seed % Int32) + Int32(2))
            jz = hash01(h, (seed % Int32) + Int32(3))
            o = m % Int32(8)
            sub = Vec3f(o & Int32(1), (o >> 1) & Int32(1), (o >> 2) & Int32(1))
            xp = Point3f(corner + (sub + Vec3f(jx, jy, jz)) * (0.5f0 * g.dx))
            dst = base + m + 1
            lag2[dst] = 0f0
            # Seeding from the surface down: a particle the level set puts in the air is
            # put where the next binning drops it.
            if seeding && trilinear(φ, indexspace(g, xp, COFF)) >= 0f0
                xp = Point3f(NaN32, NaN32, NaN32)
                x2[dst] = xp; vx2[dst] = 0f0; vy2[dst] = 0f0; vz2[dst] = 0f0
                cx2[dst] = Vec3f(0); cy2[dst] = Vec3f(0); cz2[dst] = Vec3f(0)
            else
                a1, c1 = gather(u, xp, g.lo, g.dx, UOFF); a2, c2 = gather(v, xp, g.lo, g.dx, VOFF)
                a3, c3 = gather(w, xp, g.lo, g.dx, WOFF)
                x2[dst] = xp; vx2[dst] = a1; vy2[dst] = a2; vz2[dst] = a3
                cx2[dst] = c1; cy2[dst] = c2; cz2[dst] = c3
            end
        end
    end
end

"""
    resample!(s; seeding = false)

Keep the narrow band `s.band` cells deep with `nmin`–`2nmin` particles a cell
(Ferstl et al. §3.3): drop those that left it or crowd a cell, add some where a cell
in the band has too few, and write them out in the order of their cells, so the next
step's transfers read and write memory in order.
"""
function resample!(s::FlipBasin; seeding = false)
    ps, g, be = s.particles, s.grid, s.backend
    n = ps.n
    deep = Float32(s.band * g.dx)
    fill!(s.counts, Int32(0))
    n > 0 && bin_kernel!(be, 256)(s.key, s.rank, s.counts, ps.x, s.φ, g, deep, 3f0 * g.dx, Int32(n); ndrange = n)
    totals_kernel!(be, WG3)(s.totals, s.counts, s.φ, g, s.nmin, deep, seeding; ndrange = cells(s))
    AK.accumulate!(+, s.offsets, s.totals; init = Int32(0))
    total = Int(Array(view(s.offsets, length(s.offsets):length(s.offsets)))[1])
    total <= capacity(ps) || error("the narrow band needs $total particles, more than the $(capacity(ps)) allocated")
    n > 0 && scatter_kernel!(be, 256)(ps.x2, ps.vx2, ps.vy2, ps.vz2, ps.cx2, ps.cy2, ps.cz2, ps.lag2, ps.x, ps.vx, ps.vy,
                                      ps.vz, ps.cx, ps.cy, ps.cz, ps.lag, s.key, s.rank, s.totals, s.offsets,
                                      Int32(2s.nmin), Int32(n); ndrange = n)
    spawn_kernel!(be, WG3)(ps.x2, ps.vx2, ps.vy2, ps.vz2, ps.cx2, ps.cy2, ps.cz2, ps.lag2, s.counts, s.totals, s.offsets, s.φ,
                           s.u, s.v, s.w, g, s.nmin, deep, seeding, UInt32(s.steps + 1); ndrange = cells(s))
    swap!(ps)
    ps.n = total
    return s
end

# ─── a step ──────────────────────────────────────────────────────────────────

"The fastest the water moves, from the grid's faces."
maxspeed(s::FlipBasin) = max(maxabs(s, s.u), maxabs(s, s.v), maxabs(s, s.w))

"""
    step!(s, dt)

One narrow-band FLIP step (Ferstl et al.'s figure 2): advect the particles and the
grid, map the particles to it, combine, redistance, classify, add gravity and the
stirring, project, extrapolate, map back, resample.
"""
function step!(s::FlipBasin, dt)
    advect!(s, dt)
    particles2grid!(s, dt)
    redistance!(s)
    classify!(s)
    filltrapped!(s)
    forces!(s, dt)
    project!(s, dt)
    extrapolate!(s; layers = 6)
    grid2particles!(s, dt)
    s.steps += 1
    resample!(s)
    s.t += dt
    return s
end


# ─── whitewater from the narrow band ─────────────────────────────────────────
#
# Ihmsen et al.'s emission (`emit_kernel!`) reads the fluid particles through a cell
# list. Resampling already wrote the particles out cell by cell, so the list is its
# scan: no sort. The normals are the level set's gradient, not the particles' colour
# field, which at the band's bottom would point down into water that is there but has
# no particles. And a diffuse particle is spray, foam or a bubble by where it is in the
# level set, carried by the grid's velocity, not by how many particles are round it.

"Where each cell's particles start in the order resampling left them, and one past the last."
@kernel inbounds=true function cellstart_kernel!(start, @Const(offsets), @Const(totals))
    l = @index(Global)
    n = length(offsets)
    start[l] = l <= n ? offsets[l] - totals[l] + Int32(1) : offsets[n] + Int32(1)
end

@kernel inbounds=true function pack_kernel!(v, @Const(vx), @Const(vy), @Const(vz), n::Int32)
    p = @index(Global)
    p <= n && (v[p] = Vec3f(vx[p], vy[p], vz[p]))
end

"Each particle's surface normal, the level set's gradient, within `near` of the surface; none deeper."
@kernel inbounds=true function levelnormals_kernel!(normal, @Const(x), @Const(φ), g::FlipGrid, near::Float32, n::Int32)
    p = @index(Global)
    if p <= n
        q = indexspace(g, x[p], COFF)
        nrm = Vec3f(0)
        if trilinear(φ, q) > -near
            gx = trilinear(φ, q + Vec3f(0.5f0, 0, 0)) - trilinear(φ, q - Vec3f(0.5f0, 0, 0))
            gy = trilinear(φ, q + Vec3f(0, 0.5f0, 0)) - trilinear(φ, q - Vec3f(0, 0.5f0, 0))
            gz = trilinear(φ, q + Vec3f(0, 0, 0.5f0)) - trilinear(φ, q - Vec3f(0, 0, 0.5f0))
            l = sqrt(gx * gx + gy * gy + gz * gz)
            l > 0f0 && (nrm = Vec3f(gx, gy, gz) / l)
        end
        normal[p] = nrm
    end
end

"""
Every diffuse particle a step `dt` on, by where it is in the level set: spray (more
than `above` over the surface) falls freely, foam (on it) rides the water and ages, a
bubble (more than `below` under it) rises, is dragged towards the water's velocity and
ages too. Gone when its life is spent, below the floor, outside the wall or past the
rules' `within`.
"""
@kernel inbounds=true function leveldiffuse_kernel!(wx, wv, wlife, wkind, @Const(φ), @Const(u), @Const(v), @Const(w),
                                                    g::FlipGrid, dt::Float32, r::WhitewaterRules, gravity::Float32,
                                                    above::Float32, below::Float32)
    i = @index(Global)
    life = wlife[i]
    if life > 0f0
        x = wx[i]; vel = wv[i]
        f = trilinear(φ, indexspace(g, x, COFF))
        vf = velocity(u, v, w, g, x)
        if f > above
            vel = vel + Vec3f(0, 0, -gravity * dt)
            wkind[i] = SPRAY
        elseif f < -below
            vel = vel + dt * Vec3f(0, 0, r.buoyancy * gravity) + r.drag * (vf - vel)
            life -= dt
            wkind[i] = BUBBLE
        else
            vel = vf
            life -= dt
            wkind[i] = FOAM
        end
        x = x + dt * vel
        reach = min(r.within, g.radius)
        (x[3] < g.floor || x[1] * x[1] + x[2] * x[2] > reach * reach) && (life = 0f0)
        wx[i] = x; wv[i] = vel; wlife[i] = life
    end
end

"""
    whitewater!(w::Whitewater, s::FlipBasin, u, v, dt; substeps = 4, near = 1.5, above = 0.25, below = 0.75)

The narrow band's whitewater over the frame `dt` (see above): emitted from particles
within `near` cells of the surface, spray more than `above` cells over it, bubbles
more than `below` under it. `u` and `v`, the frame on the host, are not read.
"""
function whitewater!(w::Whitewater, s::FlipBasin, u, v, dt; substeps = 4, near = 1.5f0, above = 0.25f0, below = 0.75f0)
    be, ps, g = s.backend, s.particles, s.grid
    n = ps.n
    # Between resamplings the second set of particle arrays is free: the packed
    # velocities and the normals go there.
    vel, normal = ps.cx2, ps.cy2
    cellstart_kernel!(be, 256)(s.cellstart, s.offsets, s.totals; ndrange = length(s.cellstart))
    pack_kernel!(be, 256)(vel, ps.vx, ps.vy, ps.vz, Int32(n); ndrange = n)
    levelnormals_kernel!(be, 256)(normal, ps.x, s.φ, g, Float32(near * g.dx), Int32(n); ndrange = n)
    fluid = FluidGrid(ps.x, vel, s.cellstart, g.lo, Vec{3,Int32}(g.n...), g.dx)
    w.frames += 1
    emit_kernel!(be, 256)(w.x, w.v, w.life, w.head, fluid, normal, Float32(dt), w.rules, UInt32(w.frames), w.floor,
                          w.stream; ndrange = n)
    for _ in 1:substeps
        leveldiffuse_kernel!(be, 256)(w.x, w.v, w.life, w.kind, s.φ, s.u, s.v, s.w, g, Float32(dt / substeps), w.rules,
                                      w.g, Float32(above * g.dx), Float32(below * g.dx); ndrange = length(w.life))
    end
    return w
end

# ─── whitewater at the rules' scale ──────────────────────────────────────────
#
# Ihmsen et al.'s potentials are not resolution-free: trapped air sums the neighbours'
# velocity differences within a reach, a crest the turn of their normals, and both
# grow with the reach. The rules the film's whirlpool was approved with were set on
# SPH particles 10 cm apart, reach 20 cm. On the band's particles, 2.5 cm apart with a
# reach of 5 cm, the same rules saw a quarter of the velocity difference and a
# sixteenth of the curvature: crests never fired, trapped air on two thirds of the
# surface, and the foam layer the level set kept was a quarter as thick. A tenth of
# sgs1's foam came of it. So the whitewater runs on the bake's frames at the rules'
# scale instead: the band's particles gathered into cells as far apart as the SPH
# bake's particles, and, where SPH counted its neighbours, the water the level set
# says is there, on a lattice that far apart.

"""
    BandSampler(lo, hi, spacing, least)

Where a frame's band particles are gathered into cells `spacing` wide over the box
`lo`–`hi`: their count and the sums of their positions and velocities, kept on the host
between frames. A cell with at least `least` particles stands for an SPH particle.
"""
struct BandSampler
    lo::Point3f
    spacing::Float32
    dims::NTuple{3,Int}
    count::Vector{Int32}
    sums::Matrix{Float32}
    least::Int32
end
function BandSampler(lo, hi, spacing, least)
    dims = Tuple(ceil.(Int, (Vec3f(hi) - Vec3f(lo)) / Float32(spacing)))
    return BandSampler(Point3f(lo), Float32(spacing), dims, zeros(Int32, prod(dims)), zeros(Float32, 6, prod(dims)),
                       Int32(least))
end

"""
    representatives!(s::BandSampler, u, v) -> (x, w)

The band's particles at `u` with velocities `v` (`3 × N`) as SPH particles `s.spacing`
apart (`3 × M` each): one at the mean position and velocity of every cell holding at
least `s.least` of them.
"""
function representatives!(s::BandSampler, u::AbstractMatrix{Float32}, v::AbstractMatrix{Float32})
    fill!(s.count, Int32(0)); fill!(s.sums, 0f0)
    n1, n2, n3 = s.dims
    for i in axes(u, 2)
        a = floor(Int, (u[1, i] - s.lo[1]) / s.spacing); b = floor(Int, (u[2, i] - s.lo[2]) / s.spacing)
        c = floor(Int, (u[3, i] - s.lo[3]) / s.spacing)
        (0 <= a < n1 && 0 <= b < n2 && 0 <= c < n3) || continue
        cell = a + n1 * (b + n2 * c) + 1
        s.count[cell] += Int32(1)
        for d in 1:3
            s.sums[d, cell] += u[d, i]; s.sums[d + 3, cell] += v[d, i]
        end
    end
    cells = findall(>=(s.least), s.count)
    x = Matrix{Float32}(undef, 3, length(cells)); w = similar(x)
    for (m, cell) in enumerate(cells), d in 1:3
        x[d, m] = s.sums[d, cell] / s.count[cell]; w[d, m] = s.sums[d + 3, cell] / s.count[cell]
    end
    return x, w
end

"""
    LatticeWater(φ, lo, cell, range)

A frame of a FLIP bake's level set on the device (see [`LevelSets`](@ref): UInt8 over
cells `cell` wide from `lo`, clamped to `±range`), asked where the water is.
"""
struct LatticeWater{A}
    φ::A
    lo::Point3f
    cell::Float32
    range::Float32
end
Adapt.@adapt_structure LatticeWater

"The level set at `x`."
@inline levelφ(L::LatticeWater, x) = (trilinear(L.φ, Vec3f((x - L.lo) / L.cell) - Vec3f(0.5f0)) / 127.5f0 - 1f0) * L.range

@inline function foam_direction(L::LatticeWater,p)
    for scale in (1f0,2f0,4f0,8f0)
        h=scale*L.cell
        gradient=Vec3f(levelφ(L,p+Vec3f(h,0,0))-levelφ(L,p-Vec3f(h,0,0)),
                       levelφ(L,p+Vec3f(0,h,0))-levelφ(L,p-Vec3f(0,h,0)),
                       levelφ(L,p+Vec3f(0,0,h))-levelφ(L,p-Vec3f(0,0,h)))/(2f0*h)
        sum(abs2,gradient)>1f-12 && return gradient
    end
    return Vec3f(0)
end

"Seed foam projection with the simulation's signed-distance interface, before refining the reconstructed indicator."
@inline function foam_seed(L::LatticeWater,p)
    h=L.cell
    for _ in 1:4
        value=levelφ(L,p)
        abs(value)<.05f0*h && break
        gradient=foam_direction(L,p)
        length2=sum(abs2,gradient)
        length2>1f-12 || break
        step=(value/length2)*gradient
        distance=sqrt(sum(abs2,step))
        distance>2f0*h && (step*=2f0*h/distance)
        p-=step
    end
    return p
end

"""
How many SPH particles `spacing` apart, settled on a lattice round `x`, would be within
`h` of it, and their colour field (the sum `normals_kernel!` takes): the lattice's
points at least half a spacing under the level set's surface, where an SPH bake's
particle centres are (its water reaches half a spacing past them). Counted up to the
surface itself, foam had more water round it than in the SPH bake and turned bubble
early. Under the band, which has no particles, the water goes on as it does in an SPH
bake.
"""
@inline function lattice(L::LatticeWater, x, h, spacing)
    m = unsafe_trunc(Int32, floor(h / spacing))
    k = Int32(0); s = Vec3f(0)
    for c in -m:m, b in -m:m, a in -m:m
        o = Vec3f(a, b, c) * spacing
        d = sqrt(sum(abs2, o))
        if d <= 1.001f0 * h && levelφ(L, x + o) < -0.5f0 * spacing
            k += Int32(1); s -= (1f0 - d / h) * o
        end
    end
    return k, s
end

"Each representative particle's surface normal from the lattice's colour field, as `normals_kernel!` makes it from SPH neighbours."
@kernel inbounds=true function latticenormals_kernel!(normal, @Const(x), L, h::Float32, spacing::Float32, n::Int32)
    i = @index(Global)
    if i <= n
        _, s = lattice(L, x[i], h, spacing)
        l = sqrt(sum(abs2, s))
        normal[i] = l > 0.3f0 * h ? s / l : Vec3f(0)
    end
end

"""
Every diffuse particle a step `dt` on, as `diffuse_kernel!` moves it: spray, foam or a
bubble by its lattice neighbour count (see `lattice`), carried by the representatives'
velocity within reach. A bubble out of reach of them has left the band: the water that
carries it there is not kept, and it is gone, as the SPH bake's downwelling carries its
bubbles down the throat and out of sight. Kept, with its own velocity, buoyancy sped it
up without end and the bubbles shot up the funnel's wall and back; kept turning with
the vortex's swirl, they piled up behind the wall. Either way a grey veil hung down the
throat.
"""
@kernel inbounds=true function latticediffuse_kernel!(wx, wv, wlife, wkind, g, L, stream, dt::Float32, r::WhitewaterRules,
                                                      gravity::Float32, floor::Float32, spacing::Float32)
    i = @index(Global)
    life = wlife[i]
    if life > 0f0
        x = wx[i]; v = wv[i]
        c = fluidcell(g, x)
        sv = Vec3f(0); sw = 0f0
        for dk in Int32(-1):Int32(1), dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
            for q in cellrange(g, c + Vec{3,Int32}(di, dj, dk))
                d = sqrt(sum(abs2, x - g.x[q]))
                if d < g.h
                    w = 1f0 - d / g.h
                    sv += w * g.v[q]; sw += w
                end
            end
        end
        vf = sw > 0f0 ? sv / sw : carried(stream, x)
        k, _ = lattice(L, x, g.h, spacing)
        if k < r.spray
            v = v + Vec3f(0, 0, -gravity * dt)
            wkind[i] = SPRAY
        elseif k > r.bubble
            v = v + dt * Vec3f(0, 0, r.buoyancy * gravity) + r.drag * (vf - v)
            life = sw > 0f0 ? life - dt : 0f0
            wkind[i] = BUBBLE
        else
            v = vf
            life -= dt
            wkind[i] = FOAM
        end
        x = x + dt * v
        (x[3] < floor || x[1] * x[1] + x[2] * x[2] > r.within * r.within) && (life = 0f0)
        wx[i] = x; wv[i] = v; wlife[i] = life
    end
end

"""
    BandFrame(sampler, water)

A frame of a FLIP bake as the whitewater reads it at the rules' scale: the band's
particles gathered by `sampler` (see [`BandSampler`](@ref)), the level set as
[`LatticeWater`](@ref).
"""
struct BandFrame{L}
    sampler::BandSampler
    water::L
end

"""
    whitewater!(w::Whitewater, f::BandFrame, u, v, dt; substeps = 4)

The whitewater of a FLIP bake's frame (band particles `u`, velocities `v`, host
`3 × N`) at the scale of `w`'s reach: emitted by `emit_kernel!` from the representative
particles (see [`representatives!`](@ref)), whose surface normals and neighbour counts
come from the level set's lattice.
"""
function whitewater!(w::Whitewater, f::BandFrame, u, v, dt; substeps = 4)
    backend = KA.get_backend(w.life)
    spacing = f.sampler.spacing
    xr, vr = representatives!(f.sampler, u, v)
    g = FluidGrid(xr, vr, w.h, w.lo, w.hi, backend)
    n = length(g.x)
    normal = KA.allocate(backend, Vec3f, n)
    latticenormals_kernel!(backend, 256)(normal, g.x, f.water, w.h, spacing, Int32(n); ndrange = n)
    w.frames += 1
    emit_kernel!(backend)(w.x, w.v, w.life, w.head, g, normal, Float32(dt), w.rules, UInt32(w.frames), w.floor, w.stream;
                          ndrange = n)
    for _ in 1:substeps
        latticediffuse_kernel!(backend, 256)(w.x, w.v, w.life, w.kind, g, f.water, w.stream, Float32(dt / substeps), w.rules,
                                             w.g, w.floor, spacing; ndrange = length(w.life))
    end
    return w
end

"""
    rewhiteflip(path, backend; rules = WhitewaterRules(), spacing = 0.1, out = path, frames = nothing,
                capacity = 12_000_000) -> out

A FLIP bake's whitewater made again from its frames (band particles, their velocities,
the level set) at the scale `rules` were set for: SPH particles `spacing` apart, reach
twice that (see [`BandFrame`](@ref)). Written as `out.whitewater` (the bake's own by
default), over all its frames or `frames`. The diffuse particles do not act on the
water, so nothing is simulated again.
"""
function rewhiteflip(path::AbstractString, backend; rules = WhitewaterRules(), spacing = 0.1f0, out = path, frames = nothing,
                     capacity = 12_000_000)
    pf = ParticleFrames(path)
    ls = LevelSetFrames(path)
    meta = deserialize(path * ".basin.jls")
    b = Basin(; (f => (f == :spacing ? Float32(spacing) : getfield(meta.basin, f)) for f in fieldnames(Basin))...)
    sink = WhitewaterFrames(out, b, backend; fps = meta.fps, rules, capacity)
    # A full cell holds (spacing / band spacing)³ particles; half full stands for a particle.
    sampler = BandSampler(sink.w.lo, sink.w.hi, spacing, ceil(Int, (spacing / meta.basin.spacing)^3 / 2))
    φ = KA.allocate(backend, UInt8, size(ls.data)[1:3]...)
    for k in something(frames, 1:length(pf))
        copyto!(φ, ls.data[:, :, :, k])
        record!(sink, BandFrame(sampler, LatticeWater(φ, ls.lo, ls.cell, ls.range)), Float32.(frame(pf, k)),
                Float32.(velocity(pf, k)))
    end
    close(sink)
    return out
end

"""
    SurfacePotentials(air, crest, energy)

Ihmsen et al.'s raw potentials (see `potentials_kernel!`) of the particles that may
emit (at the surface, over the floor, within the rules' `within`), of one frame or
several pooled.
"""
struct SurfacePotentials
    air::Vector{Float32}
    crest::Vector{Float32}
    energy::Vector{Float32}
end
Base.vcat(ps::SurfacePotentials...) = SurfacePotentials(reduce(vcat, (p.air for p in ps)), reduce(vcat, (p.crest for p in ps)),
                                                       reduce(vcat, (p.energy for p in ps)))

"The potentials of the fluid grid `g` with `normal`s, of the particles that may emit."
function SurfacePotentials(g::FluidGrid, normal, w::Whitewater)
    backend = KA.get_backend(normal)
    n = length(g.x)
    ta = KA.allocate(backend, Float32, n); wc = similar(ta); ek = similar(ta)
    potentials_kernel!(backend)(ta, wc, ek, g, normal, w.stream; ndrange = n)
    x = Array(g.x); nr = Array(normal)
    s = findall(i -> sum(abs2, nr[i]) > 0f0 && x[i][3] > w.floor + 2w.h && hypot(x[i][1], x[i][2]) < w.rules.within, 1:n)
    return SurfacePotentials(Array(ta)[s], Array(wc)[s], Array(ek)[s])
end

"""
    SurfacePotentials(bake::WhirlpoolBake, k, rules, backend; spacing = 0.1)

Frame `k`'s potentials as its whitewater sees them: an SPH bake's particles with their
colour-field normals, or a FLIP bake's band gathered `spacing` apart with the level
set's lattice normals (see [`whitewater!`](@ref) for a [`BandFrame`](@ref)).
"""
function SurfacePotentials(bake::WhirlpoolBake{Nothing}, k, rules, backend; spacing = 0.1f0)
    w = Whitewater(bake.basin, backend; rules, capacity = 16)
    g = FluidGrid(Float32.(frame(bake.particles, k)), Float32.(velocity(bake.particles, k)), w.h, w.lo, w.hi, backend)
    normal = KA.allocate(backend, Vec3f, length(g.x)); count = KA.allocate(backend, Int32, length(g.x))
    normals_kernel!(backend)(normal, count, g; ndrange = length(g.x))
    return SurfacePotentials(g, normal, w)
end
function SurfacePotentials(bake::WhirlpoolBake{LevelSetFrames}, k, rules, backend; spacing = 0.1f0)
    b = Basin(; (f => (f == :spacing ? Float32(spacing) : getfield(bake.basin, f)) for f in fieldnames(Basin))...)
    w = Whitewater(b, backend; rules, capacity = 16)
    sampler = BandSampler(w.lo, w.hi, spacing, ceil(Int, (spacing / bake.basin.spacing)^3 / 2))
    xr, vr = representatives!(sampler, Float32.(frame(bake.particles, k)), Float32.(velocity(bake.particles, k)))
    g = FluidGrid(xr, vr, w.h, w.lo, w.hi, backend)
    ls = bake.levels
    φ = KA.allocate(backend, UInt8, size(ls.data)[1:3]...)
    copyto!(φ, ls.data[:, :, :, k])
    normal = KA.allocate(backend, Vec3f, length(g.x))
    latticenormals_kernel!(backend, 256)(normal, g.x, LatticeWater(φ, ls.lo, ls.cell, ls.range), w.h, Float32(spacing),
                                         Int32(length(g.x)); ndrange = length(g.x))
    return SurfacePotentials(g, normal, w)
end

"How many diffuse particles a second particles with potentials `p` make under `rules`."
emitted(p::SurfacePotentials, r::WhitewaterRules) =
    sum(clampfn.(p.energy, Ref(r.energy)) .* (r.kair .* clampfn.(p.air, Ref(r.air)) .+ r.kcrest .* clampfn.(p.crest, Ref(r.crest))))

"`rules` with the fields in `changes` replaced."
Base.merge(r::WhitewaterRules, changes::NamedTuple) =
    WhitewaterRules(; (f => get(changes, f, getfield(r, f)) for f in fieldnames(WhitewaterRules))...)

"""
    matched(reference::SurfacePotentials, p::SurfacePotentials, rules) -> WhitewaterRules

`rules` for particles with potentials `p`, so they make the whitewater `rules` make of
the `reference`: the trapped-air and crest ranges where `p`'s quantiles are that the
reference's ranges sit at (the same share of the surface below, within and above each
range), and the emission rates scaled to the reference's emission. The kinetic energy
range stays: it is the water's speed, the same in both.
"""
function matched(reference::SurfacePotentials, p::SurfacePotentials, rules::WhitewaterRules)
    at(ref, mine, range) = Float32.((quantile(mine, mean(<(range[1]), ref)), quantile(mine, mean(<(range[2]), ref))))
    r = merge(rules, (; air = at(reference.air, p.air, rules.air), crest = at(reference.crest, p.crest, rules.crest)))
    m = Float32(emitted(reference, rules) / emitted(p, r))
    return merge(r, (; kair = m * r.kair, kcrest = m * r.kcrest))
end

# ─── the level set, kept ─────────────────────────────────────────────────────

"""
    LevelSets(path, s::FlipBasin)

The sink that keeps every frame's level set for the drawing, which tells the band's
bottom (water below, no particles) from the water's surface with it: averaged over
cells twice as wide, clamped to `±range` (the band and two cells), as UInt8 (0 the
deepest, 255 the highest) to `path.levelset`; the grid to `path.levelset.jls` when
closed.
"""
struct LevelSets{A}
    io::IOStream
    path::String
    coarse::A
    host::Array{UInt8,3}
    lo::Point3f
    cell::Float32
    range::Float32
    counts::Vector{Int}
end
function LevelSets(path::AbstractString, s::FlipBasin)
    dims = Int.(s.grid.n) .÷ 2
    return LevelSets(open(path * ".levelset", "w"), String(path), KA.zeros(s.backend, UInt8, dims...),
                     Array{UInt8}(undef, dims...), s.grid.lo, 2f0 * s.grid.dx, Float32((s.band + 2) * s.grid.dx), Int[])
end

@kernel inbounds=true function quantize_kernel!(q, @Const(φ), range::Float32)
    I, J, K = @index(Global, NTuple)
    s = 0f0
    for c in 0:1, b in 0:1, a in 0:1
        s += φ[2I - 1 + a, 2J - 1 + b, 2K - 1 + c]
    end
    f = clamp(0.125f0 * s / range, -1f0, 1f0)
    q[I, J, K] = unsafe_trunc(UInt8, round(127.5f0 * (f + 1f0)))
end

function record!(k::LevelSets, s::FlipBasin, u, v)
    quantize_kernel!(s.backend, WG3)(k.coarse, s.φ, k.range; ndrange = size(k.coarse))
    copyto!(k.host, k.coarse)
    write(k.io, k.host)
    push!(k.counts, length(k.host))
    return k
end

function Base.close(k::LevelSets)
    close(k.io)
    serialize(k.path * ".levelset.jls", (; dims = size(k.host), lo = k.lo, cell = k.cell, range = k.range,
                                         frames = length(k.counts)))
end

# ─── the bake ────────────────────────────────────────────────────────────────

"""
    BandParticles(path, b, backend; chunk = 4_000_000, keep, throat)

The narrow band's particles kept as a [`Particles`](@ref) sink keeps them (`keep` and
`throat` are its), with their velocities: packed as Float16 on the device a chunk at a
time, only what is kept crossing to the host.
"""
struct BandParticles{H<:AbstractMatrix{Float16},C<:AbstractVector{Int32}}
    sink::Particles
    pos::H
    vel::H
    count::C
end
BandParticles(path::AbstractString, b::Basin, backend; chunk = 4_000_000, kw...) =
    BandParticles(Particles(path, b; velocities = true, kw...), KA.allocate(backend, Float16, 3, chunk),
                  KA.allocate(backend, Float16, 3, chunk), KA.zeros(backend, Int32, 1))

"The kept particles among `first:first+n-1` (see `Particles`), each at the next free column: its position into `pos`, its velocity into `vel`, as Float16."
@kernel inbounds=true function packband_kernel!(pos, vel, count, @Const(x), @Const(vx), @Const(vy), @Const(vz), @Const(lag),
                                                keep::Float32, throat::Float32, first::Int32, n::Int32)
    t = @index(Global)
    if t <= n
        p = Int(first) + t - 1
        # Where the particle is at the frame's time: its lag on (ST-FLIP's re-synchronization).
        xp = x[p] + lag[p] * Vec3f(vx[p], vy[p], vz[p])
        if xp[3] > -keep || xp[1] * xp[1] + xp[2] * xp[2] < throat * throat
            m = KA.@atomic :monotonic count[1] += Int32(1)
            pos[1, m] = Float16(xp[1]); pos[2, m] = Float16(xp[2]); pos[3, m] = Float16(xp[3])
            vel[1, m] = Float16(vx[p]); vel[2, m] = Float16(vy[p]); vel[3, m] = Float16(vz[p])
        end
    end
end

function record!(k::BandParticles, s, u, v)
    ps, sink = s.particles, k.sink
    kept = packed!((sink.io => k.pos, sink.velocities => k.vel), k.count, ps.n) do first, n
        packband_kernel!(s.backend, 256)(k.pos, k.vel, k.count, ps.x, ps.vx, ps.vy, ps.vz, ps.lag, sink.keep, sink.throat,
                                         Int32(first), Int32(n); ndrange = n)
    end
    push!(sink.counts, kept)
    return k
end
Base.close(k::BandParticles) = close(k.sink)

"""
    perparticle(rules::WhitewaterRules, volume) -> WhitewaterRules

`rules` for particles `volume` times as large as the ones they were made for: their
emission rates scaled by it, so a volume of water makes as much whitewater either way.
"""
perparticle(r::WhitewaterRules, volume) =
    WhitewaterRules(; (f => (f in (:kair, :kcrest) ? Float32(volume) * getfield(r, f) : getfield(r, f))
                         for f in fieldnames(WhitewaterRules))...)

"""
    bakeflip(b, path, backend; dx = 0.1, tend = 16, from = 6, fps = 24, waves, race, whitewater, lead = 4, band = 4,
             cfl = 8, kernel = 5, capacity = 12_000_000, solver = FlipBasin, options = (;)) -> path

The basin `b` simulated as narrow-band FLIP on a grid `dx` apart for `tend` seconds,
stirred by the `race` with the sea's `waves` at its rim, and kept `fps` times a second
from `from` on in the files of the SPH bake (see `bakewhirlpool`): the narrow band's
particles with their velocities, the whitewater, and `path.basin.jls`, whose basin's
`spacing` is the particles' (half a cell), as the drawing reads it; and the level set
(see [`LevelSets`](@ref)), which tells the drawing where the band's bottom is. The
defaults are the SPH bake's.

Steps move the fastest water up to `cfl` cells (ST-FLIP, see `advectparticles_kernel!`),
at most a frame. The `solver` is the dense grid's (`FlipBasin`) or the sparse one's
(`SparseBasin`, sflip.jl), made with `options` too (`blocks`, `rings`). The `whitewater` rules are the SPH bake's, for particles `b.spacing` apart: the band's
are smaller, and emit as much less (see [`perparticle`](@ref)). The whitewater runs
from `lead` seconds before the first frame kept, so it has grown by then. The drawing
smooths the band's particles, which lie at random in their cells, not settled as SPH
particles are, over `kernel` spacings (the SPH bake's are drawn over 2.5).
"""
function bakeflip(b::Basin, path, backend; dx = 0.1f0, tend = 16f0, from = 6f0, fps = 24, waves = narutowaves(),
                  race = Race(; eddy = 1.2f0, eddyscale = 3f0, boil = 1f0, waves = 1f0),
                  whitewater = WhitewaterRules(; air = (0.5f0, 2.5f0), crest = (0.3f0, 3f0), energy = (0.05f0, 1f0),
                                               kair = 40f0, kcrest = 120f0, buoyancy = 0.6f0, life = (1f0, 4f0)),
                  lead = 4f0, band = 4, cfl = 8f0, kernel = 5f0, capacity = 12_000_000, whitecapacity = 4_000_000,
                  solver = FlipBasin, options = (;))
    isapprox(from * fps, round(from * fps); atol = 1e-6) ||
        throw(ArgumentError("a bake's frames start on a frame: $from is not a multiple of 1/$fps"))
    stir = Stir(b, waves; race)
    s = solver(b, backend; dx, stir, band, cfl, capacity, options...)
    initialize!(s)
    spacing = Float32(dx) / 2
    drawn = Basin(; (f => (f == :spacing ? spacing : getfield(b, f)) for f in fieldnames(Basin))...)
    rules = perparticle(whitewater, (spacing / b.spacing)^3)
    foam = WhitewaterFrames(path, drawn, backend; fps, rules, capacity = whitecapacity)
    sinks = (BandParticles(path, drawn, backend; keep = b.depth + 1f0, throat = b.radius), foam, LevelSets(path, s))
    start = round(Int, from * fps)
    stop = floor(Int, tend * fps + 1e-6)
    early = max(1, start - round(Int, lead * fps))
    clock = time()
    stepping = 0.0
    recording = 0.0
    for frame in 1:stop
        # Steps of at most `cfl` cells to the frame, all of a length.
        target = frame / fps
        stepping += @elapsed while s.t < target - 1e-9
            left = target - s.t
            dt = left / ceil(left / (cfl * dx / max(maxspeed(s), 1f-3)))
            step!(s, dt)
        end
        s.t = target
        recording += @elapsed if frame >= start
            # Everything a sink keeps is on the device: no frame on the host.
            foreach(k -> record!(k, s, nothing, nothing), sinks)
        elseif frame >= early
            whitewater!(foam.w, s, nothing, nothing, 1 / fps)
        end
        frame % fps == 0 && @info "flip basin" t = frame / fps particles = s.particles.n steps = s.steps umax = maxspeed(s) volume = volume(s) / s.volume0 iterations = maximum(s.iterations[max(1, end - 2fps):end]) stepping = round(stepping; digits = 1) recording = round(recording; digits = 1) seconds = round(time() - clock; digits = 1)
    end
    foreach(close, sinks)
    serialize(path * ".basin.jls", (; basin = drawn, fps = Float64(fps), from = Float64(start / fps),
                                     frames = stop - start + 1, waves = stir.waves, viscosity = "flip", solver = :flip,
                                     dx = Float32(dx), band = Float32(band * dx), kernel = Float32(kernel)))
    return path
end
