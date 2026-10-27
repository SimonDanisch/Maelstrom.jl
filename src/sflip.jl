# The Charybdis basin as narrow-band FLIP on a sparse, two-resolution block grid: the
# water within a block or two of its surface on cells `dx` wide, everything else on a
# grid four times coarser, the pressure solved over both at once.
#
# Braun, Bender & Thuerey 2025, "Adaptive Phase-Field-FLIP for Very Large Scale
# Two-Phase Fluid Simulation" (MSBG, github.com/tum-pbs/MSBG): fixed-size blocks of
# 16³ cells kept at fine resolution where the surface is, a dense block index; the
# pressure gradient across a resolution change discretized symmetrically (Losasso,
# Gibou & Fedkiw 2004), every fine sub-face coupling its fine cell to the coarse one;
# Galerkin multigrid by aggregation (Shao et al. 2022) as the preconditioner of a
# flexible (Polak-Ribière) conjugate gradient. Two resolutions here, not their 2:1
# graded levels: the coarse grid is dense, so the deep water, the air far above and the
# coarsest multigrid levels are plain arrays; fine blocks over it hold the detail, and
# the coarse cells under them hold their means.
# Braun, Winchenbach, Bender & Thuerey 2026, "Spatiotemporal FLIP" (ST-FLIP): the
# particles' time jitter and temporal deposition, for steps of many cells (see
# `advectparticles_kernel!` and `temporal` in flip.jl).
# Ferstl et al. 2016: the narrow band of particles, as in flip.jl.
#
# It reuses the dense solver's transfers, level-set and whitewater pieces (flip.jl).
# Everything a step does runs on the device; the host keeps the block table.

const BLOCK = Int32(16)          # fine cells a block is wide
const RATIO = Int32(4)           # fine cells a coarse cell is wide
const CPB = BLOCK ÷ RATIO        # coarse cells a block is wide
const BCELLS = BLOCK^3           # fine cells in a block
const BFACES = (BLOCK + Int32(1)) * BLOCK^2   # a block's fine faces along one axis, both boundary planes included

# ─── the grid ────────────────────────────────────────────────────────────────

"""
    SparseGrid(b::Basin; dx, top = b.rimwall, pad = 2)

The basin on blocks `16dx` wide: in x and y the cylinder and `pad` fine cells round it,
in z from a coarse cell under the floor to at least `top`; whole multiples of four
blocks, so the coarse grid (cells `4dx`) halves down to a few hundred cells.
"""
struct SparseGrid
    lo::Point3f
    dx::Float32
    nb::NTuple{3,Int32}
    radius::Float32
    floor::Float32
end

function SparseGrid(b::Basin; dx, top = b.rimwall, pad = 2)
    dx = Float32(dx)
    S = BLOCK * dx
    nxy = 4 * cld(ceil(Int, 2 * (b.radius + pad * dx) / S), 4)
    loz = -b.depth - RATIO * dx
    nz = 4 * cld(ceil(Int, (top - loz) / S), 4)
    return SparseGrid(Point3f(-nxy * S / 2, -nxy * S / 2, loz), dx, (Int32(nxy), Int32(nxy), Int32(nz)), b.radius, -b.depth)
end

coarsecells(g::SparseGrid) = Int.(g.nb .* CPB)
nblocks(g::SparseGrid) = Int(prod(g.nb))
@inline coarsedx(g::SparseGrid) = RATIO * g.dx

@inline issolid(g::SparseGrid, p) = p[1] * p[1] + p[2] * p[2] > g.radius * g.radius || p[3] < g.floor
@inline blockid(g::SparseGrid, bi, bj, bk) = bi + g.nb[1] * (bj + g.nb[2] * bk) + Int32(1)
@inline inblocks(g::SparseGrid, bi, bj, bk) =
    Int32(0) <= bi < g.nb[1] && Int32(0) <= bj < g.nb[2] && Int32(0) <= bk < g.nb[3]
"A fine cell's centre, from its global coordinates (from 0)."
@inline finecentre(g::SparseGrid, I, J, K) = g.lo + (Vec3f(I, J, K) + COFF) * g.dx
"A coarse cell's centre, from its coordinates (from 0)."
@inline coarsecentre(g::SparseGrid, I, J, K) = g.lo + (Vec3f(I, J, K) + COFF) * coarsedx(g)
"The position of the face along axis `a` on the low side of the cell (from 0) `(I, J, K)`, cells `h` wide."
@inline facepos(g::SparseGrid, a, I, J, K, h) = g.lo + (Vec3f(I, J, K) + faceoff(a)) * h
@inline unit(a) = (Int32(a == 1), Int32(a == 2), Int32(a == 3))

# ─── which blocks are fine ───────────────────────────────────────────────────

"""
The sparse grid's fine blocks as kernels see them: for every block its slot in the fine
pools (0: no fine block, the coarse grid alone), for every slot its block's coordinates
(`-1`s for a free slot).
"""
struct Topology{I,V}
    g::SparseGrid
    slot::I
    where::V
end
Adapt.@adapt_structure Topology

"A field on the sparse grid: its values in the fine blocks' pool, and on the whole coarse grid (under fine blocks, their means)."
struct Field{F,C}
    fine::F
    coarse::C
end
Adapt.@adapt_structure Field

@inline function cellslot(t::Topology, I, J, K)
    g = t.g
    bi = I >> 4; bj = J >> 4; bk = K >> 4
    inblocks(g, bi, bj, bk) || return Int32(0)
    return @inbounds t.slot[blockid(g, bi, bj, bk)]
end
"Pool index of a block's fine cell (local coordinates from 0)."
@inline fineindex(s, i, j, k) = (s - Int32(1)) * BCELLS + i + BLOCK * (j + BLOCK * k) + Int32(1)
"Pool index of fine cell `(I, J, K)` (global, from 0), or 0 if no fine block holds it."
@inline function fineat(t::Topology, I, J, K)
    s = cellslot(t, I, J, K)
    return s == Int32(0) ? Int32(0) : fineindex(s, I & Int32(15), J & Int32(15), K & Int32(15))
end
"Pool index of a block's fine face along `a` (local coordinates from 0, 0–16 along `a`)."
@inline function faceindex(s, a, i, j, k)
    f = a == 1 ? i + Int32(17) * (j + BLOCK * k) : (a == 2 ? i + BLOCK * (j + Int32(17) * k) : i + BLOCK * (j + BLOCK * k))
    return (s - Int32(1)) * BFACES + f + Int32(1)
end
"""
Pool index of the fine face along `a` on the low side of fine cell `(I, J, K)` (global,
from 0): in the cell's block if it is fine, else on the high plane of the block below
it along `a`, else 0. On a plane between two fine blocks both hold the face; this is the
upper block's copy, the one particles deposit into (see `syncplanes!`).
"""
@inline function finefaceat(t::Topology, a, I, J, K)
    s = cellslot(t, I, J, K)
    s != Int32(0) && return faceindex(s, a, I & Int32(15), J & Int32(15), K & Int32(15))
    e1, e2, e3 = unit(a)
    Im = I - e1; Jm = J - e2; Km = K - e3
    sm = cellslot(t, Im, Jm, Km)
    sm == Int32(0) && return Int32(0)
    # The face is on the lower block's high plane only if the two cells are in different blocks.
    ((Im >> 4) != (I >> 4) || (Jm >> 4) != (J >> 4) || (Km >> 4) != (K >> 4)) || return Int32(0)
    return faceindex(sm, a, a == 1 ? BLOCK : Im & Int32(15), a == 2 ? BLOCK : Jm & Int32(15), a == 3 ? BLOCK : Km & Int32(15))
end
"Whether coarse cell `(I, J, K)` (from 0) lies in a fine block."
@inline covered(t::Topology, I, J, K) = cellslot(t, I * RATIO, J * RATIO, K * RATIO) != Int32(0)

# ─── sampling ────────────────────────────────────────────────────────────────

@inline lerp8(c000, c100, c010, c110, c001, c101, c011, c111, fx, fy, fz) =
    ((c000 * (1f0 - fx) + c100 * fx) * (1f0 - fy) + (c010 * (1f0 - fx) + c110 * fx) * fy) * (1f0 - fz) +
    ((c001 * (1f0 - fx) + c101 * fx) * (1f0 - fy) + (c011 * (1f0 - fx) + c111 * fx) * fy) * fz

"""
The cell field `f` at `p`: trilinear on the fine cells where all eight it reads are in
fine blocks, trilinear on the coarse grid elsewhere.
"""
@inline function cellsample(t::Topology, f::Field, p)
    g = t.g
    q = Vec3f((p - g.lo) / g.dx) - COFF
    x = floor(q[1]); y = floor(q[2]); z = floor(q[3])
    I = unsafe_trunc(Int32, x); J = unsafe_trunc(Int32, y); K = unsafe_trunc(Int32, z)
    a000 = fineat(t, I, J, K); a100 = fineat(t, I + Int32(1), J, K)
    a010 = fineat(t, I, J + Int32(1), K); a110 = fineat(t, I + Int32(1), J + Int32(1), K)
    a001 = fineat(t, I, J, K + Int32(1)); a101 = fineat(t, I + Int32(1), J, K + Int32(1))
    a011 = fineat(t, I, J + Int32(1), K + Int32(1)); a111 = fineat(t, I + Int32(1), J + Int32(1), K + Int32(1))
    if a000 > 0 && a100 > 0 && a010 > 0 && a110 > 0 && a001 > 0 && a101 > 0 && a011 > 0 && a111 > 0
        F = f.fine
        @inbounds return lerp8(Float32(F[a000]), Float32(F[a100]), Float32(F[a010]), Float32(F[a110]), Float32(F[a001]),
                               Float32(F[a101]), Float32(F[a011]), Float32(F[a111]), q[1] - x, q[2] - y, q[3] - z)
    end
    return trilinear(f.coarse, Vec3f((p - g.lo) / coarsedx(g)) - COFF)
end

"The face field `f` along axis `a` at `p`: as `cellsample`, on the fine faces or the coarse grid's."
@inline function facesample(t::Topology, f::Field, a, p)
    g = t.g
    off = faceoff(a)
    q = Vec3f((p - g.lo) / g.dx) - off
    x = floor(q[1]); y = floor(q[2]); z = floor(q[3])
    I = unsafe_trunc(Int32, x); J = unsafe_trunc(Int32, y); K = unsafe_trunc(Int32, z)
    o = Int32(1)
    a000 = finefaceat(t, a, I, J, K); a100 = finefaceat(t, a, I + o, J, K)
    a010 = finefaceat(t, a, I, J + o, K); a110 = finefaceat(t, a, I + o, J + o, K)
    a001 = finefaceat(t, a, I, J, K + o); a101 = finefaceat(t, a, I + o, J, K + o)
    a011 = finefaceat(t, a, I, J + o, K + o); a111 = finefaceat(t, a, I + o, J + o, K + o)
    if a000 > 0 && a100 > 0 && a010 > 0 && a110 > 0 && a001 > 0 && a101 > 0 && a011 > 0 && a111 > 0
        F = f.fine
        @inbounds return lerp8(Float32(F[a000]), Float32(F[a100]), Float32(F[a010]), Float32(F[a110]), Float32(F[a001]),
                               Float32(F[a101]), Float32(F[a011]), Float32(F[a111]), q[1] - x, q[2] - y, q[3] - z)
    end
    return trilinear(f.coarse, Vec3f((p - g.lo) / coarsedx(g)) - off)
end

"The velocity at `p`."
@inline velocity(t::Topology, u::Field, v::Field, w::Field, p) =
    Vec3f(facesample(t, u, 1, p), facesample(t, v, 2, p), facesample(t, w, 3, p))

"Whether the velocity was carried out to `p` (see `reached` for the dense grid)."
@inline reached(t::Topology, fu::Field, fv::Field, fw::Field, p) =
    min(facesample(t, fu, 1, p), facesample(t, fv, 2, p), facesample(t, fw, 3, p)) > 0.999f0

@inline function inwalls(g::SparseGrid, p)
    m = 0.25f0 * g.dx
    r = sqrt(p[1] * p[1] + p[2] * p[2])
    r > g.radius - m && (p = Point3f(p[1] * (g.radius - m) / r, p[2] * (g.radius - m) / r, p[3]))
    p[3] < g.floor + m && (p = Point3f(p[1], p[2], g.floor + m))
    return p
end

"""
APIC's gather of one fine face component at `xp` (see `gather` for the dense grid):
the velocity, its affine part, and the weight of the faces it found (1 when all 27
are in fine blocks).
"""
@inline function finegather(t::Topology, u, a, xp)
    g = t.g
    dx = g.dx
    q = Vec3f((xp - g.lo) / dx) - faceoff(a)
    bx, x0, x1, x2 = bspline(q[1]); by, y0, y1, y2 = bspline(q[2]); bz, z0, z1, z2 = bspline(q[3])
    vel = 0f0; aff = Vec3f(0); found = 0f0
    for mz in Int32(0):Int32(2), my in Int32(0):Int32(2), mx in Int32(0):Int32(2)
        idx = finefaceat(t, a, Int32(bx) + mx, Int32(by) + my, Int32(bz) + mz)
        if idx > 0
            wgt = pick(mx, x0, x1, x2) * pick(my, y0, y1, y2) * pick(mz, z0, z1, z2)
            val = Float32(@inbounds u[idx])
            vel += wgt * val
            aff += (wgt * val) * Vec3f(bx + mx - q[1], by + my - q[2], bz + mz - q[3]) * dx
            found += wgt
        end
    end
    return vel, aff * (4f0 / (dx * dx)), found
end

# The particles' kernels (flip.jl) on the sparse grid: particles stay in fine
# blocks, so they reach fine faces and cells only.
@inline spacing(t::Topology) = t.g.dx
@inline origin(t::Topology) = t.g.lo
@inline nodeindex(t::Topology, A, a, I, J, K) = finefaceat(t, a, I, J, K)
@inline cellindex(t::Topology, A, I, J, K) = fineat(t, I, J, K)
@inline centre0(t::Topology, I, J, K) = finecentre(t.g, I, J, K)
@inline inwalls(t::Topology, p) = inwalls(t.g, p)
@inline velocity(f::Flow{<:Topology}, p) = velocity(f.g, f.u, f.v, f.w, p)
@inline reached(f::Flow{<:Topology}, p) = reached(f.g, f.fu, f.fv, f.fw, p)
@inline function gathered(f::Flow{<:Topology}, a, xp)
    vel, aff, _ = finegather(f.g, component(f, a).fine, a, xp)
    return vel, aff, finegather(f.g, flag(f, a).fine, a, xp)[1] > 0.999f0
end

# ─── the state ───────────────────────────────────────────────────────────────

"One level of the pressure solve's multigrid hierarchy: its unknowns, right-hand side, residual and smoothing buffer, the faces' coefficients (lower face of every cell, along each axis), the Dirichlet part of the diagonal and which cells are unknowns."
struct Level{X,C,A}
    x::X; b::X; r::X; t::X
    cx::C; cy::C; cz::C
    dD::X
    active::A
end

"""
    SparseBasin(b::Basin, backend; dx, stir, band = 4, nmin = 8, cfl = 8, capacity, blocks, rings = 0, vary = 0.5,
                reflect = true)

The basin as narrow-band FLIP on a [`SparseGrid`](@ref): fine blocks wherever there are
particles or surface, wherever the water's velocity changes by more than `vary` m/s
from one coarse cell to the next, and `rings` blocks round them (room for `blocks` of
them), the coarse grid everywhere. With `reflect`, each step is advection-reflection's
two halves (see `step!`). Otherwise as [`FlipBasin`](@ref).
"""
mutable struct SparseBasin{B,S,T,FF,FU,FI,FX,L1,LD,P,I1,F1,U3}
    basin::Basin
    grid::SparseGrid
    backend::B
    stir::S
    topo::T
    slots::Vector{Int32}             # the block table on the host
    wheres::Vector{Vec{3,Int32}}
    free::Vector{Int32}
    nslots::Int                      # slots in use, free ones among them included
    marks::U3; marks2::U3            # per block, while refining
    # cells
    φ::FF; φ0::FF; types::FU
    φp::FI
    P::FF; b::FF; r::FF; z::FF; d::FF; q::FF; rold::FF; dD::FF; tmp::FF
    # faces along x, y, z
    u::FF; v::FF; w::FF; u0::FF; v0::FF; w0::FF
    mu::FX; mv::FX; mw::FX; wu::FX; wv::FX; ww::FX
    fu::FU; fv::FU; fw::FU; fu2::FU; fv2::FU; fw2::FU
    cu::FF; cv::FF; cw::FF                 # the pressure solve's face coefficients
    ur::FF; vr::FF; wr::FF                 # advection-reflection: the reflected velocity
    ur0::FF; vr0::FF; wr0::FF              # and its extrapolation's scratch
    reach::U3; changed::I1                 # trapped air, on the coarse grid
    level1::L1                             # the fine blocks aggregated to 8³ over the coarse grid
    dense::Vector{LD}                      # the coarse grid and coarser, all dense
    particles::P
    counts::I1; totals::I1; offsets::I1; cellstart::I1
    key::I1; rank::I1
    partials::F1
    band::Int32
    nmin::Int32
    rings::Int
    vary::Float32                          # m/s from coarse cell to cell that makes a block fine
    reflect::Bool                          # each step as two halves, the first reflected (see `step!`)
    cfl::Float32
    volume0::Float32
    t::Float64
    steps::Int
    iterations::Vector{Int}
end

cells(s::SparseBasin) = coarsecells(s.grid)
const BLOCKS1 = Int32(512)       # a fine block's cells once aggregated to 8³
const BFACES1 = Int32(9 * 64)

"Upload a stirring model's waves without changing the host model."
device_stir(stir::Stir, backend) =
    Stir(stir.basin, Mantle.devicearray(backend, collect(stir.waves)), stir.origin, stir.offset, stir.race)

function SparseBasin(b::Basin, backend; dx, stir = Stir(b), band = 4, nmin = 8, cfl = 8f0, capacity = 30_000_000,
                     blocks = 8000, rings = 0, vary = 0.5f0, reflect = true)
    g = SparseGrid(b; dx)
    nc = coarsecells(g)
    F(n...) = KA.zeros(backend, Float32, n...)
    U(n...) = KA.zeros(backend, UInt8, n...)
    I(n...) = KA.zeros(backend, Int32, n...)
    nfine = blocks * Int(BCELLS); nface = blocks * Int(BFACES)
    cellF() = Field(F(nfine), F(nc...))
    faceF(a) = Field(F(nface), F((nc .+ unit(a))...))
    faceU(a) = Field(U(nface), U((nc .+ unit(a))...))
    topo = Topology(g, I(nblocks(g)), Mantle.devicearray(backend, fill(Vec{3,Int32}(-1, -1, -1), blocks)))
    # Level 1: the fine blocks at 8³; its coarse part shares the coarse grid's coefficients.
    n1 = blocks * Int(BLOCKS1); f1 = blocks * Int(BFACES1)
    level1 = Level(Field(F(n1), F(nc...)), Field(F(n1), F(nc...)), Field(F(n1), F(nc...)), Field(F(n1), F(nc...)),
                   F(f1), F(f1), F(f1), Field(F(n1), F(nc...)), U(n1))
    # The coarse grid and its halvings, until a workgroup can solve the coarsest.
    dense = Level[]
    m = nc
    while true
        push!(dense, Level(F(m...), F(m...), F(m...), F(m...), F((m .+ (1, 0, 0))...), F((m .+ (0, 1, 0))...),
                           F((m .+ (0, 0, 1))...), F(m...), U(m...)))
        (prod(m) <= 1024 || any(isodd, m)) && break
        m = m .÷ 2
    end
    prod(m) <= 1024 || error("the coarse grid $(nc) does not halve to at most 1024 cells")
    dense = [l for l in dense]
    devstir = device_stir(stir, backend)
    ncells = nfine
    s = SparseBasin(b, g, backend, devstir, topo, zeros(Int32, nblocks(g)), fill(Vec{3,Int32}(-1, -1, -1), blocks),
                    Int32[], 0, U(g.nb...), U(g.nb...),
                    cellF(), cellF(), Field(U(nfine), U(nc...)), I(nfine),
                    cellF(), cellF(), cellF(), cellF(), cellF(), cellF(), cellF(), cellF(), cellF(),
                    faceF(1), faceF(2), faceF(3), faceF(1), faceF(2), faceF(3),
                    F(nface), F(nface), F(nface), F(nface), F(nface), F(nface),
                    faceU(1), faceU(2), faceU(3), faceU(1), faceU(2), faceU(3),
                    faceF(1), faceF(2), faceF(3),
                    faceF(1), faceF(2), faceF(3), faceF(1), faceF(2), faceF(3),
                    U(nc...), I(1),
                    level1, dense,
                    FlipParticles(backend, capacity),
                    I(ncells), I(ncells), I(ncells), I(ncells + 1),
                    I(capacity), I(capacity),
                    F(cld(max(nface, prod(nc .+ 1)), 256)),
                    Int32(band), Int32(nmin), rings, Float32(vary), reflect, Float32(cfl), 0f0, 0.0, 0, Int[])
    return s
end

spacing(s::SparseBasin) = s.grid.dx
Flow(s::SparseBasin) = Flow(s.topo, s.u, s.v, s.w, s.fu, s.fv, s.fw)
finecells(s::SparseBasin) = s.nslots * Int(BCELLS)
finefaces(s::SparseBasin) = s.nslots * Int(BFACES)

# ─── threads over fine blocks ────────────────────────────────────────────────

"A thread's fine cell, from its index over the slots' cells: slot, local coordinates (from 0), the block's coordinates (`-1`s for a free slot)."
@inline function finecell(t::Topology, n)
    n0 = Int32(n - 1)
    s = (n0 >> 12) + Int32(1)
    c = n0 & Int32(4095)
    return s, c & Int32(15), (c >> 4) & Int32(15), c >> 8, @inbounds(t.where[s])
end

"A thread's fine face along `a`, from its index over the slots' faces: slot, local coordinates (0–16 along `a`), the block's coordinates."
@inline function fineface(t::Topology, a, n)
    n0 = Int32(n - 1)
    s = n0 ÷ BFACES + Int32(1)
    f = n0 - (s - Int32(1)) * BFACES
    if a == 1
        i = f % Int32(17); r = f ÷ Int32(17); j = r & Int32(15); k = r >> 4
    elseif a == 2
        i = f & Int32(15); r = f >> 4; j = r % Int32(17); k = r ÷ Int32(17)
    else
        i = f & Int32(15); j = (f >> 4) & Int32(15); k = f >> 8
    end
    return s, i, j, k, @inbounds(t.where[s])
end

@inline global0(w, i, j, k) = (w[1] * BLOCK + i, w[2] * BLOCK + j, w[3] * BLOCK + k)
@inline coarse0(i, j, k) = (Int32(i - 1), Int32(j - 1), Int32(k - 1))

# ─── refinement: which blocks are fine ───────────────────────────────────────

"Mark the blocks the surface passes through: a coarse cell (fine blocks' means under them) whose level set changes sign to a neighbour, or is within a cell and a half of zero."
@kernel inbounds=true function marksurface_kernel!(marks, @Const(φ), g::SparseGrid)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    if !issolid(g, coarsecentre(g, I, J, K))
        c = φ[i, j, k]
        surf = abs(c) < 1.5f0 * coarsedx(g)
        for m in 1:6
            a, b, cc = neighbour(i, j, k, m)
            if inrange(φ, a, b, cc) && !issolid(g, coarsecentre(g, coarse0(a, b, cc)...))
                (φ[a, b, cc] < 0f0) != (c < 0f0) && (surf = true)
            end
        end
        surf && (marks[(I >> 2) + 1, (J >> 2) + 1, (K >> 2) + 1] = 0x01)
    end
end

"A coarse cell's velocity: the means of its faces."
@inline cellvelocity(u, v, w, i, j, k) =
    0.5f0 * Vec3f(u[i, j, k] + u[i + 1, j, k], v[i, j, k] + v[i, j + 1, k], w[i, j, k] + w[i, j, k + 1])

"""
Mark the blocks where the water's velocity changes by more than `vary` from a coarse
cell to a water neighbour. The coarse grid smears the flow there: left coarse, the
vortex's core lost half its spin round the throat, and the funnel ended metres higher.
"""
@kernel inbounds=true function markflow_kernel!(marks, @Const(u), @Const(v), @Const(w), @Const(φ), g::SparseGrid, vary::Float32)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    if φ[i, j, k] < 0f0 && !issolid(g, coarsecentre(g, I, J, K))
        c = cellvelocity(u, v, w, i, j, k)
        jump = 0f0
        for m in 1:6
            a, b, cc = neighbour(i, j, k, m)
            if inrange(φ, a, b, cc) && φ[a, b, cc] < 0f0 && !issolid(g, coarsecentre(g, coarse0(a, b, cc)...))
                jump = max(jump, sqrt(sum(abs2, cellvelocity(u, v, w, a, b, cc) - c)))
            end
        end
        jump > vary && (marks[(I >> 2) + 1, (J >> 2) + 1, (K >> 2) + 1] = 0x01)
    end
end

"""
Mark the blocks within `reach` of every particle (the box round it, up to eight blocks):
where it can move in a step, and the faces its transfers read, must be fine.
"""
@kernel inbounds=true function markparticles_kernel!(marks, @Const(x), g::SparseGrid, reach::Float32, n::Int32)
    p = @index(Global)
    if p <= n
        xp = x[p]
        if isfinite(xp[1]) && isfinite(xp[2]) && isfinite(xp[3])
            S = BLOCK * g.dx
            for c in Int32(0):Int32(1), b in Int32(0):Int32(1), a in Int32(0):Int32(1)
                q = Vec3f((xp - g.lo + Vec3f(2a - 1, 2b - 1, 2c - 1) * reach) / S)
                bi = clamp(unsafe_trunc(Int32, floor(q[1])), Int32(0), g.nb[1] - Int32(1))
                bj = clamp(unsafe_trunc(Int32, floor(q[2])), Int32(0), g.nb[2] - Int32(1))
                bk = clamp(unsafe_trunc(Int32, floor(q[3])), Int32(0), g.nb[3] - Int32(1))
                marks[bi + 1, bj + 1, bk + 1] = 0x01
            end
        end
    end
end

"The marked blocks grown by one in every direction (26 neighbours)."
@kernel inbounds=true function dilate_kernel!(out, @Const(marks))
    i, j, k = @index(Global, NTuple)
    m = 0x00
    for dk in -1:1, dj in -1:1, di in -1:1
        inrange(marks, i + di, j + dj, k + dk) && marks[i + di, j + dj, k + dk] != 0x00 && (m = 0x01)
    end
    out[i, j, k] = m
end

"Zero a list of slots' part of the fine pool `A` (`per` values a slot)."
@kernel inbounds=true function zeroslots_kernel!(A, @Const(list), per::Int32)
    n = @index(Global)
    n0 = Int32(n - 1)
    s = list[n0 ÷ per + Int32(1)]
    A[(s - Int32(1)) * per + n0 % per + Int32(1)] = zero(eltype(A))
end

@inline fromcoarse(::Type{Float32}, x) = x
@inline fromcoarse(::Type{UInt8}, x) = x > 0.999f0 ? 0x01 : 0x00

"A list of new blocks' fine cells from the coarse grid (trilinear)."
@kernel inbounds=true function prolongcells_kernel!(fine, @Const(coarse), t::Topology, @Const(list))
    n = @index(Global)
    n0 = Int32(n - 1)
    s = list[(n0 >> 12) + Int32(1)]
    c = n0 & Int32(4095)
    i = c & Int32(15); j = (c >> 4) & Int32(15); k = c >> 8
    w = t.where[s]
    p = finecentre(t.g, global0(w, i, j, k)...)
    fine[fineindex(s, i, j, k)] = fromcoarse(eltype(fine), trilinear(coarse, Vec3f((p - t.g.lo) / coarsedx(t.g)) - COFF))
end

"A list of new blocks' fine faces along `a` from the coarse grid's."
@kernel inbounds=true function prolongfaces_kernel!(fine, @Const(coarse), t::Topology, @Const(list), a::Int32)
    n = @index(Global)
    n0 = Int32(n - 1)
    s = list[n0 ÷ BFACES + Int32(1)]
    f = n0 % BFACES
    if a == 1
        i = f % Int32(17); r = f ÷ Int32(17); j = r & Int32(15); k = r >> 4
    elseif a == 2
        i = f & Int32(15); r = f >> 4; j = r % Int32(17); k = r ÷ Int32(17)
    else
        i = f & Int32(15); j = (f >> 4) & Int32(15); k = f >> 8
    end
    w = t.where[s]
    p = facepos(t.g, a, global0(w, i, j, k)..., t.g.dx)
    fine[faceindex(s, a, i, j, k)] = fromcoarse(eltype(fine), trilinear(coarse, Vec3f((p - t.g.lo) / coarsedx(t.g)) - faceoff(a)))
end

"The fine pools a block's data lives in, cells and faces, for freeing a slot."
finepools(s::SparseBasin) =
    ((s.φ.fine, s.φ0.fine, s.types.fine, s.φp, s.P.fine, s.b.fine, s.r.fine, s.z.fine, s.d.fine, s.q.fine, s.rold.fine,
      s.dD.fine, s.tmp.fine, s.counts, s.totals),
     (s.u.fine, s.v.fine, s.w.fine, s.u0.fine, s.v0.fine, s.w0.fine, s.mu, s.mv, s.mw, s.wu, s.wv, s.ww, s.fu.fine,
      s.fv.fine, s.fw.fine, s.fu2.fine, s.fv2.fine, s.fw2.fine, s.cu.fine, s.cv.fine, s.cw.fine))

"""
    refine!(s) -> (added, removed)

Make fine the blocks the surface passes through, those within reach of a particle
(the `cfl` cells it may move in a step and the three its transfers read), those where
the water's velocity varies by more than `s.vary` from coarse cell to cell, and
`s.rings` rings of blocks round all of them; coarse the others. A new fine block starts
from the coarse grid (level set, pressure, velocity and its flags); a coarse one keeps
nothing but the coarse grid's means, which the step keeps up to date. Particles never
leave fine blocks.
"""
function refine!(s::SparseBasin)
    g, be, t = s.grid, s.backend, s.topo
    fill!(s.marks, 0x00)
    marksurface_kernel!(be, WG3)(s.marks, s.φ.coarse, g; ndrange = cells(s))
    markflow_kernel!(be, WG3)(s.marks, s.u.coarse, s.v.coarse, s.w.coarse, s.φ.coarse, g, s.vary; ndrange = cells(s))
    n = s.particles.n
    reach = Float32((ceil(s.cfl) + 3) * g.dx)
    n > 0 && markparticles_kernel!(be, 256)(s.marks, s.particles.x, g, reach, Int32(n); ndrange = n)
    for _ in 1:s.rings
        dilate_kernel!(be, (4, 4, 4))(s.marks2, s.marks; ndrange = size(s.marks))
        s.marks, s.marks2 = s.marks2, s.marks
    end
    want = vec(Array(s.marks))
    required=count(!iszero,want)
    required<=length(s.wheres) || error("the sparse grid needs $required fine blocks, more than its $(length(s.wheres)) allocated")
    added = Int32[]; removed = Int32[]
    # Release all departing blocks before allocating arrivals. Interleaving the two
    # can exhaust the high-water mark even when the final topology fits the pool.
    for blk in eachindex(want)
        if want[blk] == 0x00 && s.slots[blk] != 0
            slot = s.slots[blk]
            s.wheres[slot] = Vec{3,Int32}(-1, -1, -1)
            s.slots[blk] = 0
            push!(s.free, slot)
            push!(removed, slot)
        end
    end
    for blk in eachindex(want)
        if want[blk] != 0x00 && s.slots[blk] == 0
            slot=isempty(s.free) ? Int32(s.nslots+=1) : pop!(s.free)
            b0=blk-1
            s.wheres[slot]=Vec{3,Int32}(b0%g.nb[1],(b0÷g.nb[1])%g.nb[2],b0÷(g.nb[1]*g.nb[2]))
            s.slots[blk]=slot
            push!(added,slot)
        end
    end
    isempty(added) && isempty(removed) && return (0, 0)
    copyto!(t.slot, s.slots)
    copyto!(t.where, s.wheres)
    if !isempty(removed)
        list = Mantle.devicearray(be, removed)
        cellpools, facepools = finepools(s)
        for A in cellpools
            zeroslots_kernel!(be, 256)(A, list, BCELLS; ndrange = length(removed) * Int(BCELLS))
        end
        for A in facepools
            zeroslots_kernel!(be, 256)(A, list, BFACES; ndrange = length(removed) * Int(BFACES))
        end
    end
    if !isempty(added)
        list = Mantle.devicearray(be, added)
        for f in (s.φ, s.P)
            prolongcells_kernel!(be, 256)(f.fine, f.coarse, t, list; ndrange = length(added) * Int(BCELLS))
        end
        # The reflected velocity too: a step's second half carries it.
        for (a, fs) in ((1, (s.u, s.fu, s.ur)), (2, (s.v, s.fv, s.vr)), (3, (s.w, s.fw, s.wr))), f in fs
            prolongfaces_kernel!(be, 256)(f.fine, f.coarse, t, list, Int32(a); ndrange = length(added) * Int(BFACES))
        end
    end
    return (length(added), length(removed))
end

# ─── fine to coarse ──────────────────────────────────────────────────────────

@inline meanof(::Type{Float32}, s, n) = s / n
@inline meanof(::Type{UInt8}, s, n) = s >= n - 0.5f0 ? 0x01 : 0x00

"Coarse cells under fine blocks: the mean of their 64 fine cells (flags: all of them)."
@kernel inbounds=true function restrictcells_kernel!(coarse, @Const(fine), t::Topology)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    if covered(t, I, J, K)
        acc = 0f0
        for c in Int32(0):Int32(3), b in Int32(0):Int32(3), a in Int32(0):Int32(3)
            acc += Float32(fine[fineat(t, RATIO * I + a, RATIO * J + b, RATIO * K + c)])
        end
        coarse[i, j, k] = meanof(eltype(coarse), acc, 64f0)
    end
end

"Coarse faces along `a` that fine blocks hold: the mean of their 16 fine faces (flags: all of them)."
@kernel inbounds=true function restrictfaces_kernel!(coarse, @Const(fine), t::Topology, a::Int32)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    acc = 0f0
    found = Int32(0)
    for q in Int32(0):Int32(3), p in Int32(0):Int32(3)
        FI = RATIO * I + (a == 1 ? Int32(0) : p)
        FJ = RATIO * J + (a == 1 ? p : (a == 2 ? Int32(0) : q))
        FK = RATIO * K + (a == 3 ? Int32(0) : q)
        idx = finefaceat(t, a, FI, FJ, FK)
        if idx > 0
            acc += Float32(fine[idx]); found += Int32(1)
        end
    end
    found == Int32(16) && (coarse[i, j, k] = meanof(eltype(coarse), acc, 16f0))
end

restrict!(s::SparseBasin, f::Field) =
    (restrictcells_kernel!(s.backend, WG3)(f.coarse, f.fine, s.topo; ndrange = size(f.coarse)); s)
restrict!(s::SparseBasin, f::Field, a) =
    (restrictfaces_kernel!(s.backend, WG3)(f.coarse, f.fine, s.topo, Int32(a); ndrange = size(f.coarse)); s)
restrictvelocity!(s::SparseBasin) = (restrict!(s, s.u, 1); restrict!(s, s.v, 2); restrict!(s, s.w, 3); s)

"After the particles' deposit: a block's high-plane faces along `a` take the low-plane faces of the fine block above it, where the particles deposited (see `finefaceat`)."
@kernel inbounds=true function syncplanes_kernel!(A, t::Topology, a::Int32)
    n = @index(Global)
    n0 = Int32(n - 1)
    s = (n0 >> 8) + Int32(1)
    f = n0 & Int32(255)
    p = f & Int32(15); q = f >> 4
    w = t.where[s]
    if w[1] >= Int32(0)
        e1, e2, e3 = unit(a)
        if inblocks(t.g, w[1] + e1, w[2] + e2, w[3] + e3)
            s2 = t.slot[blockid(t.g, w[1] + e1, w[2] + e2, w[3] + e3)]
            if s2 != Int32(0)
                hi = a == 1 ? faceindex(s, a, BLOCK, p, q) : (a == 2 ? faceindex(s, a, p, BLOCK, q) : faceindex(s, a, p, q, BLOCK))
                lo = a == 1 ? faceindex(s2, a, Int32(0), p, q) : (a == 2 ? faceindex(s2, a, p, Int32(0), q) : faceindex(s2, a, p, q, Int32(0)))
                A[hi] = A[lo]
            end
        end
    end
end
syncplanes!(s::SparseBasin, A, a) = (syncplanes_kernel!(s.backend, 256)(A, s.topo, Int32(a); ndrange = s.nslots * 256); s)

# ─── the start ───────────────────────────────────────────────────────────────

@inline function restφ(η, rmax, p)
    r = sqrt(p[1] * p[1] + p[2] * p[2])
    t = clamp(r / rmax, 0f0, 1f0) * Float32(length(η) - 1)
    a = min(unsafe_trunc(Int, t), length(η) - 2)
    f = t - a
    return p[3] - (@inbounds(η[a + 1]) * (1f0 - f) + @inbounds(η[a + 2]) * f)
end
@inline function swirlvelocity(Γ, core, p, a)
    r = sqrt(p[1] * p[1] + p[2] * p[2])
    turn = r > 1f-4 ? oseen(Γ, core, r) / r : 0f0
    return a == 1 ? -turn * p[2] : (a == 2 ? turn * p[1] : 0f0)
end

@kernel inbounds=true function initcoarseφ_kernel!(φ, @Const(η), g::SparseGrid, rmax::Float32)
    i, j, k = @index(Global, NTuple)
    φ[i, j, k] = restφ(η, rmax, coarsecentre(g, coarse0(i, j, k)...))
end
@kernel inbounds=true function initfineφ_kernel!(φ, @Const(η), t::Topology, rmax::Float32)
    s, i, j, k, w = finecell(t, @index(Global))
    w[1] >= Int32(0) && (φ[fineindex(s, i, j, k)] = restφ(η, rmax, finecentre(t.g, global0(w, i, j, k)...)))
end
@kernel inbounds=true function initcoarseface_kernel!(u, g::SparseGrid, Γ::Float32, core::Float32, a::Int32)
    i, j, k = @index(Global, NTuple)
    u[i, j, k] = swirlvelocity(Γ, core, facepos(g, a, coarse0(i, j, k)..., coarsedx(g)), a)
end
@kernel inbounds=true function initfineface_kernel!(u, t::Topology, Γ::Float32, core::Float32, a::Int32)
    s, i, j, k, w = fineface(t, a, @index(Global))
    w[1] >= Int32(0) && (u[faceindex(s, a, i, j, k)] = swirlvelocity(Γ, core, facepos(t.g, a, global0(w, i, j, k)..., t.g.dx), a))
end

"""
    initialize!(s::SparseBasin)

The basin's water at rest in its vortex, as for the dense grid (see `initialize!`): the
balanced surface and the swirl on the coarse grid, fine blocks round the surface with
both on them too, the band filled with particles.
"""
function initialize!(s::SparseBasin)
    b, g, be, t = s.basin, s.grid, s.backend, s.topo
    rmax = hypot(g.lo[1], g.lo[2])
    η = Mantle.devicearray(be, Float32[surface(b, r) for r in range(0f0, rmax; length = 2048)])
    initcoarseφ_kernel!(be, WG3)(s.φ.coarse, η, g, Float32(rmax); ndrange = cells(s))
    for (a, f) in ((1, s.u), (2, s.v), (3, s.w))
        initcoarseface_kernel!(be, WG3)(f.coarse, g, b.Γ, b.core, Int32(a); ndrange = size(f.coarse))
    end
    refine!(s)
    initfineφ_kernel!(be, 256)(s.φ.fine, η, t, Float32(rmax); ndrange = finecells(s))
    for (a, f) in ((1, s.u), (2, s.v), (3, s.w))
        initfineface_kernel!(be, 256)(f.fine, t, b.Γ, b.core, Int32(a); ndrange = finefaces(s))
    end
    redistance!(s; iterations = 2s.band + 4)
    classify!(s)
    solidify!(s)
    for f in (s.fu, s.fv, s.fw)
        fill!(f.fine, 0x01); fill!(f.coarse, 0x01)
    end
    s.particles.n = 0
    resample!(s; seeding = true)
    resample!(s)
    copyto!(s.φ0.fine, s.φ.fine); copyto!(s.φ0.coarse, s.φ.coarse)
    for (f, f0) in ((s.u, s.u0), (s.v, s.v0), (s.w, s.w0))
        copyto!(f0.fine, f.fine); copyto!(f0.coarse, f.coarse)
    end
    particles2grid!(s, 1f0)
    redistance!(s)
    classify!(s)
    s.volume0 = volume(s)
    s.t = 0.0
    return s
end

# ─── advection ───────────────────────────────────────────────────────────────

"""
Back along the flow from `p` for `dt`, in RK2 sub-steps of at most `cells` cells `h`
wide: the grid's back-traces need no particle's one-cell sub-steps (RK2 follows the
vortex's curve over a few cells), and they run for every cell and face.
"""
@inline function backtrace(flow, p, dt, h; cells = 4f0)
    m = substeps(sqrt(sum(abs2, velocity(flow, p))) * dt / (cells * h))
    k = dt / m
    for _ in 1:m
        v1 = velocity(flow, p)
        v2 = velocity(flow, p - 0.5f0 * k * v1)
        p = p - k * v2
    end
    return p
end

@kernel inbounds=true function advectfinecells_kernel!(out, src, flow, dt::Float32)
    t = flow.g
    s, i, j, k, w = finecell(t, @index(Global))
    if w[1] >= Int32(0)
        p = backtrace(flow, finecentre(t.g, global0(w, i, j, k)...), dt, t.g.dx)
        out[fineindex(s, i, j, k)] = cellsample(t, src, p)
    end
end
@kernel inbounds=true function advectfinefaces_kernel!(out, src, flow, a::Int32, dt::Float32)
    t = flow.g
    s, i, j, k, w = fineface(t, a, @index(Global))
    if w[1] >= Int32(0)
        p = backtrace(flow, facepos(t.g, a, global0(w, i, j, k)..., t.g.dx), dt, t.g.dx)
        out[faceindex(s, a, i, j, k)] = facesample(t, src, a, p)
    end
end
@kernel inbounds=true function advectcoarsecells_kernel!(out, src, flow, dt::Float32)
    t = flow.g
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    if !covered(t, I, J, K)
        p = backtrace(flow, coarsecentre(t.g, I, J, K), dt, coarsedx(t.g))
        out[i, j, k] = cellsample(t, src, p)
    end
end
@kernel inbounds=true function advectcoarsefaces_kernel!(out, src, flow, a::Int32, dt::Float32)
    t = flow.g
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    p = backtrace(flow, facepos(t.g, a, I, J, K, coarsedx(t.g)), dt, coarsedx(t.g))
    out[i, j, k] = facesample(t, src, a, p)
end

"""
    advect!(s::SparseBasin, dt; carried = (s.u, s.v, s.w))

The particles, the level set and the `carried` velocity (the reflected one in a step's
second half, see `step!`) along the grid's divergence-free velocity, into `φ0` and
`u0`, `v0`, `w0`.
"""
function advect!(s::SparseBasin, dt; carried = (s.u, s.v, s.w))
    be, ps = s.backend, s.particles
    flow = Flow(s)
    n = ps.n
    n > 0 && advectparticles_kernel!(be, 256)(ps.x, ps.lag, ps.vx, ps.vy, ps.vz, flow, Float32(dt), s.basin.g,
                                              Int32(s.steps), Int32(n); ndrange = n)
    advectfinecells_kernel!(be, 256)(s.φ0.fine, s.φ, flow, Float32(dt); ndrange = finecells(s))
    advectcoarsecells_kernel!(be, WG3)(s.φ0.coarse, s.φ, flow, Float32(dt); ndrange = cells(s))
    for (a, f, f0) in ((1, carried[1], s.u0), (2, carried[2], s.v0), (3, carried[3], s.w0))
        advectfinefaces_kernel!(be, 256)(f0.fine, f, flow, Int32(a), Float32(dt); ndrange = finefaces(s))
        advectcoarsefaces_kernel!(be, WG3)(f0.coarse, f, flow, Int32(a), Float32(dt); ndrange = size(f0.coarse))
    end
    return s
end

# ─── particles to grid ───────────────────────────────────────────────────────

@kernel inbounds=true function sparseunion_kernel!(φ, @Const(φ0), @Const(φp), t::Topology)
    n = @index(Global)
    s, i, j, k, w = finecell(t, n)
    w[1] >= Int32(0) && (φ[n] = min(φ0[n] + t.g.dx, orderfloat(φp[n])))
end

"Ferstl et al.'s eq. 3 on the fine faces, as `combine_kernel!` on the dense grid: the particles' velocity within `r` of the surface, the grid's where they lack weight."
@kernel inbounds=true function sparsecombine_kernel!(u, @Const(u0), @Const(mom), @Const(wt), φ, t::Topology, r::Float32,
                                                     full::Float32, a::Int32)
    s, i, j, k, w = fineface(t, a, @index(Global))
    if w[1] >= Int32(0)
        idx = faceindex(s, a, i, j, k)
        φf = cellsample(t, φ, facepos(t.g, a, global0(w, i, j, k)..., t.g.dx))
        wgt = wt[idx]
        u[idx] = φf >= -r ? (mom[idx] + max(full - wgt, 0f0) * u0[idx]) / max(full, wgt) : u0[idx]
    end
end

function particles2grid!(s::SparseBasin, dt)
    be, ps, t = s.backend, s.particles, s.topo
    n = ps.n
    for A in (s.mu, s.mv, s.mw, s.wu, s.wv, s.ww)
        fill!(A, 0f0)
    end
    fill!(s.φp, orderkey(1f30))
    if n > 0
        for (mom, wt, va, ca, a) in ((s.mu, s.wu, ps.vx, ps.cx, 1), (s.mv, s.wv, ps.vy, ps.cy, 2), (s.mw, s.ww, ps.vz, ps.cz, 3))
            p2g_kernel!(be, 256)(mom, wt, ps.x, va, ca, ps.lag, t, Int32(a), Float32(dt), Int32(n); ndrange = n)
            syncplanes!(s, mom, a); syncplanes!(s, wt, a)
        end
        splat_kernel!(be, 256)(s.φp, ps.x, ps.lag, ps.vx, ps.vy, ps.vz, t, 0.5f0 * s.grid.dx, Float32(dt), Int32(n); ndrange = n)
    end
    sparseunion_kernel!(be, 256)(s.φ.fine, s.φ0.fine, s.φp, t; ndrange = finecells(s))
    copyto!(s.φ.coarse, s.φ0.coarse)
    restrict!(s, s.φ)
    r = Float32((s.band - 1) * s.grid.dx)
    for (a, f, f0, mom, wt) in ((1, s.u, s.u0, s.mu, s.wu), (2, s.v, s.v0, s.mv, s.wv), (3, s.w, s.w0, s.mw, s.ww))
        sparsecombine_kernel!(be, 256)(f.fine, f0.fine, mom, wt, s.φ, t, r, Float32(s.nmin), Int32(a); ndrange = finefaces(s))
        copyto!(f.coarse, f0.coarse)
    end
    return restrictvelocity!(s)
end

# ─── the level set ───────────────────────────────────────────────────────────

"""
One Jacobi sweep of redistancing on the fine cells (see `reinit_kernel!`): a neighbour
outside fine blocks counts with the coarse grid's value there, which is clamped deep in
the water or the air anyway (fine blocks reach further from the surface than the
redistanced band).
"""
@kernel inbounds=true function sparsereinit_kernel!(out, φ, t::Topology, φmax::Float32)
    n = @index(Global)
    s, i, j, k, w = finecell(t, n)
    if w[1] >= Int32(0)
        g = t.g
        I, J, K = global0(w, i, j, k)
        c = φ.fine[n]
        h = g.dx
        if issolid(g, finecentre(g, I, J, K))
            lowest = c; found = false
            for m in 1:6
                a, b, cc = neighbour(I, J, K, m)
                if !issolid(g, finecentre(g, a, b, cc))
                    f = neighbourφ(t, φ, a, b, cc)
                    lowest = found ? min(lowest, f) : f
                    found = true
                end
            end
            out[n] = found ? lowest : clamp(c, -φmax, φmax)
        else
            inside = c < 0f0
            frozen = false
            ax = φmax; ay = φmax; az = φmax
            for m in 1:6
                a, b, cc = neighbour(I, J, K, m)
                if !issolid(g, finecentre(g, a, b, cc))
                    f = neighbourφ(t, φ, a, b, cc)
                    (f < 0f0) != inside && (frozen = true)
                    d = abs(f)
                    m <= 2 ? (ax = min(ax, d)) : (m <= 4 ? (ay = min(ay, d)) : (az = min(az, d)))
                end
            end
            if frozen
                out[n] = c
            else
                d = min(eikonal(ax, ay, az, h), φmax)
                out[n] = inside ? -d : d
            end
        end
    end
end
"The level set at fine cell `(I, J, K)`: its own value in a fine block, else the coarse cell's it lies in."
@inline function neighbourφ(t::Topology, φ, I, J, K)
    idx = fineat(t, I, J, K)
    n1, n2, n3 = size(φ.coarse)
    return ifelse(idx > 0, @inbounds(φ.fine[max(idx,Int32(1))]),
        @inbounds(φ.coarse[clamp((I >> 2) + 1, 1, n1), clamp((J >> 2) + 1, 1, n2), clamp((K >> 2) + 1, 1, n3)]))
end

"Coarse wall cells outside fine blocks take the lowest level set of the water next to them, as fine ones do (see `sparsereinit_kernel!`): a wall is wet where water touches it."
@kernel inbounds=true function coarsewall_kernel!(out, @Const(φ), t::Topology)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    c = φ[i, j, k]
    if !covered(t, I, J, K) && issolid(t.g, coarsecentre(t.g, I, J, K))
        lowest = c; found = false
        for m in 1:6
            a, b, cc = neighbour(i, j, k, m)
            if inrange(φ, a, b, cc) && !issolid(t.g, coarsecentre(t.g, coarse0(a, b, cc)...))
                lowest = found ? min(lowest, φ[a, b, cc]) : φ[a, b, cc]
                found = true
            end
        end
        out[i, j, k] = lowest
    else
        out[i, j, k] = c
    end
end

function redistance!(s::SparseBasin; iterations = s.band + 3)
    φmax = Float32((iterations + 1) * s.grid.dx)
    for _ in 1:iterations
        sparsereinit_kernel!(s.backend, 256)(s.φ0.fine, s.φ, s.topo, φmax; ndrange = finecells(s))
        s.φ, s.φ0 = Field(s.φ0.fine, s.φ.coarse), Field(s.φ.fine, s.φ0.coarse)
    end
    coarsewall_kernel!(s.backend, WG3)(s.φ0.coarse, s.φ.coarse, s.topo; ndrange = cells(s))
    s.φ, s.φ0 = Field(s.φ.fine, s.φ0.coarse), Field(s.φ0.fine, s.φ.coarse)
    return restrict!(s, s.φ)
end

@kernel inbounds=true function classifyfine_kernel!(types, @Const(φ), t::Topology)
    n = @index(Global)
    s, i, j, k, w = finecell(t, n)
    w[1] >= Int32(0) &&
        (types[n] = issolid(t.g, finecentre(t.g, global0(w, i, j, k)...)) ? SOLID : (φ[n] < 0f0 ? LIQUID : AIR))
end
"Coarse cells: from their own level set outside fine blocks; under them, air if any fine cell is, else water if any is, else wall."
@kernel inbounds=true function classifycoarse_kernel!(types, @Const(φ), @Const(fine), t::Topology)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    if covered(t, I, J, K)
        air = false; liquid = false
        for c in Int32(0):Int32(3), b in Int32(0):Int32(3), a in Int32(0):Int32(3)
            ty = fine[fineat(t, RATIO * I + a, RATIO * J + b, RATIO * K + c)]
            ty == AIR && (air = true); ty == LIQUID && (liquid = true)
        end
        types[i, j, k] = air ? AIR : (liquid ? LIQUID : SOLID)
    else
        types[i, j, k] = issolid(t.g, coarsecentre(t.g, I, J, K)) ? SOLID : (φ[i, j, k] < 0f0 ? LIQUID : AIR)
    end
end
function classify!(s::SparseBasin)
    classifyfine_kernel!(s.backend, 256)(s.types.fine, s.φ.fine, s.topo; ndrange = finecells(s))
    classifycoarse_kernel!(s.backend, WG3)(s.types.coarse, s.φ.coarse, s.types.fine, s.topo; ndrange = cells(s))
    return s
end

# ─── trapped air ─────────────────────────────────────────────────────────────

"Fill the air the coarse sweeps did not reach (see `filltrapped!` on the dense grid): fine cells under an unreached coarse cell, coarse cells outside fine blocks."
@kernel inbounds=true function fillfine_kernel!(types, φ, @Const(reach), @Const(coarsetypes), t::Topology, depth::Float32)
    n = @index(Global)
    s, i, j, k, w = finecell(t, n)
    if w[1] >= Int32(0) && types[n] == AIR
        I, J, K = global0(w, i, j, k)
        ci = (I >> 2) + 1; cj = (J >> 2) + 1; ck = (K >> 2) + 1
        if reach[ci, cj, ck] == 0x00 && coarsetypes[ci, cj, ck] == AIR
            types[n] = LIQUID
            φ[n] = min(φ[n], -depth)
        end
    end
end
@kernel inbounds=true function fillcoarse_kernel!(types, φ, @Const(reach), t::Topology, depth::Float32)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    if !covered(t, I, J, K) && types[i, j, k] == AIR && reach[i, j, k] == 0x00
        types[i, j, k] = LIQUID
        φ[i, j, k] = min(φ[i, j, k], -depth)
    end
end

function filltrapped!(s::SparseBasin; maxrounds = 32)
    be = s.backend
    ct = s.types.coarse
    n1, n2, n3 = size(ct)
    seedair_kernel!(be, WG3)(s.reach, ct; ndrange = size(ct))
    for _ in 1:maxrounds
        fill!(s.changed, Int32(0))
        sweepair_kernel!(be, (16, 16))(s.reach, s.changed, ct, Int32(3); ndrange = (n1, n2))
        sweepair_kernel!(be, (16, 16))(s.reach, s.changed, ct, Int32(1); ndrange = (n2, n3))
        sweepair_kernel!(be, (16, 16))(s.reach, s.changed, ct, Int32(2); ndrange = (n1, n3))
        Array(s.changed)[1] == 0 && break
    end
    fillfine_kernel!(be, 256)(s.types.fine, s.φ.fine, s.reach, ct, s.topo, 2f0 * s.grid.dx; ndrange = finecells(s))
    fillcoarse_kernel!(be, WG3)(ct, s.φ.coarse, s.reach, s.topo, 2f0 * coarsedx(s.grid); ndrange = size(ct))
    return s
end

# ─── forces and walls ────────────────────────────────────────────────────────

"Gravity and the race's stirring on the fine faces along `a` (see `forces_kernel!`)."
@kernel inbounds=true function sparseforcesfine_kernel!(u, flow, stir, a::Int32, dt::Float32, time::Float32, ρ0::Float32,
                                                        gravity::Float32)
    t = flow.g
    s, i, j, k, w = fineface(t, a, @index(Global))
    if w[1] >= Int32(0)
        p = facepos(t.g, a, global0(w, i, j, k)..., t.g.dx)
        if !issolid(t.g, p)
            acc = stir(p, velocity(flow, p), ρ0, 0f0, time)
            u[faceindex(s, a, i, j, k)] += dt * (a == 1 ? acc[1] : (a == 2 ? acc[2] : acc[3] - gravity))
        end
    end
end
@kernel inbounds=true function sparseforcescoarse_kernel!(u, flow, stir, a::Int32, dt::Float32, time::Float32, ρ0::Float32,
                                                          gravity::Float32)
    t = flow.g
    i, j, k = @index(Global, NTuple)
    p = facepos(t.g, a, coarse0(i, j, k)..., coarsedx(t.g))
    if !issolid(t.g, p)
        acc = stir(p, velocity(flow, p), ρ0, 0f0, time)
        u[i, j, k] += dt * (a == 1 ? acc[1] : (a == 2 ? acc[2] : acc[3] - gravity))
    end
end

"Whether the face between the cells centred at `p - e h/2` and `p + e h/2` has a wall on either side."
@inline walledface(g::SparseGrid, p, a, h) =
    (e = Vec3f(unit(a)...) * (0.5f0 * h); issolid(g, p - e) || issolid(g, p + e))

@kernel inbounds=true function sparsesolidfine_kernel!(u, t::Topology, a::Int32)
    s, i, j, k, w = fineface(t, a, @index(Global))
    w[1] >= Int32(0) && walledface(t.g, facepos(t.g, a, global0(w, i, j, k)..., t.g.dx), a, t.g.dx) &&
        (u[faceindex(s, a, i, j, k)] = 0f0)
end
@kernel inbounds=true function sparsesolidcoarse_kernel!(u, g::SparseGrid, a::Int32)
    i, j, k = @index(Global, NTuple)
    h = coarsedx(g)
    I, J, K = coarse0(i, j, k)
    e1, e2, e3 = unit(a)
    nc = size(u) .- (e1, e2, e3)
    outside = (a == 1 && (I == 0 || I == nc[1])) || (a == 2 && (J == 0 || J == nc[2])) || (a == 3 && (K == 0 || K == nc[3]))
    (outside || walledface(g, facepos(g, a, I, J, K, h), a, h)) && (u[i, j, k] = 0f0)
end

function solidify!(s::SparseBasin)
    be = s.backend
    for (a, f) in ((1, s.u), (2, s.v), (3, s.w))
        sparsesolidfine_kernel!(be, 256)(f.fine, s.topo, Int32(a); ndrange = finefaces(s))
        sparsesolidcoarse_kernel!(be, WG3)(f.coarse, s.grid, Int32(a); ndrange = size(f.coarse))
    end
    return s
end

function forces!(s::SparseBasin, dt)
    be, b = s.backend, s.basin
    for (f, f0) in ((s.u, s.u0), (s.v, s.v0), (s.w, s.w0))
        copyto!(f0.fine, f.fine); copyto!(f0.coarse, f.coarse)
    end
    old = Flow(s.topo, s.u0, s.v0, s.w0, s.fu, s.fv, s.fw)
    for (a, f) in ((1, s.u), (2, s.v), (3, s.w))
        sparseforcesfine_kernel!(be, 256)(f.fine, old, s.stir, Int32(a), Float32(dt), Float32(s.t), b.ρ0, b.g;
                                          ndrange = finefaces(s))
        sparseforcescoarse_kernel!(be, WG3)(f.coarse, old, s.stir, Int32(a), Float32(dt), Float32(s.t), b.ρ0, b.g;
                                            ndrange = size(f.coarse))
    end
    solidify!(s)
    return restrictvelocity!(s)
end

# ─── the pressure ────────────────────────────────────────────────────────────
#
# Unknown P = p Δt/ρ in every water cell, fine and coarse. Per cell, in units of the
# fine cell `h`: Σ_faces c (P_c - P_n) + D P_c = -Σ_faces (A/h) u·n + γ V/h, with
# c = A/(d h): 1 between fine cells, 4 between coarse ones, 0.4 on each of the 16 fine
# sub-faces between a fine and a coarse cell (centres 2.5 h apart; Losasso et al.'s
# symmetric coupling, stored once, on the fine block's face). `D` holds the air
# neighbours' share (the ghost fluid's c/θ). The levels of the multigrid are the fine
# blocks at 16³ and at 8³ over the coarse grid, then the coarse grid alone and its
# halvings; each level's coefficients are the sums of the finer one's (Galerkin by
# aggregation, Braun et al.'s algorithm 1).

const FINE_FINE = 1f0
const FINE_COARSE = 0.4f0
const COARSE_COARSE = 4f0
const OMEGA = 6f0 / 7f0          # Jacobi's damping, the 3D optimum for smoothing

@inline log2of(::Val{16}) = Int32(4)
@inline log2of(::Val{8}) = Int32(3)
@inline percell(::Val{N}) where {N} = Int32(N^3)
@inline perface(::Val{N}) where {N} = Int32((N + 1) * N^2)
@inline cellidx(::Val{N}, s, i, j, k) where {N} = (s - Int32(1)) * Int32(N^3) + i + Int32(N) * (j + Int32(N) * k) + Int32(1)
@inline function faceidx(::Val{N}, s, a, i, j, k) where {N}
    n = Int32(N); m = Int32(N + 1)
    f = a == 1 ? i + m * (j + n * k) : (a == 2 ? i + n * (j + m * k) : i + n * (j + n * k))
    return (s - Int32(1)) * Int32((N + 1) * N^2) + f + Int32(1)
end
"Index of a level's cell `(I, J, K)` (global at that level, from 0), or 0 outside fine blocks."
@inline function cellat(t::Topology, v::Val{N}, I, J, K) where {N}
    sh = log2of(v)
    bi = I >> sh; bj = J >> sh; bk = K >> sh
    inblocks(t.g, bi, bj, bk) || return Int32(0)
    s = @inbounds t.slot[blockid(t.g, bi, bj, bk)]
    m = Int32(N - 1)
    return s == Int32(0) ? Int32(0) : cellidx(v, s, I & m, J & m, K & m)
end
"Index of a level's face along `a` on the low side of cell `(I, J, K)`, in either block that holds it, or 0 (see `finefaceat`)."
@inline function faceat(t::Topology, v::Val{N}, a, I, J, K) where {N}
    sh = log2of(v); m = Int32(N - 1)
    bi = I >> sh; bj = J >> sh; bk = K >> sh
    if inblocks(t.g, bi, bj, bk)
        s = @inbounds t.slot[blockid(t.g, bi, bj, bk)]
        s != Int32(0) && return faceidx(v, s, a, I & m, J & m, K & m)
    end
    e1, e2, e3 = unit(a)
    Im = I - e1; Jm = J - e2; Km = K - e3
    ((Im >> sh) != bi || (Jm >> sh) != bj || (Km >> sh) != bk) || return Int32(0)
    inblocks(t.g, Im >> sh, Jm >> sh, Km >> sh) || return Int32(0)
    s = @inbounds t.slot[blockid(t.g, Im >> sh, Jm >> sh, Km >> sh)]
    s == Int32(0) && return Int32(0)
    n = Int32(N)
    return faceidx(v, s, a, a == 1 ? n : Im & m, a == 2 ? n : Jm & m, a == 3 ? n : Km & m)
end
"A thread's cell of a level's fine part (`N`³ a block)."
@inline function levelcell(t::Topology, v::Val{N}, n) where {N}
    n0 = Int32(n - 1)
    sh = 3 * log2of(v)
    s = (n0 >> sh) + Int32(1)
    c = n0 & (percell(v) - Int32(1))
    m = Int32(N - 1); l = log2of(v)
    return s, c & m, (c >> l) & m, c >> (2l), @inbounds(t.where[s])
end
"A thread's face along `a` of a level's fine part."
@inline function levelface(t::Topology, v::Val{N}, a, n) where {N}
    n0 = Int32(n - 1)
    s = n0 ÷ perface(v) + Int32(1)
    f = n0 - (s - Int32(1)) * perface(v)
    nn = Int32(N); mm = Int32(N + 1)
    if a == 1
        i = f % mm; r = f ÷ mm; j = r % nn; k = r ÷ nn
    elseif a == 2
        i = f % nn; r = f ÷ nn; j = r % mm; k = r ÷ mm
    else
        i = f % nn; r = f ÷ nn; j = r % nn; k = r ÷ nn
    end
    return s, i, j, k, @inbounds(t.where[s])
end
"How many of a level's fine cells make one coarse cell along an axis, and its shift."
@inline coarseshift(v::Val{N}) where {N} = log2of(v) - Int32(2)

"""
`A x` at a level's fine cell: the diagonal and the sum of the neighbours' couplings,
`A x = diag x_c - off`. A neighbour outside fine blocks is the coarse cell there.
"""
@inline function finerow(t::Topology, v::Val{N}, x, xc, cx, cy, cz, dD, s, i, j, k, w) where {N}
    I = w[1] * Int32(N) + i; J = w[2] * Int32(N) + j; K = w[3] * Int32(N) + k
    sh = coarseshift(v)
    diag = @inbounds dD[cellidx(v, s, i, j, k)]
    off = 0f0
    n1, n2, n3 = size(xc)
    # Select loaded values instead of branching between fine and coarse loads.
    # The branching form produced invalid neighbour values on NVIDIA SPIR-V,
    # despite valid indices and finite input buffers. Both safe loads are explicit.
    neighbourvalue(In,Jn,Kn) = begin
        idx = cellat(t,v,In,Jn,Kn)
        ifelse(idx > 0, @inbounds(x[max(idx,Int32(1))]),
            @inbounds(xc[clamp((In>>sh)+1,1,n1),clamp((Jn>>sh)+1,1,n2),clamp((Kn>>sh)+1,1,n3)]))
    end
    c = @inbounds cx[faceidx(v,s,Int32(1),i,j,k)]
    c > 0f0 && (diag+=c; off+=c*neighbourvalue(I-Int32(1),J,K))
    c = @inbounds cx[faceidx(v,s,Int32(1),i+Int32(1),j,k)]
    c > 0f0 && (diag+=c; off+=c*neighbourvalue(I+Int32(1),J,K))
    c = @inbounds cy[faceidx(v,s,Int32(2),i,j,k)]
    c > 0f0 && (diag+=c; off+=c*neighbourvalue(I,J-Int32(1),K))
    c = @inbounds cy[faceidx(v,s,Int32(2),i,j+Int32(1),k)]
    c > 0f0 && (diag+=c; off+=c*neighbourvalue(I,J+Int32(1),K))
    c = @inbounds cz[faceidx(v,s,Int32(3),i,j,k)]
    c > 0f0 && (diag+=c; off+=c*neighbourvalue(I,J,K-Int32(1)))
    c = @inbounds cz[faceidx(v,s,Int32(3),i,j,k+Int32(1))]
    c > 0f0 && (diag+=c; off+=c*neighbourvalue(I,J,K+Int32(1)))
    return diag, off
end

"""
`A x` at a coarse cell outside fine blocks: its coarse neighbours through the coarse
faces, a fine block's cells through the block's sub-faces.
"""
@inline function coarserow(t::Topology, v::Val{N}, xf, cfx, cfy, cfz, xc, ccx, ccy, ccz, dDc, i, j, k) where {N}
    I, J, K = coarse0(i, j, k)
    R = Int32(N ÷ 4)
    diag = @inbounds dDc[i, j, k]
    off = 0f0
    n1, n2, n3 = size(xc)
    for a in Int32(1):Int32(3)
        e1, e2, e3 = unit(a)
        Cc = a == 1 ? ccx : (a == 2 ? ccy : ccz)
        Cf = a == 1 ? cfx : (a == 2 ? cfy : cfz)
        for σ in (Int32(-1), Int32(1))
            In = I + σ * e1; Jn = J + σ * e2; Kn = K + σ * e3
            if Int32(0) <= In < n1 && Int32(0) <= Jn < n2 && Int32(0) <= Kn < n3
                if covered(t, In, Jn, Kn)
                    # The fine block's cells across the plane, each through its own sub-face.
                    for q in Int32(0):R-Int32(1), p in Int32(0):R-Int32(1)
                        FI = a == 1 ? (σ > 0 ? R * In : R * I - Int32(1)) : R * I + p
                        FJ = a == 2 ? (σ > 0 ? R * Jn : R * J - Int32(1)) : R * J + (a == 1 ? p : q)
                        FK = a == 3 ? (σ > 0 ? R * Kn : R * K - Int32(1)) : R * K + q
                        # The face is on the low side of the cell above the plane.
                        PI = a == 1 ? (σ > 0 ? FI : FI + Int32(1)) : FI
                        PJ = a == 2 ? (σ > 0 ? FJ : FJ + Int32(1)) : FJ
                        PK = a == 3 ? (σ > 0 ? FK : FK + Int32(1)) : FK
                        fidx = faceat(t, v, a, PI, PJ, PK)
                        cidx = cellat(t, v, FI, FJ, FK)
                        if fidx > 0 && cidx > 0
                            c = @inbounds Cf[fidx]
                            diag += c; off += c * @inbounds(xf[cidx])
                        end
                    end
                else
                    c = σ < 0 ? @inbounds(Cc[i, j, k]) : @inbounds(Cc[i + e1, j + e2, k + e3])
                    if c > 0f0
                        diag += c; off += c * @inbounds(xc[In + 1, Jn + 1, Kn + 1])
                    end
                end
            end
        end
    end
    return diag, off
end

"One damped Jacobi sweep on a level's fine cells: `out = x + ω (b - A x) / diag` where active, else 0."
@kernel inbounds=true function jacobifine_kernel!(out, @Const(x), @Const(b), @Const(xc), @Const(cx), @Const(cy), @Const(cz),
                                                  @Const(dD), @Const(active), t::Topology, v::Val{N}) where {N}
    n = @index(Global)
    s, i, j, k, w = levelcell(t, v, n)
    if w[1] >= Int32(0)
        if active[n] != 0x00
            diag, off = finerow(t, v, x, xc, cx, cy, cz, dD, s, i, j, k, w)
            out[n] = diag > 0f0 ? x[n] + OMEGA * (b[n] - (diag * x[n] - off)) / diag : 0f0
        else
            out[n] = 0f0
        end
    end
end
@kernel inbounds=true function jacobicoarse_kernel!(out, @Const(x), @Const(b), @Const(xf), @Const(cfx), @Const(cfy), @Const(cfz),
                                                    @Const(ccx), @Const(ccy), @Const(ccz), @Const(dDc), @Const(active),
                                                    t::Topology, v::Val{N}) where {N}
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    if active[i, j, k] != 0x00 && !covered(t, I, J, K)
        diag, off = coarserow(t, v, xf, cfx, cfy, cfz, x, ccx, ccy, ccz, dDc, i, j, k)
        out[i, j, k] = diag > 0f0 ? x[i, j, k] + OMEGA * (b[i, j, k] - (diag * x[i, j, k] - off)) / diag : 0f0
    else
        out[i, j, k] = 0f0
    end
end
"`r = b - A x` on a level's fine cells."
@kernel inbounds=true function residualfine_kernel!(r, @Const(x), @Const(b), @Const(xc), @Const(cx), @Const(cy), @Const(cz),
                                                    @Const(dD), @Const(active), t::Topology, v::Val{N}) where {N}
    n = @index(Global)
    s, i, j, k, w = levelcell(t, v, n)
    if w[1] >= Int32(0)
        if active[n] != 0x00
            diag, off = finerow(t, v, x, xc, cx, cy, cz, dD, s, i, j, k, w)
            r[n] = b[n] - (diag * x[n] - off)
        else
            r[n] = 0f0
        end
    end
end
@kernel inbounds=true function residualcoarse_kernel!(r, @Const(x), @Const(b), @Const(xf), @Const(cfx), @Const(cfy), @Const(cfz),
                                                      @Const(ccx), @Const(ccy), @Const(ccz), @Const(dDc), @Const(active),
                                                      t::Topology, v::Val{N}) where {N}
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    if active[i, j, k] != 0x00 && !covered(t, I, J, K)
        diag, off = coarserow(t, v, xf, cfx, cfy, cfz, x, ccx, ccy, ccz, dDc, i, j, k)
        r[i, j, k] = b[i, j, k] - (diag * x[i, j, k] - off)
    else
        r[i, j, k] = 0f0
    end
end

"""
A level of the multigrid over the fine blocks: the topology, its fine part's resolution
(`Val(16)` or `Val(8)`), its fine coefficients, Dirichlet diagonal and active cells,
and the coarse grid's (cells outside fine blocks).
"""
struct Composite{V,F,U,C,D,A}
    v::V
    cfx::F; cfy::F; cfz::F
    dDf::F
    activef::U
    ccx::C; ccy::C; ccz::C
    dDc::D
    activec::A
end

level0(s::SparseBasin) = Composite(Val(16), s.cu.fine, s.cv.fine, s.cw.fine, s.dD.fine, s.types.fine,
                                   s.cu.coarse, s.cv.coarse, s.cw.coarse, s.dD.coarse, s.dense[1].active)
level1(s::SparseBasin) = Composite(Val(8), s.level1.cx, s.level1.cy, s.level1.cz, s.level1.dD.fine, s.level1.active,
                                   s.cu.coarse, s.cv.coarse, s.cw.coarse, s.dD.coarse, s.dense[1].active)
levelcells(s::SparseBasin, ::Val{N}) where {N} = s.nslots * N^3

function smooth!(s::SparseBasin, L::Composite, x::Field, b::Field, tmp::Field, sweeps)
    be, t = s.backend, s.topo
    nf = levelcells(s, L.v)
    for _ in 1:sweeps
        for (src, dst) in ((x, tmp), (tmp, x))
            jacobifine_kernel!(be, 256)(dst.fine, src.fine, b.fine, src.coarse, L.cfx, L.cfy, L.cfz, L.dDf, L.activef, t, L.v;
                                        ndrange = nf)
            jacobicoarse_kernel!(be, WG3)(dst.coarse, src.coarse, b.coarse, src.fine, L.cfx, L.cfy, L.cfz, L.ccx, L.ccy, L.ccz,
                                          L.dDc, L.activec, t, L.v; ndrange = size(src.coarse))
        end
    end
end

function residual!(s::SparseBasin, L::Composite, r::Field, x::Field, b::Field)
    be, t = s.backend, s.topo
    residualfine_kernel!(be, 256)(r.fine, x.fine, b.fine, x.coarse, L.cfx, L.cfy, L.cfz, L.dDf, L.activef, t, L.v;
                                  ndrange = levelcells(s, L.v))
    residualcoarse_kernel!(be, WG3)(r.coarse, x.coarse, b.coarse, x.fine, L.cfx, L.cfy, L.cfz, L.ccx, L.ccy, L.ccz, L.dDc,
                                    L.activec, t, L.v; ndrange = size(x.coarse))
end

# Level 0's activity on the fine part is the cell type; on the coarse part the dense
# activity array of the coarse level (`dense[1].active`), kept as "uncovered water".
@kernel inbounds=true function activecoarse_kernel!(active, @Const(types), t::Topology)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    active[i, j, k] = (types[i, j, k] == LIQUID && !covered(t, I, J, K)) ? 0x01 : 0x00
end

# ─── coefficients, level 0 ───────────────────────────────────────────────────

"A fine face's coupling: 1 between fine cells, 0.4 to a coarse one, 0 unless both sides are water."
@kernel inbounds=true function finecoef_kernel!(c, @Const(tf), @Const(tc), t::Topology, a::Int32)
    s, i, j, k, w = fineface(t, a, @index(Global))
    if w[1] >= Int32(0)
        I, J, K = global0(w, i, j, k)
        e1, e2, e3 = unit(a)
        tp, fp = celltype(t, tf, tc, I, J, K)
        tm, fm = celltype(t, tf, tc, I - e1, J - e2, K - e3)
        c[faceindex(s, a, i, j, k)] = (tp == LIQUID && tm == LIQUID) ? (fp && fm ? FINE_FINE : FINE_COARSE) : 0f0
    end
end
"The type of fine cell `(I, J, K)` and whether it is fine: its own in a fine block, else the coarse cell's it lies in (outside the grid: wall)."
@inline function celltype(t::Topology, tf, tc, I, J, K)
    idx = fineat(t, I, J, K)
    idx > 0 && return @inbounds(tf[idx]), true
    n1, n2, n3 = size(tc)
    ci = I >> 2; cj = J >> 2; ck = K >> 2
    (Int32(0) <= ci < n1 && Int32(0) <= cj < n2 && Int32(0) <= ck < n3) || return SOLID, false
    return @inbounds(tc[ci + 1, cj + 1, ck + 1]), false
end
"A coarse face's coupling: 4 between water cells outside fine blocks, 0 otherwise (a fine block's sub-faces carry its couplings)."
@kernel inbounds=true function coarsecoef_kernel!(c, @Const(tc), t::Topology, a::Int32)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    e1, e2, e3 = unit(a)
    n1, n2, n3 = size(tc)
    Im = I - e1; Jm = J - e2; Km = K - e3
    ok = Int32(0) <= Im && Int32(0) <= Jm && Int32(0) <= Km && I < n1 && J < n2 && K < n3
    c[i, j, k] = (ok && !covered(t, I, J, K) && !covered(t, Im, Jm, Km) && tc[i, j, k] == LIQUID &&
                  tc[Im + 1, Jm + 1, Km + 1] == LIQUID) ? COARSE_COARSE : 0f0
end

"The air neighbours' share of a fine water cell's diagonal: c/θ for each, the surface between the centres (ghost fluid)."
@kernel inbounds=true function finedirichlet_kernel!(dD, @Const(tf), @Const(tc), φ, t::Topology)
    n = @index(Global)
    s, i, j, k, w = finecell(t, n)
    if w[1] >= Int32(0)
        d = 0f0
        if tf[n] == LIQUID
            I, J, K = global0(w, i, j, k)
            φc = φ.fine[n]
            for m in 1:6
                a, b, cc = neighbour(I, J, K, m)
                ty, isfine = celltype(t, tf, tc, a, b, cc)
                if ty == AIR
                    d += (isfine ? FINE_FINE : FINE_COARSE) / theta(φc, neighbourφ(t, φ, a, b, cc))
                end
            end
        end
        dD[n] = d
    end
end
"The same for a coarse water cell outside fine blocks: its coarse air neighbours, and the air cells of a fine block next to it."
@kernel inbounds=true function coarsedirichlet_kernel!(dD, @Const(tf), @Const(tc), φ, t::Topology)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    d = 0f0
    if tc[i, j, k] == LIQUID && !covered(t, I, J, K)
        n1, n2, n3 = size(tc)
        φc = φ.coarse[i, j, k]
        for a in Int32(1):Int32(3)
            e1, e2, e3 = unit(a)
            for σ in (Int32(-1), Int32(1))
                In = I + σ * e1; Jn = J + σ * e2; Kn = K + σ * e3
                if Int32(0) <= In < n1 && Int32(0) <= Jn < n2 && Int32(0) <= Kn < n3
                    if covered(t, In, Jn, Kn)
                        for q in Int32(0):Int32(3), p in Int32(0):Int32(3)
                            FI = a == 1 ? (σ > 0 ? RATIO * In : RATIO * I - Int32(1)) : RATIO * I + p
                            FJ = a == 2 ? (σ > 0 ? RATIO * Jn : RATIO * J - Int32(1)) : RATIO * J + (a == 1 ? p : q)
                            FK = a == 3 ? (σ > 0 ? RATIO * Kn : RATIO * K - Int32(1)) : RATIO * K + q
                            idx = fineat(t, FI, FJ, FK)
                            if idx > 0 && tf[idx] == AIR
                                d += FINE_COARSE / theta(φc, φ.fine[idx])
                            end
                        end
                    elseif tc[In + 1, Jn + 1, Kn + 1] == AIR
                        d += COARSE_COARSE / theta(φc, φ.coarse[In + 1, Jn + 1, Kn + 1])
                    end
                end
            end
        end
    end
    dD[i, j, k] = d
end

"The right-hand side: the fine cells' and the uncovered coarse cells' outflow, less the volume they should gain (`γ`, see `project!`)."
@kernel inbounds=true function finerhs_kernel!(b, @Const(tf), u, v, w, t::Topology, γ::Float32)
    n = @index(Global)
    s, i, j, k, wh = finecell(t, n)
    if wh[1] >= Int32(0)
        h = t.g.dx
        if tf[n] == LIQUID
            o = Int32(1)
            div = u.fine[faceindex(s, 1, i + o, j, k)] - u.fine[faceindex(s, 1, i, j, k)] +
                  v.fine[faceindex(s, 2, i, j + o, k)] - v.fine[faceindex(s, 2, i, j, k)] +
                  w.fine[faceindex(s, 3, i, j, k + o)] - w.fine[faceindex(s, 3, i, j, k)]
            b[n] = -h * div + γ * h * h
        else
            b[n] = 0f0
        end
    end
end
@kernel inbounds=true function coarserhs_kernel!(b, @Const(tc), @Const(u), @Const(v), @Const(w), t::Topology, γ::Float32)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    if tc[i, j, k] == LIQUID && !covered(t, I, J, K)
        h = t.g.dx
        div = u[i + 1, j, k] - u[i, j, k] + v[i, j + 1, k] - v[i, j, k] + w[i, j, k + 1] - w[i, j, k]
        b[i, j, k] = -16f0 * h * div + 64f0 * γ * h * h
    else
        b[i, j, k] = 0f0
    end
end

function coefficients!(s::SparseBasin)
    be, t = s.backend, s.topo
    tf, tc = s.types.fine, s.types.coarse
    for (a, c) in ((1, s.cu), (2, s.cv), (3, s.cw))
        finecoef_kernel!(be, 256)(c.fine, tf, tc, t, Int32(a); ndrange = finefaces(s))
        coarsecoef_kernel!(be, WG3)(c.coarse, tc, t, Int32(a); ndrange = size(c.coarse))
    end
    finedirichlet_kernel!(be, 256)(s.dD.fine, tf, tc, s.φ, t; ndrange = finecells(s))
    coarsedirichlet_kernel!(be, WG3)(s.dD.coarse, tf, tc, s.φ, t; ndrange = cells(s))
    activecoarse_kernel!(be, WG3)(s.dense[1].active, tc, t; ndrange = cells(s))
    return s
end

# ─── coefficients, coarser levels (Galerkin by aggregation) ──────────────────

"Level 1's fine faces: each the sum of the four level-0 faces it covers."
@kernel inbounds=true function coef1_kernel!(c1, @Const(c0), t::Topology, a::Int32)
    s, i, j, k, w = levelface(t, Val(8), a, @index(Global))
    if w[1] >= Int32(0)
        acc = 0f0
        for q in Int32(0):Int32(1), p in Int32(0):Int32(1)
            ii = a == 1 ? 2i : 2i + p
            jj = a == 2 ? 2j : 2j + (a == 1 ? p : q)
            kk = a == 3 ? 2k : 2k + q
            acc += c0[faceidx(Val(16), s, a, ii, jj, kk)]
        end
        c1[faceidx(Val(8), s, a, i, j, k)] = acc
    end
end
"Level 1's fine cells: the sum of their eight children's Dirichlet diagonal, active if any child is water."
@kernel inbounds=true function cells1_kernel!(dD1, active1, @Const(dD0), @Const(types0), t::Topology)
    n = @index(Global)
    s, i, j, k, w = levelcell(t, Val(8), n)
    if w[1] >= Int32(0)
        d = 0f0; act = 0x00
        for c in Int32(0):Int32(1), b in Int32(0):Int32(1), a in Int32(0):Int32(1)
            m = cellidx(Val(16), s, 2i + a, 2j + b, 2k + c)
            d += dD0[m]
            types0[m] == LIQUID && (act = 0x01)
        end
        dD1[n] = d; active1[n] = act
    end
end
"""
The coarse grid as level 2, every cell an unknown: outside fine blocks level 0's
couplings and diagonal; a face a fine block holds the sum of the level-1 faces on it,
a cell under a fine block the sum of its eight level-1 cells' diagonal, active if any is.
"""
@kernel inbounds=true function coef2_kernel!(c2, @Const(c0c), @Const(c1), t::Topology, a::Int32)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    e1, e2, e3 = unit(a)
    if covered(t, I, J, K) || covered(t, I - e1, J - e2, K - e3)
        acc = 0f0
        for q in Int32(0):Int32(1), p in Int32(0):Int32(1)
            FI = a == 1 ? 2I : 2I + p
            FJ = a == 2 ? 2J : 2J + (a == 1 ? p : q)
            FK = a == 3 ? 2K : 2K + q
            idx = faceat(t, Val(8), a, FI, FJ, FK)
            idx > 0 && (acc += c1[idx])
        end
        c2[i, j, k] = acc
    else
        c2[i, j, k] = c0c[i, j, k]
    end
end
"""
The coarse grid's cells as level 2: under a fine block the sum of its eight level-1
cells' diagonal, active if any is; elsewhere level 0's diagonal. `active` comes in as
levels 0 and 1's (water outside fine blocks) and is updated in place, each cell by its
own thread.
"""
@kernel inbounds=true function cells2_kernel!(dD2, active, @Const(dD0c), @Const(dD1), @Const(active1), t::Topology)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    if covered(t, I, J, K)
        d = 0f0; act = 0x00
        for c in Int32(0):Int32(1), b in Int32(0):Int32(1), a in Int32(0):Int32(1)
            m = cellat(t, Val(8), 2I + a, 2J + b, 2K + c)
            d += dD1[m]
            active1[m] != 0x00 && (act = 0x01)
        end
        dD2[i, j, k] = d; active[i, j, k] = act
    else
        dD2[i, j, k] = dD0c[i, j, k]
    end
end
"A dense level's faces from the finer one's: each the sum of the four it covers."
@kernel inbounds=true function coefdense_kernel!(cc, @Const(cf), a::Int32)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    acc = 0f0
    for q in Int32(0):Int32(1), p in Int32(0):Int32(1)
        FI = a == 1 ? 2I : 2I + p
        FJ = a == 2 ? 2J : 2J + (a == 1 ? p : q)
        FK = a == 3 ? 2K : 2K + q
        acc += cf[FI + 1, FJ + 1, FK + 1]
    end
    cc[i, j, k] = acc
end
@kernel inbounds=true function cellsdense_kernel!(dDc, activec, @Const(dDf), @Const(activef))
    i, j, k = @index(Global, NTuple)
    d = 0f0; act = 0x00
    for c in 0:1, b in 0:1, a in 0:1
        d += dDf[2i - 1 + a, 2j - 1 + b, 2k - 1 + c]
        activef[2i - 1 + a, 2j - 1 + b, 2k - 1 + c] != 0x00 && (act = 0x01)
    end
    dDc[i, j, k] = d; activec[i, j, k] = act
end

"Every level's coefficients from level 0's (see `coefficients!`)."
function hierarchy!(s::SparseBasin)
    be, t = s.backend, s.topo
    L1, D = s.level1, s.dense
    for (a, c1, c0) in ((1, L1.cx, s.cu.fine), (2, L1.cy, s.cv.fine), (3, L1.cz, s.cw.fine))
        coef1_kernel!(be, 256)(c1, c0, t, Int32(a); ndrange = s.nslots * Int(BFACES1))
    end
    cells1_kernel!(be, 256)(L1.dD.fine, L1.active, s.dD.fine, s.types.fine, t; ndrange = s.nslots * Int(BLOCKS1))
    # dense[1] is level 2: the coarse grid with every cell an unknown. Its activity array
    # doubles as levels 0 and 1's coarse activity until here, so read that first.
    D2 = D[1]
    for (a, c2, c0c, c1) in ((1, D2.cx, s.cu.coarse, L1.cx), (2, D2.cy, s.cv.coarse, L1.cy), (3, D2.cz, s.cw.coarse, L1.cz))
        coef2_kernel!(be, WG3)(c2, c0c, c1, t, Int32(a); ndrange = size(c2))
    end
    cells2_kernel!(be, WG3)(D2.dD, D2.active, s.dD.coarse, L1.dD.fine, L1.active, t; ndrange = size(D2.dD))
    for l in 2:length(D)
        F, C = D[l - 1], D[l]
        for (a, cc, cf) in ((1, C.cx, F.cx), (2, C.cy, F.cy), (3, C.cz, F.cz))
            coefdense_kernel!(be, WG3)(cc, cf, Int32(a); ndrange = size(cc))
        end
        cellsdense_kernel!(be, WG3)(C.dD, C.active, F.dD, F.active; ndrange = size(C.dD))
    end
    return s
end

# ─── the dense levels ────────────────────────────────────────────────────────

"`A x` at a dense level's cell from its faces' couplings: `diag x - off`."
@inline function denserow(x, cx, cy, cz, dD, i, j, k)
    n1, n2, n3 = size(x)
    diag = @inbounds dD[i, j, k]; off = 0f0
    @inbounds begin
        c = cx[i, j, k];     (c > 0f0 && i > 1) && (diag += c; off += c * x[i - 1, j, k])
        c = cx[i + 1, j, k]; (c > 0f0 && i < n1) && (diag += c; off += c * x[i + 1, j, k])
        c = cy[i, j, k];     (c > 0f0 && j > 1) && (diag += c; off += c * x[i, j - 1, k])
        c = cy[i, j + 1, k]; (c > 0f0 && j < n2) && (diag += c; off += c * x[i, j + 1, k])
        c = cz[i, j, k];     (c > 0f0 && k > 1) && (diag += c; off += c * x[i, j, k - 1])
        c = cz[i, j, k + 1]; (c > 0f0 && k < n3) && (diag += c; off += c * x[i, j, k + 1])
    end
    return diag, off
end
@kernel inbounds=true function jacobidense_kernel!(out, @Const(x), @Const(b), @Const(cx), @Const(cy), @Const(cz), @Const(dD),
                                                   @Const(active))
    i, j, k = @index(Global, NTuple)
    if active[i, j, k] != 0x00
        diag, off = denserow(x, cx, cy, cz, dD, i, j, k)
        out[i, j, k] = diag > 0f0 ? x[i, j, k] + OMEGA * (b[i, j, k] - (diag * x[i, j, k] - off)) / diag : 0f0
    else
        out[i, j, k] = 0f0
    end
end
@kernel inbounds=true function residualdense_kernel!(r, @Const(x), @Const(b), @Const(cx), @Const(cy), @Const(cz), @Const(dD),
                                                     @Const(active))
    i, j, k = @index(Global, NTuple)
    if active[i, j, k] != 0x00
        diag, off = denserow(x, cx, cy, cz, dD, i, j, k)
        r[i, j, k] = b[i, j, k] - (diag * x[i, j, k] - off)
    else
        r[i, j, k] = 0f0
    end
end
"A coarser dense level's right-hand side: the sum of the eight finer residuals."
@kernel inbounds=true function restrictdense_kernel!(bc, @Const(rf))
    i, j, k = @index(Global, NTuple)
    acc = 0f0
    for c in 0:1, b in 0:1, a in 0:1
        acc += rf[2i - 1 + a, 2j - 1 + b, 2k - 1 + c]
    end
    bc[i, j, k] = acc
end
"The coarser level's correction added to each active finer cell it covers."
@kernel inbounds=true function prolongdense_kernel!(xf, @Const(xc), @Const(activef), boost::Float32)
    i, j, k = @index(Global, NTuple)
    activef[i, j, k] != 0x00 && (xf[i, j, k] += boost * xc[(i + 1) >> 1, (j + 1) >> 1, (k + 1) >> 1])
end

"""
The coarsest level solved in one workgroup: red-black Gauss-Seidel from zero, `sweeps`
times and as many again in the reverse colour order.
"""
@kernel function coarsest_kernel!(x, @Const(b), @Const(cx), @Const(cy), @Const(cz), @Const(dD), @Const(active), n1::Int32,
                                  n2::Int32, n3::Int32, sweeps::Int32)
    t = @index(Local)
    xs = @localmem Float32 (1024,)
    N = n1 * n2 * n3
    inside = t <= N
    i = (t - 1) % n1 + 1; j = ((t - 1) ÷ n1) % n2 + 1; k = (t - 1) ÷ (n1 * n2) + 1
    act = inside && @inbounds(active[i, j, k]) != 0x00
    bi = inside ? @inbounds(b[i, j, k]) : 0f0
    @inbounds xs[t] = 0f0
    @synchronize
    for s in Int32(1):Int32(2) * sweeps
        for half in Int32(0):Int32(1)
            colour = s <= sweeps ? half : Int32(1) - half
            if act && (i + j + k) % 2 == colour
                diag = @inbounds dD[i, j, k]; acc = bi
                @inbounds begin
                    c = cx[i, j, k];     (c > 0f0 && i > 1) && (diag += c; acc += c * xs[t - 1])
                    c = cx[i + 1, j, k]; (c > 0f0 && i < n1) && (diag += c; acc += c * xs[t + 1])
                    c = cy[i, j, k];     (c > 0f0 && j > 1) && (diag += c; acc += c * xs[t - n1])
                    c = cy[i, j + 1, k]; (c > 0f0 && j < n2) && (diag += c; acc += c * xs[t + n1])
                    c = cz[i, j, k];     (c > 0f0 && k > 1) && (diag += c; acc += c * xs[t - n1 * n2])
                    c = cz[i, j, k + 1]; (c > 0f0 && k < n3) && (diag += c; acc += c * xs[t + n1 * n2])
                end
                @inbounds xs[t] = diag > 0f0 ? acc / diag : 0f0
            end
            @synchronize
        end
    end
    inside && (@inbounds x[i, j, k] = xs[t])
end

# ─── between levels ──────────────────────────────────────────────────────────

"Level 1's fine right-hand side: the sum of the eight level-0 residuals each cell covers."
@kernel inbounds=true function restrict01_kernel!(b1, @Const(r0), t::Topology)
    n = @index(Global)
    s, i, j, k, w = levelcell(t, Val(8), n)
    if w[1] >= Int32(0)
        acc = 0f0
        for c in Int32(0):Int32(1), b in Int32(0):Int32(1), a in Int32(0):Int32(1)
            acc += r0[cellidx(Val(16), s, 2i + a, 2j + b, 2k + c)]
        end
        b1[n] = acc
    end
end
"Level 2's right-hand side: under a fine block the sum of its eight level-1 residuals, elsewhere level 1's own."
@kernel inbounds=true function restrict12_kernel!(b2, @Const(r1f), @Const(r1c), t::Topology)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    if covered(t, I, J, K)
        acc = 0f0
        for c in Int32(0):Int32(1), b in Int32(0):Int32(1), a in Int32(0):Int32(1)
            acc += r1f[cellat(t, Val(8), 2I + a, 2J + b, 2K + c)]
        end
        b2[i, j, k] = acc
    else
        b2[i, j, k] = r1c[i, j, k]
    end
end
"Level 2's correction added to level 1: to each active fine cell from the coarse cell it lies in, to each active coarse cell outside fine blocks its own."
@kernel inbounds=true function prolong21fine_kernel!(x1f, @Const(x2), @Const(active1), t::Topology, boost::Float32)
    n = @index(Global)
    s, i, j, k, w = levelcell(t, Val(8), n)
    if w[1] >= Int32(0) && active1[n] != 0x00
        x1f[n] += boost * x2[((w[1] * Int32(8) + i) >> 1) + 1, ((w[2] * Int32(8) + j) >> 1) + 1, ((w[3] * Int32(8) + k) >> 1) + 1]
    end
end
@kernel inbounds=true function prolongcoarse_kernel!(xc, @Const(xcoarser), @Const(active), t::Topology, boost::Float32)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    (active[i, j, k] != 0x00 && !covered(t, I, J, K)) && (xc[i, j, k] += boost * xcoarser[i, j, k])
end
"Level 1's correction added to level 0's active fine cells."
@kernel inbounds=true function prolong10fine_kernel!(x0f, @Const(x1f), @Const(types0), t::Topology, boost::Float32)
    n = @index(Global)
    s, i, j, k, w = finecell(t, n)
    (w[1] >= Int32(0) && types0[n] == LIQUID) && (x0f[n] += boost * x1f[cellidx(Val(8), s, i >> 1, j >> 1, k >> 1)])
end

@kernel inbounds=true function zerofine_kernel!(A, t::Topology, per::Int32)
    n = @index(Global)
    s = (Int32(n - 1)) ÷ per + Int32(1)
    @inbounds t.where[s][1] >= Int32(0) && (A[n] = 0f0)
end
zero!(s::SparseBasin, f::Field, per = BCELLS) =
    (fill!(f.fine,0f0); fill!(f.coarse,0f0); f)

"""
    vcycle!(s, z, r; sweeps = 2)

`z = M⁻¹ r`: one V-cycle from zero through the levels (fine blocks at 16³ and 8³ over
the coarse grid, then the dense levels), `sweeps` pairs of damped Jacobi sweeps down
and up, the coarsest solved in one workgroup.
"""
function vcycle!(s::SparseBasin, z::Field, r::Field; sweeps = 1, coarse = 40, boost = 1f0)
    be, t = s.backend, s.topo
    L0, L1 = level0(s), level1(s)
    X1, D = s.level1, s.dense
    zero!(s, z)
    smooth!(s, L0, z, r, s.tmp, sweeps)
    residual!(s, L0, s.q, z, r)
    restrict01_kernel!(be, 256)(X1.b.fine, s.q.fine, t; ndrange = levelcells(s, Val(8)))
    copyto!(X1.b.coarse, s.q.coarse)
    zero!(s, X1.x, BLOCKS1)
    smooth!(s, L1, X1.x, X1.b, X1.t, sweeps)
    residual!(s, L1, X1.r, X1.x, X1.b)
    restrict12_kernel!(be, WG3)(D[1].b, X1.r.fine, X1.r.coarse, t; ndrange = size(D[1].b))
    for l in 1:length(D)-1
        Dl = D[l]
        fill!(Dl.x, 0f0)
        for _ in 1:sweeps, (src, dst) in ((Dl.x, Dl.t), (Dl.t, Dl.x))
            jacobidense_kernel!(be, WG3)(dst, src, Dl.b, Dl.cx, Dl.cy, Dl.cz, Dl.dD, Dl.active; ndrange = size(Dl.x))
        end
        residualdense_kernel!(be, WG3)(Dl.r, Dl.x, Dl.b, Dl.cx, Dl.cy, Dl.cz, Dl.dD, Dl.active; ndrange = size(Dl.x))
        restrictdense_kernel!(be, WG3)(D[l + 1].b, Dl.r; ndrange = size(D[l + 1].b))
    end
    Dm = D[end]
    m1, m2, m3 = size(Dm.x)
    coarsest_kernel!(be, 1024)(Dm.x, Dm.b, Dm.cx, Dm.cy, Dm.cz, Dm.dD, Dm.active, Int32(m1), Int32(m2), Int32(m3),
                               Int32(coarse); ndrange = 1024)
    for l in length(D)-1:-1:1
        Dl = D[l]
        prolongdense_kernel!(be, WG3)(Dl.x, D[l + 1].x, Dl.active, Float32(boost); ndrange = size(Dl.x))
        for _ in 1:sweeps, (src, dst) in ((Dl.x, Dl.t), (Dl.t, Dl.x))
            jacobidense_kernel!(be, WG3)(dst, src, Dl.b, Dl.cx, Dl.cy, Dl.cz, Dl.dD, Dl.active; ndrange = size(Dl.x))
        end
    end
    prolong21fine_kernel!(be, 256)(X1.x.fine, D[1].x, X1.active, t, Float32(boost); ndrange = levelcells(s, Val(8)))
    prolongcoarse_kernel!(be, WG3)(X1.x.coarse, D[1].x, D[1].active, t, Float32(boost); ndrange = size(D[1].x))
    smooth!(s, L1, X1.x, X1.b, X1.t, sweeps)
    prolong10fine_kernel!(be, 256)(z.fine, X1.x.fine, s.types.fine, t, Float32(boost); ndrange = finecells(s))
    prolongcoarse_kernel!(be, WG3)(z.coarse, X1.x.coarse, D[1].active, t, Float32(boost); ndrange = size(z.coarse))
    smooth!(s, L0, z, r, s.tmp, sweeps)
    return z
end

# ─── vectors ─────────────────────────────────────────────────────────────────

@kernel function dotn_kernel!(partials, @Const(a), @Const(b), n::Int32)
    i = @index(Global); li = @index(Local); gi = @index(Group)
    sm = @localmem Float32 (256,)
    v = i <= n ? @inbounds(a[i] * b[i]) : 0f0
    groupsum!(partials, sm, v, li, gi)
end
@kernel function dotfinefield_kernel!(partials,@Const(a),@Const(b),@Const(types),t::Topology,n::Int32)
    i=@index(Global);li=@index(Local);gi=@index(Group)
    sm=@localmem Float32 (256,)
    v=0f0
    if i<=n && types[i]==LIQUID
        slot=(Int32(i)-Int32(1))÷BCELLS+Int32(1)
        t.where[slot][1]>=Int32(0) && (v=a[i]*b[i])
    end
    groupsum!(partials,sm,v,li,gi)
end
@kernel function dotcoarsefield_kernel!(partials,@Const(a),@Const(b),@Const(types),t::Topology,n::Int32)
    i=@index(Global);li=@index(Local);gi=@index(Group)
    sm=@localmem Float32 (256,)
    v=0f0
    if i<=n && types[i]==LIQUID
        nx,ny,_=size(types)
        q=i-1;I=q%nx;J=(q÷nx)%ny;K=q÷(nx*ny)
        !covered(t,I,J,K) && (v=a[i]*b[i])
    end
    groupsum!(partials,sm,v,li,gi)
end
"The dot product over actual pressure unknowns; free slots and covered coarse cells are excluded."
function inner(s::SparseBasin, a::Field, b::Field)
    fill!(s.partials,0f0)
    nf=finecells(s)
    dotfinefield_kernel!(s.backend,256)(s.partials,a.fine,b.fine,s.types.fine,s.topo,Int32(nf);
        ndrange=256cld(nf,256))
    acc=sum(s.partials)
    fill!(s.partials,0f0)
    nc=length(a.coarse)
    dotcoarsefield_kernel!(s.backend,256)(s.partials,a.coarse,b.coarse,s.types.coarse,s.topo,Int32(nc);
        ndrange=256cld(nc,256))
    return acc+sum(s.partials)
end

@kernel inbounds=true function axpyfield_kernel!(y, @Const(x), α::Float32)
    i = @index(Global)
    y[i] += α * x[i]
end
@kernel inbounds=true function xpbyfield_kernel!(y, @Const(x), β::Float32)
    i = @index(Global)
    y[i] = x[i] + β * y[i]
end
axpy!(s::SparseBasin, y::Field, α, x::Field) =
    (axpyfield_kernel!(s.backend, 256)(y.fine, x.fine, Float32(α); ndrange = finecells(s));
     axpyfield_kernel!(s.backend, 256)(y.coarse, x.coarse, Float32(α); ndrange = length(y.coarse)); y)
"`y = x + β y`."
xpby!(s::SparseBasin, y::Field, x::Field, β) =
    (xpbyfield_kernel!(s.backend, 256)(y.fine, x.fine, Float32(β); ndrange = finecells(s));
     xpbyfield_kernel!(s.backend, 256)(y.coarse, x.coarse, Float32(β); ndrange = length(y.coarse)); y)
copyfield!(s::SparseBasin, y::Field, x::Field) = (copyto!(y.fine, x.fine); copyto!(y.coarse, x.coarse); y)

@kernel inbounds=true function applyfine_kernel!(q, @Const(x), @Const(xc), @Const(cx), @Const(cy), @Const(cz), @Const(dD),
                                                 @Const(types), t::Topology)
    n = @index(Global)
    s, i, j, k, w = finecell(t, n)
    if w[1] >= Int32(0)
        if types[n] == LIQUID
            diag, off = finerow(t, Val(16), x, xc, cx, cy, cz, dD, s, i, j, k, w)
            q[n] = diag * x[n] - off
        else
            q[n] = 0f0
        end
    end
end
@kernel inbounds=true function applycoarse_kernel!(q, @Const(x), @Const(xf), @Const(cfx), @Const(cfy), @Const(cfz), @Const(ccx),
                                                   @Const(ccy), @Const(ccz), @Const(dDc), @Const(active), t::Topology)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    if active[i, j, k] != 0x00 && !covered(t, I, J, K)
        diag, off = coarserow(t, Val(16), xf, cfx, cfy, cfz, x, ccx, ccy, ccz, dDc, i, j, k)
        q[i, j, k] = diag * x[i, j, k] - off
    else
        q[i, j, k] = 0f0
    end
end
"`q = A x` at level 0."
function apply!(s::SparseBasin, q::Field, x::Field)
    be, t = s.backend, s.topo
    applyfine_kernel!(be, 256)(q.fine, x.fine, x.coarse, s.cu.fine, s.cv.fine, s.cw.fine, s.dD.fine, s.types.fine, t;
                               ndrange = finecells(s))
    applycoarse_kernel!(be, WG3)(q.coarse, x.coarse, x.fine, s.cu.fine, s.cv.fine, s.cw.fine, s.cu.coarse, s.cv.coarse,
                                 s.cw.coarse, s.dD.coarse, s.dense[1].active, t; ndrange = size(x.coarse))
    return q
end

"""
    pressure!(s; tol = 1e-4, maxiter = 200, sweeps = 1, boost = 1) -> iterations

The pressure by flexible conjugate gradients (Polak-Ribière, which stays robust with a
preconditioner that is not exactly symmetric) preconditioned by `vcycle!`, from the
last step's pressure, to `tol` of the right-hand side. `boost > 1` scales coarse
corrections, but can make the preconditioner indefinite on refined 5 cm grids.
The unscaled V-cycle is the conservative default.
"""
function pressure!(s::SparseBasin; tol = 1f-4, maxiter = 200, sweeps = 1, boost = 1f0, warn_on_nonconvergence = true)
    hierarchy!(s)
    # r = b - A P
    apply!(s, s.r, s.P)
    xpby!(s, s.r, s.b, -1f0)
    bnorm = sqrt(inner(s, s.b, s.b))
    isfinite(bnorm) || error("Nonfinite pressure right-hand side at t=$(s.t)")
    bnorm == 0f0 && return 0
    target = tol * bnorm
    rnorm = sqrt(inner(s, s.r, s.r))
    rnorm <= target && return 0
    vcycle!(s, s.z, s.r; sweeps, boost)
    copyfield!(s, s.d, s.z)
    ρ = inner(s, s.z, s.r)
    isfinite(ρ) || error("Nonfinite pressure preconditioner at t=$(s.t)")
    ρ>0f0 || error("Pressure CG requires a positive preconditioner at t=$(s.t)")
    for it in 1:maxiter
        apply!(s, s.q, s.d)
        denominator=inner(s,s.d,s.q)
        denominator>0f0 || error("Pressure CG requires a positive operator at t=$(s.t), iteration $it")
        α = ρ / denominator
        isfinite(α) || error("Nonfinite pressure update at t=$(s.t), iteration $it")
        axpy!(s, s.P, α, s.d)
        copyfield!(s, s.rold, s.r)
        axpy!(s, s.r, -α, s.q)
        rnorm = sqrt(inner(s, s.r, s.r))
        rnorm <= target && return it
        vcycle!(s, s.z, s.r; sweeps, boost)
        ρnew = inner(s, s.z, s.r)
        ρnew>0f0 || error("Pressure CG requires a positive preconditioner at t=$(s.t), iteration $it")
        β = max(0f0,(ρnew - inner(s, s.z, s.rold)) / ρ)
        isfinite(β) || error("Nonfinite pressure direction at t=$(s.t), iteration $it")
        ρ = ρnew
        xpby!(s, s.d, s.z, β)
    end
    warn_on_nonconvergence && @warn "pressure solve did not converge" residual = rnorm / bnorm maxiter
    return maxiter
end

# ─── projection ──────────────────────────────────────────────────────────────

"The pressure of fine cell `(I, J, K)`: in a fine block its own, else the coarse cell's."
@inline function pressureat(t::Topology, P, I, J, K)
    idx = fineat(t, I, J, K)
    n1, n2, n3 = size(P.coarse)
    return ifelse(idx > 0, @inbounds(P.fine[max(idx,Int32(1))]),
        @inbounds(P.coarse[clamp((I >> 2) + 1, 1, n1), clamp((J >> 2) + 1, 1, n2), clamp((K >> 2) + 1, 1, n3)]))
end

"u ← u - ∇P on the fine faces along `a` (centres `h` apart, 2.5h across to a coarse cell; ghost fluid at the surface), flagging those with water on a side and no wall."
@kernel inbounds=true function projectfine_kernel!(u, valid, P, @Const(tf), @Const(tc), φ, t::Topology, a::Int32)
    s, i, j, k, w = fineface(t, a, @index(Global))
    if w[1] >= Int32(0)
        idx = faceindex(s, a, i, j, k)
        I, J, K = global0(w, i, j, k)
        e1, e2, e3 = unit(a)
        tp, fp = celltype(t, tf, tc, I, J, K)
        tm, fm = celltype(t, tf, tc, I - e1, J - e2, K - e3)
        ok = tp != SOLID && tm != SOLID && (tp == LIQUID || tm == LIQUID)
        if ok
            d = (fp && fm ? 1f0 : 2.5f0) * t.g.dx
            Pp = pressureat(t, P, I, J, K); Pm = pressureat(t, P, I - e1, J - e2, K - e3)
            Δ = if tp == LIQUID && tm == LIQUID
                Pp - Pm
            elseif tm == LIQUID
                -Pm / theta(neighbourφ(t, φ, I - e1, J - e2, K - e3), neighbourφ(t, φ, I, J, K))
            else
                Pp / theta(neighbourφ(t, φ, I, J, K), neighbourφ(t, φ, I - e1, J - e2, K - e3))
            end
            u[idx] -= Δ / d
        end
        valid[idx] = ok ? 0x01 : 0x00
    end
end
"The same on the coarse faces between cells outside fine blocks (the others are the fine faces' means, see `restrict!`)."
@kernel inbounds=true function projectcoarse_kernel!(u, valid, @Const(P), @Const(tc), @Const(φ), t::Topology, a::Int32)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    e1, e2, e3 = unit(a)
    n1, n2, n3 = size(tc)
    Im = I - e1; Jm = J - e2; Km = K - e3
    inside = Int32(0) <= Im && Int32(0) <= Jm && Int32(0) <= Km && I < n1 && J < n2 && K < n3
    if inside && !covered(t, I, J, K) && !covered(t, Im, Jm, Km)
        tp = tc[i, j, k]; tm = tc[Im + 1, Jm + 1, Km + 1]
        ok = tp != SOLID && tm != SOLID && (tp == LIQUID || tm == LIQUID)
        if ok
            Pp = P[i, j, k]; Pm = P[Im + 1, Jm + 1, Km + 1]
            φp = φ[i, j, k]; φm = φ[Im + 1, Jm + 1, Km + 1]
            Δ = (tp == LIQUID && tm == LIQUID) ? Pp - Pm : (tm == LIQUID ? -Pm / theta(φm, φp) : Pp / theta(φp, φm))
            u[i, j, k] -= Δ / coarsedx(t.g)
        end
        valid[i, j, k] = ok ? 0x01 : 0x00
    elseif !inside
        valid[i, j, k] = 0x00
    end
end

@kernel inbounds=true function clearfine_kernel!(P, @Const(types), t::Topology)
    n = @index(Global)
    s, i, j, k, w = finecell(t, n)
    (w[1] >= Int32(0) && types[n] != LIQUID) && (P[n] = 0f0)
end
@kernel inbounds=true function clearcoarse_kernel!(P, @Const(types), t::Topology)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    (types[i, j, k] != LIQUID || covered(t, I, J, K)) && (P[i, j, k] = 0f0)
end

"The water's volume: each cell by how much of it is under the surface; coarse cells outside fine blocks only."
@kernel inbounds=true function finevolume_kernel!(out, @Const(φ), t::Topology)
    n = @index(Global)
    s, i, j, k, w = finecell(t, n)
    if w[1] >= Int32(0)
        out[n] = issolid(t.g, finecentre(t.g, global0(w, i, j, k)...)) ? 0f0 : clamp(0.5f0 - φ[n] / t.g.dx, 0f0, 1f0)
    end
end
@kernel inbounds=true function coarsevolume_kernel!(out, @Const(φ), t::Topology)
    i, j, k = @index(Global, NTuple)
    I, J, K = coarse0(i, j, k)
    out[i, j, k] = (covered(t, I, J, K) || issolid(t.g, coarsecentre(t.g, I, J, K))) ? 0f0 :
                   64f0 * clamp(0.5f0 - φ[i, j, k] / coarsedx(t.g), 0f0, 1f0)
end
"The water's volume, in cubic metres."
function volume(s::SparseBasin)
    be, t = s.backend, s.topo
    finevolume_kernel!(be, 256)(s.tmp.fine, s.φ.fine, t; ndrange = finecells(s))
    coarsevolume_kernel!(be, WG3)(s.tmp.coarse, s.φ.coarse, t; ndrange = cells(s))
    # Free and unused slots hold zeros: the whole pool sums to the fine blocks' volume.
    return (sum(s.tmp.fine) + sum(s.tmp.coarse)) * s.grid.dx^3
end

function project!(s::SparseBasin, dt; recover = 0.5)
    be, t = s.backend, s.topo
    loss = (s.volume0 - volume(s)) / s.volume0
    γ = Float32(clamp(loss / recover, -0.01 / dt, 0.01 / dt))
    coefficients!(s)
    finerhs_kernel!(be, 256)(s.b.fine, s.types.fine, s.u, s.v, s.w, t, γ; ndrange = finecells(s))
    coarserhs_kernel!(be, WG3)(s.b.coarse, s.types.coarse, s.u.coarse, s.v.coarse, s.w.coarse, t, γ; ndrange = cells(s))
    clearfine_kernel!(be, 256)(s.P.fine, s.types.fine, t; ndrange = finecells(s))
    clearcoarse_kernel!(be, WG3)(s.P.coarse, s.types.coarse, t; ndrange = cells(s))
    push!(s.iterations, pressure!(s))
    for (a, f, fl) in ((1, s.u, s.fu), (2, s.v, s.fv), (3, s.w, s.fw))
        projectfine_kernel!(be, 256)(f.fine, fl.fine, s.P, s.types.fine, s.types.coarse, s.φ, t, Int32(a); ndrange = finefaces(s))
        projectcoarse_kernel!(be, WG3)(f.coarse, fl.coarse, s.P.coarse, s.types.coarse, s.φ.coarse, t, Int32(a);
                                       ndrange = size(f.coarse))
        restrict!(s, f, a); restrict!(s, fl, a)
    end
    return s
end

# ─── extrapolation ───────────────────────────────────────────────────────────

"One layer of extrapolation on the fine faces along `a`: an unflagged face takes the mean of its flagged neighbours along each axis."
@kernel inbounds=true function extrapfine_kernel!(out, fout, @Const(src), @Const(fin), t::Topology, a::Int32)
    s, i, j, k, w = fineface(t, a, @index(Global))
    if w[1] >= Int32(0)
        idx = faceindex(s, a, i, j, k)
        if fin[idx] != 0x00
            out[idx] = src[idx]; fout[idx] = 0x01
        else
            I, J, K = global0(w, i, j, k)
            acc = 0f0; m = Int32(0)
            if !walledface(t.g, facepos(t.g, a, I, J, K, t.g.dx), a, t.g.dx)
                for q in 1:6
                    x, y, z = neighbour(I, J, K, q)
                    nidx = finefaceat(t, a, Int32(x), Int32(y), Int32(z))
                    if nidx > 0 && fin[nidx] != 0x00
                        acc += src[nidx]; m += Int32(1)
                    end
                end
            end
            out[idx] = m > Int32(0) ? acc / m : src[idx]
            fout[idx] = m > Int32(0) ? 0x01 : 0x00
        end
    end
end
"The same on the coarse faces outside fine blocks."
@kernel inbounds=true function extrapcoarse_kernel!(out, fout, @Const(src), @Const(fin), t::Topology, a::Int32)
    i, j, k = @index(Global, NTuple)
    if fin[i, j, k] != 0x00
        out[i, j, k] = src[i, j, k]; fout[i, j, k] = 0x01
    else
        h = coarsedx(t.g)
        acc = 0f0; m = 0
        if !walledface(t.g, facepos(t.g, a, coarse0(i, j, k)..., h), a, h)
            for q in 1:6
                x, y, z = neighbour(i, j, k, q)
                if inrange(src, x, y, z) && fin[x, y, z] != 0x00
                    acc += src[x, y, z]; m += 1
                end
            end
        end
        out[i, j, k] = m > 0 ? acc / m : src[i, j, k]
        fout[i, j, k] = m > 0 ? 0x01 : 0x00
    end
end
@kernel inbounds=true function unreachedfine_kernel!(u, @Const(reached), t::Topology, a::Int32)
    s, i, j, k, w = fineface(t, a, @index(Global))
    if w[1] >= Int32(0)
        idx = faceindex(s, a, i, j, k)
        reached[idx] == 0x00 && (u[idx] = 0f0)
    end
end

"""
`layers` layers of extrapolation of each of the arrays `fields` (each with a scratch
array of its size in `scratch`) by the same flags `fl` (scratch `fl2`), through
`layer!(dst, fout, src, fin)`: the results end up in the arrays, the flags of what was
reached in `fl`.
"""
function extrapolated!(layer!, fields, scratch, fl, fl2, layers)
    srcs, dsts = fields, scratch
    fin, fout = fl, fl2
    for _ in 1:layers
        foreach((src, dst) -> layer!(dst, fout, src, fin), srcs, dsts)
        srcs, dsts = dsts, srcs; fin, fout = fout, fin
    end
    foreach((f, src) -> src === f || copyto!(f, src), fields, srcs)
    fin === fl || copyto!(fl, fin)
    return fl
end

"""
    extrapolate!(s::SparseBasin; layers = 6, coarselayers = 2, kick = false)

The velocity out from the faces the projection flagged, `layers` faces on the fine
grid and `coarselayers` on the coarse one; the faces it does not reach zeroed. With
`kick`, the projection's kick in `ur`, `vr`, `wr` (see `kick!`) along with it, by the
same flags.
"""
function extrapolate!(s::SparseBasin; layers = 6, coarselayers = 2, kick = false)
    be, t = s.backend, s.topo
    for (a, f, tmp, k, ktmp, fl, fl2) in ((1, s.u, s.u0, s.ur, s.ur0, s.fu, s.fu2), (2, s.v, s.v0, s.vr, s.vr0, s.fv, s.fv2),
                                          (3, s.w, s.w0, s.wr, s.wr0, s.fw, s.fw2))
        fields, scratch = kick ? ((f, k), (tmp, ktmp)) : ((f,), (tmp,))
        extrapolated!((dst, fout, src, fin) -> extrapfine_kernel!(be, 256)(dst, fout, src, fin, t, Int32(a); ndrange = finefaces(s)),
                      map(x -> x.fine, fields), map(x -> x.fine, scratch), fl.fine, fl2.fine, layers)
        foreach(x -> restrict!(s, x, a), fields)
        restrict!(s, fl, a)
        extrapolated!((dst, fout, src, fin) -> extrapcoarse_kernel!(be, WG3)(dst, fout, src, fin, t, Int32(a); ndrange = size(src)),
                      map(x -> x.coarse, fields), map(x -> x.coarse, scratch), fl.coarse, fl2.coarse, coarselayers)
        for x in fields
            unreachedfine_kernel!(be, 256)(x.fine, fl.fine, t, Int32(a); ndrange = finefaces(s))
            unreached_kernel!(be, 256)(x.coarse, fl.coarse; ndrange = length(x.coarse))
        end
    end
    solidify!(s)
    return restrictvelocity!(s)
end

"The projection's kick where it acted (the faces flagged `valid`), zero elsewhere: `k` holds the velocity before it, `u` after."
@kernel inbounds=true function kick_kernel!(k, @Const(u), @Const(valid))
    i = @index(Global)
    k[i] = valid[i] != 0x00 ? u[i] - k[i] : 0f0
end

"In `ur`, `vr`, `wr`, which held the velocity before the projection: the projection's kick (see `kick_kernel!`)."
function kick!(s::SparseBasin)
    for (k, f, fl) in ((s.ur, s.u, s.fu), (s.vr, s.v, s.fv), (s.wr, s.w, s.fw))
        kick_kernel!(s.backend, 256)(k.fine, f.fine, fl.fine; ndrange = finefaces(s))
        kick_kernel!(s.backend, 256)(k.coarse, f.coarse, fl.coarse; ndrange = length(k.coarse))
    end
    return s
end

"The reflected velocity in `ur`, `vr`, `wr`: the projected one kicked once more (their extrapolated kick added)."
function reflected!(s::SparseBasin)
    for (k, f) in ((s.ur, s.u), (s.vr, s.v), (s.wr, s.w))
        k.fine .+= f.fine
        k.coarse .+= f.coarse
    end
    return s
end

grid2particles!(s::SparseBasin, dt; flow = Flow(s)) =
    (s.particles.n > 0 && g2p_kernel!(s.backend, 256)(s.particles.vx, s.particles.vy, s.particles.vz, s.particles.cx,
                                                      s.particles.cy, s.particles.cz, s.particles.x, flow, Float32(dt),
                                                      s.basin.g, Int32(s.particles.n); ndrange = s.particles.n); s)

# ─── resampling ──────────────────────────────────────────────────────────────

"Each particle's fine cell (pool index), if it stays (in a fine block, out of the walls, in the band), and its rank in the cell."
@kernel inbounds=true function sparsebin_kernel!(key, rank, counts, @Const(x), φ, t::Topology, deep::Float32, above::Float32,
                                                 n::Int32)
    p = @index(Global)
    if p <= n
        xp = x[p]
        c = Vec3f((xp - t.g.lo) / t.g.dx)
        idx = Int32(0)
        if isfinite(c[1]) && isfinite(c[2]) && isfinite(c[3]) && !issolid(t.g, xp)
            I = unsafe_trunc(Int32, floor(c[1])); J = unsafe_trunc(Int32, floor(c[2])); K = unsafe_trunc(Int32, floor(c[3]))
            idx = fineat(t, I, J, K)
            if idx > Int32(0)
                f = cellsample(t, φ, xp)
                (-deep <= f < above) || (idx = Int32(0))
            end
        end
        if idx > Int32(0)
            key[p] = idx
            rank[p] = (KA.@atomic :monotonic counts[idx] += Int32(1)) - Int32(1)
        else
            key[p] = Int32(0); rank[p] = Int32(-1)
        end
    end
end

@kernel inbounds=true function sparsetotals_kernel!(totals, @Const(counts), @Const(φ), t::Topology, nmin::Int32, deep::Float32,
                                                    seeding::Bool)
    n = @index(Global)
    s, i, j, k, w = finecell(t, n)
    if w[1] >= Int32(0)
        kept, add = cellcounts(counts[n], φ[n], issolid(t.g, finecentre(t.g, global0(w, i, j, k)...)), nmin, deep, t.g.dx,
                               seeding)
        totals[n] = kept + add
    end
end

"New particles in the fine cells that need them (see `spawn_kernel!`): stratified, jittered, the grid's velocity and its affine part."
@kernel inbounds=true function sparsespawn_kernel!(x2, vx2, vy2, vz2, cx2, cy2, cz2, lag2, @Const(counts), @Const(totals),
                                                   @Const(offsets), φ, flow, nmin::Int32, deep::Float32, seeding::Bool,
                                                   seed::UInt32)
    t = flow.g
    n = @index(Global)
    s, i, j, k, w = finecell(t, n)
    if w[1] >= Int32(0)
        g = t.g
        I, J, K = global0(w, i, j, k)
        kept, add = cellcounts(counts[n], φ.fine[n], issolid(g, finecentre(g, I, J, K)), nmin, deep, g.dx, seeding)
        if add > Int32(0)
            base = offsets[n] - totals[n] + kept
            corner = g.lo + Vec3f(I, J, K) * g.dx
            for m in Int32(0):add-Int32(1)
                h = (n % Int32) * Int32(31) + m
                jx = hash01(h, (seed % Int32) + Int32(1)); jy = hash01(h, (seed % Int32) + Int32(2))
                jz = hash01(h, (seed % Int32) + Int32(3))
                o = m % Int32(8)
                sub = Vec3f(o & Int32(1), (o >> 1) & Int32(1), (o >> 2) & Int32(1))
                xp = Point3f(corner + (sub + Vec3f(jx, jy, jz)) * (0.5f0 * g.dx))
                dst = base + m + 1
                lag2[dst] = 0f0
                if seeding && cellsample(t, φ, xp) >= 0f0
                    x2[dst] = Point3f(NaN32, NaN32, NaN32); vx2[dst] = 0f0; vy2[dst] = 0f0; vz2[dst] = 0f0
                    cx2[dst] = Vec3f(0); cy2[dst] = Vec3f(0); cz2[dst] = Vec3f(0)
                else
                    a1, c1, _ = gathered(flow, 1, xp); a2, c2, _ = gathered(flow, 2, xp); a3, c3, _ = gathered(flow, 3, xp)
                    x2[dst] = xp; vx2[dst] = a1; vy2[dst] = a2; vz2[dst] = a3
                    cx2[dst] = c1; cy2[dst] = c2; cz2[dst] = c3
                end
            end
        end
    end
end

"""
    resample!(s::SparseBasin; seeding = false, flow = Flow(s))

The narrow band kept `s.band` cells deep with `nmin`–`2nmin` particles a fine cell, as
on the dense grid (see `resample!`), written out in the order of the fine pool; new
particles take `flow`'s velocity.
"""
function resample!(s::SparseBasin; seeding = false, flow = Flow(s))
    ps, g, be, t = s.particles, s.grid, s.backend, s.topo
    n = ps.n
    deep = Float32(s.band * g.dx)
    fill!(s.counts, Int32(0)); fill!(s.totals, Int32(0))
    n > 0 && sparsebin_kernel!(be, 256)(s.key, s.rank, s.counts, ps.x, s.φ, t, deep, 3f0 * g.dx, Int32(n); ndrange = n)
    sparsetotals_kernel!(be, 256)(s.totals, s.counts, s.φ.fine, t, s.nmin, deep, seeding; ndrange = finecells(s))
    used = finecells(s)
    AK.accumulate!(+, view(s.offsets, 1:used), view(s.totals, 1:used); init = Int32(0))
    total = Int(Array(view(s.offsets, used:used))[1])
    total <= capacity(ps) || error("the narrow band needs $total particles, more than the $(capacity(ps)) allocated")
    n > 0 && scatter_kernel!(be, 256)(ps.x2, ps.vx2, ps.vy2, ps.vz2, ps.cx2, ps.cy2, ps.cz2, ps.lag2, ps.x, ps.vx, ps.vy,
                                      ps.vz, ps.cx, ps.cy, ps.cz, ps.lag, s.key, s.rank, s.totals, s.offsets,
                                      Int32(2s.nmin), Int32(n); ndrange = n)
    sparsespawn_kernel!(be, 256)(ps.x2, ps.vx2, ps.vy2, ps.vz2, ps.cx2, ps.cy2, ps.cz2, ps.lag2, s.counts, s.totals,
                                 s.offsets, s.φ, flow, s.nmin, deep, seeding, UInt32(s.steps + 1); ndrange = finecells(s))
    swap!(ps)
    ps.n = total
    return s
end

# ─── a step ──────────────────────────────────────────────────────────────────

@kernel function maxabsn_kernel!(partials, @Const(a), n::Int32)
    i = @index(Global); li = @index(Local); gi = @index(Group)
    sm = @localmem Float32 (256,)
    @inbounds sm[li] = i <= n ? abs(a[i]) : 0f0
    @synchronize
    step = 128
    while step >= 1
        li <= step && (@inbounds sm[li] = max(sm[li], sm[li + step]))
        @synchronize
        step ÷= 2
    end
    li == 1 && (@inbounds partials[gi] = sm[1])
end
"The fastest the water moves, from the fine and the coarse faces."
function maxspeed(s::SparseBasin)
    m = 0f0
    for f in (s.u, s.v, s.w), (A, n) in ((f.fine, finefaces(s)), (f.coarse, length(f.coarse)))
        fill!(s.partials, 0f0)
        maxabsn_kernel!(s.backend, 256)(s.partials, A, Int32(n); ndrange = 256 * cld(n, 256))
        m = max(m, maximum(s.partials))
    end
    return m
end

"""
    step!(s::SparseBasin, dt)

One step, as on the dense grid (see `step!`), the fine blocks first made to follow the
surface and the particles (see `refine!`).

With `s.reflect`, Zehnder, Narain and Thomaszewski's advection-reflection (2018): two
halves, the first ending in the projection's kick applied twice (the velocity reflected
about the divergence-free ones, which keeps its energy), the second carrying that
along the projected velocity and projecting as usual. Advect-then-project turns a
rotating flow by pushing it out and projecting the push away, which costs the swirl
`1 - cos(ω dt)` a step. In the throat (ω of 3–5 per second) one step a frame lost half
its angular momentum; the reflection's loss is of higher order in `dt`.
"""
function step!(s::SparseBasin, dt)
    s.reflect || return substep!(s, dt)
    substep!(s, dt / 2; reflect = true)
    return substep!(s, dt / 2; carried = (s.ur, s.vr, s.wr))
end

"One step `dt` carrying the `carried` velocity, and with `reflect` leaving the reflected velocity in `ur`, `vr`, `wr` for the particles and the next step."
function substep!(s::SparseBasin, dt; reflect = false, carried = (s.u, s.v, s.w))
    refine!(s)
    advect!(s, dt; carried)
    particles2grid!(s, dt)
    redistance!(s)
    classify!(s)
    filltrapped!(s)
    forces!(s, dt)
    if reflect
        for (k, f) in ((s.ur, s.u), (s.vr, s.v), (s.wr, s.w))
            copyto!(k.fine, f.fine); copyto!(k.coarse, f.coarse)
        end
    end
    project!(s, dt)
    reflect && kick!(s)
    extrapolate!(s; layers = 6, kick = reflect)
    reflect && reflected!(s)
    flow = reflect ? Flow(s.topo, s.ur, s.vr, s.wr, s.fu, s.fv, s.fw) : Flow(s)
    grid2particles!(s, dt; flow)
    s.steps += 1
    resample!(s; flow)
    s.t += dt
    return s
end

# ─── whitewater ──────────────────────────────────────────────────────────────

"""
The narrow band's particles as Ihmsen et al.'s emission (`emit_kernel!`) reads them,
through the fine cells: resampling wrote the particles out in the fine pool's order, so
a fine cell's particles are a run of them (see the dense grid's whitewater).
"""
struct SparseFluid{P,V,I,T}
    x::P
    v::V
    cellstart::I
    t::T
    h::Float32
end
Adapt.@adapt_structure SparseFluid

@inline function fluidcell(g::SparseFluid, x)
    c = Vec3f((x - g.t.g.lo) / g.h)
    return Vec{3,Int32}(unsafe_trunc(Int32, floor(c[1])), unsafe_trunc(Int32, floor(c[2])), unsafe_trunc(Int32, floor(c[3])))
end
@inline function cellrange(g::SparseFluid, n)
    idx = fineat(g.t, n[1], n[2], n[3])
    idx == Int32(0) && return Int32(1):Int32(0)
    return @inbounds(g.cellstart[idx]):(@inbounds(g.cellstart[idx + Int32(1)]) - Int32(1))
end

"Each particle's surface normal, the level set's gradient, within `near` of the surface (see `levelnormals_kernel!`)."
@kernel inbounds=true function sparsenormals_kernel!(normal, @Const(x), φ, t::Topology, near::Float32, n::Int32)
    p = @index(Global)
    if p <= n
        xp = x[p]
        nrm = Vec3f(0)
        if cellsample(t, φ, xp) > -near
            h = 0.5f0 * t.g.dx
            gx = cellsample(t, φ, xp + Vec3f(h, 0, 0)) - cellsample(t, φ, xp - Vec3f(h, 0, 0))
            gy = cellsample(t, φ, xp + Vec3f(0, h, 0)) - cellsample(t, φ, xp - Vec3f(0, h, 0))
            gz = cellsample(t, φ, xp + Vec3f(0, 0, h)) - cellsample(t, φ, xp - Vec3f(0, 0, h))
            l = sqrt(gx * gx + gy * gy + gz * gz)
            l > 0f0 && (nrm = Vec3f(gx, gy, gz) / l)
        end
        normal[p] = nrm
    end
end

"The diffuse particles a step on, classified by the level set and carried by the grid's velocity (see `leveldiffuse_kernel!`)."
@kernel inbounds=true function sparsediffuse_kernel!(wx, wv, wlife, wkind, φ, flow, dt::Float32, r::WhitewaterRules,
                                                     gravity::Float32, above::Float32, below::Float32)
    i = @index(Global)
    life = wlife[i]
    if life > 0f0
        t = flow.g
        x = wx[i]; vel = wv[i]
        f = cellsample(t, φ, x)
        vf = velocity(flow, x)
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
        reach = min(r.within, t.g.radius)
        (x[3] < t.g.floor || x[1] * x[1] + x[2] * x[2] > reach * reach) && (life = 0f0)
        wx[i] = x; wv[i] = vel; wlife[i] = life
    end
end

"Where each fine cell's particles start, and one past the last cell in use (resampling scanned only the slots in use)."
@kernel inbounds=true function sparsecellstart_kernel!(start, @Const(offsets), @Const(totals), used::Int32)
    l = @index(Global)
    start[l] = l <= used ? offsets[l] - totals[l] + Int32(1) : offsets[used] + Int32(1)
end

function whitewater!(w::Whitewater, s::SparseBasin, u, v, dt; substeps = 4, near = 1.5f0, above = 0.25f0, below = 0.75f0)
    be, ps, g = s.backend, s.particles, s.grid
    n = ps.n
    vel, normal = ps.cx2, ps.cy2
    used = finecells(s)
    sparsecellstart_kernel!(be, 256)(s.cellstart, s.offsets, s.totals, Int32(used); ndrange = used + 1)
    pack_kernel!(be, 256)(vel, ps.vx, ps.vy, ps.vz, Int32(n); ndrange = n)
    sparsenormals_kernel!(be, 256)(normal, ps.x, s.φ, s.topo, Float32(near * g.dx), Int32(n); ndrange = n)
    fluid = SparseFluid(ps.x, vel, s.cellstart, s.topo, g.dx)
    w.frames += 1
    emit_kernel!(be, 256)(w.x, w.v, w.life, w.head, fluid, normal, Float32(dt), w.rules, UInt32(w.frames), w.floor,
                          w.stream; ndrange = n)
    flow = Flow(s)
    for _ in 1:substeps
        sparsediffuse_kernel!(be, 256)(w.x, w.v, w.life, w.kind, s.φ, flow, Float32(dt / substeps), w.rules, w.g,
                                       Float32(above * g.dx), Float32(below * g.dx); ndrange = length(w.life))
    end
    return w
end

# ─── the level set, kept ─────────────────────────────────────────────────────

function LevelSets(path::AbstractString, s::SparseBasin)
    dims = Int.(s.grid.nb .* (BLOCK ÷ Int32(2)))
    return LevelSets(open(path * ".levelset", "w"), String(path), KA.zeros(s.backend, UInt8, dims...),
                     Array{UInt8}(undef, dims...), s.grid.lo, 2f0 * s.grid.dx, Float32((s.band + 2) * s.grid.dx), Int[])
end

@kernel inbounds=true function sparsequantize_kernel!(q, φ, t::Topology, cell::Float32, range::Float32)
    i, j, k = @index(Global, NTuple)
    p = t.g.lo + (Vec3f(i, j, k) - Vec3f(0.5f0)) * cell
    f = clamp(cellsample(t, φ, p) / range, -1f0, 1f0)
    q[i, j, k] = unsafe_trunc(UInt8, round(127.5f0 * (f + 1f0)))
end

function record!(k::LevelSets, s::SparseBasin, u, v)
    sparsequantize_kernel!(s.backend, WG3)(k.coarse, s.φ, s.topo, k.cell, k.range; ndrange = size(k.coarse))
    copyto!(k.host, k.coarse)
    write(k.io, k.host)
    push!(k.counts, length(k.host))
    return k
end
