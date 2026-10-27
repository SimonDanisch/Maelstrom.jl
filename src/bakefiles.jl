# A bake's files: the water particles (`Particles`, read back as `ParticleFrames`),
# the whitewater (`WhitewaterData`), the level sets a FLIP bake keeps (`LevelSetFrames`)
# and the basin's metadata, opened together as a `WhirlpoolBake`.

"""
    Particles(path, b; keep = 3.5, throat = 3, velocities = false)

The sink that writes each frame's water particles to `path.particles` as Float16
positions, `3 × N` one frame after the other, and how many each frame has to
`path.particles.jls` when closed; with `velocities`, their velocities the same way to
`path.velocities` (the whitewater can then be made again afterwards, see `rewhite`).
Only the water that shapes the surface is kept: all of it within `throat` of the
centre, where the funnel goes down to the floor, and the top `keep` metres elsewhere;
below that the sea goes on and no surface is drawn (see [`BasinRegion`](@ref)). The
film's water is traced from them as metaballs.
"""
struct Particles{V<:Union{Nothing,IOStream}}
    io::IOStream
    path::String
    counts::Vector{Int}
    keep::Float32
    throat::Float32
    velocities::V
end
Particles(path::AbstractString, b::Basin; keep = 3.5f0, throat = 3f0, velocities = false) =
    Particles(open(path * ".particles", "w"), path, Int[], Float32(keep), Float32(throat),
              velocities ? open(path * ".velocities", "w") : nothing)

function record!(k::Particles, s, u, v)
    host = Array(u)
    kept = [i for i in axes(host, 2) if host[3, i] > -k.keep || hypot(host[1, i], host[2, i]) < k.throat]
    write(k.io, Float16.(host[:, kept]))
    record!(k.velocities, v, kept)
    push!(k.counts, length(kept))
    return k
end
record!(::Nothing, v, kept) = nothing
record!(io::IOStream, v, kept) = write(io, Float16.(Array(view(v, 1:3, :))[:, kept]))

function Base.close(k::Particles)
    close(k.io)
    k.velocities isa IOStream && close(k.velocities)
    serialize(k.path * ".particles.jls", (; counts = k.counts, keep = k.keep, throat = k.throat))
end

"""
    ParticleFrames(path)

The frames a [`Particles`](@ref) sink wrote, memory-mapped: `frame(pf, k)` is the
`k`th, `3 × N` Float16.
"""
struct ParticleFrames
    data::Vector{Float16}
    velocities::Union{Nothing,Vector{Float16}}
    offsets::Vector{Int}      # frame k's values are data[offsets[k]+1 : offsets[k+1]]
    keep::Float32
    throat::Float32
end
function ParticleFrames(path::AbstractString)
    meta = deserialize(path * ".particles.jls")
    offsets = [0; cumsum(3 .* meta.counts)]
    mapped(f) = Mmap.mmap(f, Vector{Float16}, (last(offsets),))
    velocities = isfile(path * ".velocities") ? mapped(path * ".velocities") : nothing
    return ParticleFrames(mapped(path * ".particles"), velocities, offsets, meta.keep, meta.throat)
end
frame(pf::ParticleFrames, k) = reshape(view(pf.data, pf.offsets[k]+1:pf.offsets[k+1]), 3, :)
"Frame `k`'s particles' velocities, `3 × N` Float16 (the bake has to have kept them)."
velocity(pf::ParticleFrames, k) = reshape(view(pf.velocities, pf.offsets[k]+1:pf.offsets[k+1]), 3, :)
Base.length(pf::ParticleFrames) = length(pf.offsets) - 1

# The kinds of whitewater particle (Ihmsen et al. 2012), by where they are: in the air,
# on the surface, under it.
const SPRAY = 0x01
const FOAM = 0x02
const BUBBLE = 0x03

"""
    WhitewaterData(path)

The whitewater a bake wrote (`path.whitewater`, see `WhitewaterFrames`), memory-mapped:
`frame(w, k)` is the `k`th frame's diffuse particles, `4 × N` Float16 `(x, y, z, kind)`,
and `velocity(w, k)` their velocities, `3 × N`, in a bake that kept them
(`path.whitewater.vel`; older ones did not).
"""
struct WhitewaterData
    data::Vector{Float16}
    offsets::Vector{Int}
    velocities::Union{Nothing,Vector{Float16}}
end
function WhitewaterData(path::AbstractString)
    meta = deserialize(path * ".whitewater.jls")
    offsets = [0; cumsum(4 .* meta.counts)]
    velocities = get(meta, :velocities, false) ? Mmap.mmap(path * ".whitewater.vel", Vector{Float16}, (3sum(meta.counts),)) : nothing
    return WhitewaterData(Mmap.mmap(path * ".whitewater", Vector{Float16}, (last(offsets),)), offsets, velocities)
end
frame(w::WhitewaterData, k) = reshape(view(w.data, w.offsets[k]+1:w.offsets[k+1]), 4, :)
function velocity(w::WhitewaterData, k)
    w.velocities === nothing && throw(ArgumentError("this bake's whitewater kept no velocities: make it again with `rewhite`"))
    return reshape(view(w.velocities, 3w.offsets[k]÷4+1:3w.offsets[k+1]÷4), 3, :)
end

"""
    trimbake(src, dst, frames) -> dst

The [`bakewhirlpool`](@ref) bake at `src` cut down to `frames` (a range of its frames)
and to what drawing it needs: the particles' positions, the whitewater with its
velocities, and the metadata; not the particles' velocities, which only `rewhite` reads.
The film draws a bake's last seconds (see [`bakeframe`](@ref)), and this is what another
machine needs of it: a fifth of the bake.
"""
function trimbake(src::AbstractString, dst::AbstractString, frames::AbstractUnitRange)
    pf = ParticleFrames(src)
    ww = WhitewaterData(src)
    meta = deserialize(src * ".basin.jls")
    pmeta = deserialize(src * ".particles.jls")
    wmeta = deserialize(src * ".whitewater.jls")
    open(io -> foreach(k -> write(io, frame(pf, k)), frames), dst * ".particles", "w")
    serialize(dst * ".particles.jls", (; counts = pmeta.counts[frames], keep = pmeta.keep, throat = pmeta.throat))
    open(io -> foreach(k -> write(io, frame(ww, k)), frames), dst * ".whitewater", "w")
    open(io -> foreach(k -> write(io, velocity(ww, k)), frames), dst * ".whitewater.vel", "w")
    serialize(dst * ".whitewater.jls", (; counts = wmeta.counts[frames], velocities = true))
    serialize(dst * ".basin.jls", merge(meta, (; from = meta.from + (first(frames) - 1) / meta.fps, frames = length(frames))))
    return dst
end

"""
    WhirlpoolBake(path)

A [`bakewhirlpool`](@ref) opened for drawing: the basin, the waves, the particle and
whitewater frames, and the simulation time of the first frame.
"""
struct WhirlpoolBake{L}
    basin::Basin
    waves::Vector{Wave}
    particles::ParticleFrames
    whitewater::WhitewaterData
    from::Float64
    fps::Float64
    levels::L
    kernel::Float32     # the metaballs' radius, in particle spacings
end
function WhirlpoolBake(path::AbstractString)
    meta = deserialize(path * ".basin.jls")
    return WhirlpoolBake(meta.basin, collect(meta.waves), ParticleFrames(path), WhitewaterData(path), meta.from, meta.fps,
                         levelsets(path), Float32(get(meta, :kernel, 2.5f0)))
end

"""
    LevelSetFrames(path)

The level sets a FLIP bake kept (see `LevelSets`), memory-mapped:
every frame's on cells `cell` wide from `lo`, as UInt8 for `-range` to `range`; and
how deep its particles reach under the surface (`band`).
"""
struct LevelSetFrames
    data::Array{UInt8,4}
    lo::Point3f
    cell::Float32
    range::Float32
    band::Float32
end
function LevelSetFrames(path::AbstractString)
    m = deserialize(path * ".levelset.jls")
    data = Mmap.mmap(path * ".levelset", Array{UInt8,4}, (m.dims..., m.frames))
    return LevelSetFrames(data, m.lo, m.cell, m.range, deserialize(path * ".basin.jls").band)
end
"A bake's level sets: a FLIP bake's (see [`LevelSetFrames`](@ref)); none for one whose particles fill the water."
levelsets(path) = isfile(path * ".levelset.jls") ? LevelSetFrames(path) : nothing
