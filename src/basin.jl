# The simulated basin and what drives it: a cylinder of water holding a Lamb-Oseen
# vortex, a forcing band at the rim that relaxes the water to the swirl with the sea's
# waves in it, and the tidal race's eddies and boils (`Race`, `Stir`, `SurfaceStir`).
# A stirring model is called like a TrixiParticles source term,
# `stir(coords, velocity, density, pressure, t)`, and gives an acceleration.

"""
    Basin(; kw...)

The simulated basin: its size, the particles' spacing, the vortex it holds, the drain,
the forcing band at the rim, and the water's physics. Field comments give the units.
"""
Base.@kwdef struct Basin
    radius::Float32 = 14f0           # m, to the inside of the wall
    depth::Float32 = 6f0             # m, rest surface to floor
    spacing::Float32 = 0.3f0         # m between particles
    Γ::Float32 = 150f0               # m²/s, circulation of the vortex
    core::Float32 = 2.5f0            # m, Lamb-Oseen core radius
    drain::Float32 = 1.2f0           # m, radius of the hole in the floor
    band::Float32 = 3f0              # m, width of the forcing band at the rim
    relax::Float32 = 0.3f0           # s, relaxation to the swirl and the waves at the wall
    shortest::Float32 = 3f0          # m, the shortest of the sea's waves the basin carries
    headroom::Float32 = 6f0          # m of air above the rest surface in the search grid
    g::Float32 = 9.81f0
    ρ0::Float32 = 1000f0
    sound::Float32 = 100f0           # m/s, ten times the fastest water
    layers::Int = 4                  # boundary particle layers
    rimwall::Float32 = 3f0           # m, height of the wall above the rest surface
    arc::Float32 = 1.5f0             # rad, the sector of the rim the drained water comes back through
end

"Azimuthal speed of the basin's vortex at radius `r` (see `oseen`)."
@inline swirl(b::Basin, r) = oseen(b.Γ, b.core, r)

"The basin's vortex's free surface in balance at radius `r` (see `balanced`)."
surface(b::Basin, r) = balanced(b.Γ, b.core, b.g, r)

"""
    narutobasin(; radius = 10, depth = 3, spacing = 0.1, Γ = 45, core = 1.5, drain = 0, kw...) -> Basin

A basin the size of a Naruto whirlpool and its surroundings: with `Γ = 45` m²/s round a
1.5 m core the water turns at up to 3.4 m/s and the funnel is 1.6 m deep. No drain: the
race's guide holds the vortex (see `Race`), as the shear of the tidal current holds
Naruto's, which have none; water put back at the rim from a drain, dropped onto waves
higher than where it was put, landed inside their crests and blew the water apart.
"""
narutobasin(; radius = 10f0, depth = 3f0, spacing = 0.1f0, Γ = 45f0, core = 1.5f0, drain = 0f0, band = 2.5f0,
            shortest = 1f0, headroom = 3f0, sound = 60f0, rimwall = 1.5f0, kw...) =
    Basin(; radius, depth, spacing, Γ, core, drain, band, shortest, headroom, sound, rimwall, kw...)

"""
    Race(; current = (0, 0), eddy = 0.6, eddyscale = 3.5, eddytime = 4, guide = 4, free = 0.6, turbulence = 0, boil = 0.5,
           reach = 2.5, waves = 2.5)

The tidal race a whirlpool like Naruto's or Charybdis's lives in, as the forcing sees
it. Inside the basin the water is guided (Nielsen & Bridson 2011): its horizontal
velocity is drawn, over `guide` seconds, towards the vortex's swirl with the race's
eddies in it, the curl of a noise stream function `eddyscale` metres across, `eddy` m/s
fast, that changes over `eddytime` seconds; so the vortex holds and the water round it
churns, while the waves, the breaking and the foam stay the water's own. The guide
holds the water below the top `free` metres (see `Stir`), where the waves' orbits
are, which it would damp; there the eddies stir the water instead, as an acceleration
`turbulence` m/s² strong along them, which churns the surface without holding it.
Boils, upwellings as strong as `boil` m/s², bulge the surface. All of it fades over
`reach` metres down. The sea's waves run in `waves` times as high as the
sea around draws them. A whirlpool travels with the race's `current`; in a closed
basin a current only heaps the water against the wall, so it is drawn in the water's
own frame and `current` stays zero unless the basin is meant to tilt.
"""
Base.@kwdef struct Race
    current::Vec2f = Vec2f(0, 0)
    eddy::Float32 = 0.6f0
    eddyscale::Float32 = 3.5f0
    eddytime::Float32 = 4f0
    guide::Float32 = 4f0
    free::Float32 = 0.6f0
    turbulence::Float32 = 0f0
    boil::Float32 = 0.5f0
    reach::Float32 = 2.5f0
    waves::Float32 = 2.5f0
end

"""
    Stir(b, waves = Wave[]; origin = (0, 0), offset = 0, race = Race())

The source term: in the band at the rim, relax the water to the vortex's swirl, the
race's current and the sea's `waves` running through it (those at least `b.shortest`
long, scaled by the race; the sea's coordinates of the vortex's centre are `origin`,
film time is simulation time less `offset`), so the band makes the waves that run in
and swallows what comes back out; fully relaxed (over `b.relax` seconds) at the wall,
not at all inside the band, eased between. Inside, the race's eddies and boils.
"""
struct Stir{W<:AbstractVector{Wave}}
    basin::Basin
    waves::W
    origin::Vec2f
    offset::Float32
    race::Race
end
function Stir(b::Basin, waves = Wave[]; origin = (0, 0), offset = 0, race = Race())
    scaled = Wave[Wave(w.d, w.k, w.ω, race.waves * w.a, w.q, w.φ) for w in waves if 2π / w.k >= b.shortest]
    return Stir(b, scaled, Vec2f(origin), Float32(offset), race)
end

"""
The race's eddies at `(x, y)`, time `t`: a horizontal velocity, the curl of two octaves
of noise, drifting with the current.
"""
@inline function eddies(r::Race, x, y, t)
    s = r.eddyscale
    a = (x - r.current[1] * t) / s; b = (y - r.current[2] * t) / s
    τ = t / r.eddytime
    ψ(p, q) = noise(p + 17f0, q + τ) + 0.5f0 * noise(2.1f0 * p - 5f0, 2.1f0 * q - 1.7f0 * τ)
    δ = 0.25f0
    dψdy = (ψ(a, b + δ) - ψ(a, b - δ)) / (2f0 * δ)
    dψdx = (ψ(a + δ, b) - ψ(a - δ, b)) / (2f0 * δ)
    # The noise's gradient is about one per noise cell at most.
    return Vec2f(r.eddy * dψdy, -r.eddy * dψdx)
end

"Boils at `(x, y)`, time `t`: an upward acceleration where the noise is high, a little downwards elsewhere."
@inline function boils(r::Race, x, y, t)
    n = noise(0.6f0 * (x - r.current[1] * t) / r.eddyscale + 41f0, 0.6f0 * (y - r.current[2] * t) / r.eddyscale + 0.2f0 * t / r.eddytime)
    return r.boil * (max(n - 0.6f0, 0f0) / 0.4f0 - 0.15f0)
end

# The waves go to the device as an array: a tuple of them made the fluid system so big
# that TrixiParticles' `particle_system === neighbor_system` compared it with `memcmp`,
# which no GPU has.
Adapt.@adapt_structure Stir

"""
    orbital(waves, x, z, t) -> (horizontal, vertical)

The velocity of the water in linear `waves` (whose surface is `a sin θ`) at `x` in the
sea's coordinates, `z` below the still surface, at time `t`.
"""
@inline function orbital(waves, x::Vec2f, z, t)
    u = Vec2f(0)
    w = 0f0
    for wave in waves
        s, c = sincos(wave.k * (wave.d ⋅ x) - wave.ω * t + wave.φ)
        e = wave.a * wave.ω * exp(wave.k * min(z, 0f0))
        u += (e * s) * wave.d
        w -= e * c
    end
    return u, w
end

@inline function (s::Stir)(coords, velocity, density, pressure, t)
    b = s.basin
    race = s.race
    x, y, z = coords[1], coords[2], coords[3]
    r = sqrt(x * x + y * y)
    k = clamp((r - (b.radius - b.band)) / b.band, 0f0, 1f0)
    # Eased in: a forcing that starts with a kink left a ring on the water there.
    k = k * k * (3f0 - 2f0 * k)
    # Inside the band the race guides the water to the swirl and its eddies from below:
    # not in the top `free` metres, where the waves' orbits are, which a guide would
    # damp within a few periods, fully from twice that down, the vortex being a column
    # turning as a whole; round the core, whose funnel goes down through that layer,
    # all the way up. The eddies and boils none round the throat.
    v = swirl(b, r)
    core = clamp(2f0 - r / b.core, 0f0, 1f0)
    hold = (1f0 - k) * max(clamp((-z - race.free) / race.free, 0f0, 1f0), core)
    churn = clamp((r - 2f0) / 2f0, 0f0, 1f0)
    turn = r > 1f-4 ? v / r : 0f0
    eddy = eddies(race, x, y, t)
    aim = Vec2f(-turn * y, turn * x) + race.current + churn * eddy
    guide = (hold / race.guide) * (aim - Vec2f(velocity[1], velocity[2]))
    # In the free layer the eddies push instead (their field per m/s of `eddy`).
    top = (1f0 - k) * churn * clamp(1f0 + z / race.free, 0f0, 1f0)
    push = race.eddy > 0f0 ? (top * race.turbulence / race.eddy) * eddy : Vec2f(0)
    rising = (1f0 - k) * exp(min(z, 0f0) / race.reach)
    stir = Vec3f(guide[1] + push[1], guide[2] + push[2], rising * churn * boils(race, x, y, t))
    k == 0f0 && return stir
    sea, up = orbital(s.waves, s.origin + Vec2f(x, y), z, t - s.offset)
    target = Vec3f(-v * y / r + race.current[1] + sea[1], v * x / r + race.current[2] + sea[2], up)
    return stir .+ (k / b.relax) .* (target .- Vec3f(velocity[1], velocity[2], velocity[3]))
end

"""
    SurfaceStir(b, waves; race, ramp=2)

A tidal-race force whose free layer follows the balanced funnel rather than z=0.
This exposes the funnel wall to the same eddies as the surrounding sea. The deep
water is still gently guided to the swirl. The lookup is the balanced surface,
not an invented displacement of the rendered mesh. Incoming waves and all breakup
are evolved by the normal pressure-projected FLIP solver.
"""
struct SurfaceStir{S,H}
    base::S
    heights::H
    radius::Float32
    ramp::Float32
end
function SurfaceStir(b::Basin, waves; race, ramp=2f0)
    radius = sqrt(2f0) * (b.radius + 1f0)
    heights = Float32[surface(b, r) for r in range(0f0, radius; length=2048)]
    return SurfaceStir(Stir(b, waves; race), heights, radius, Float32(ramp))
end
Adapt.@adapt_structure SurfaceStir
device_stir(s::SurfaceStir, backend) =
    SurfaceStir(device_stir(s.base, backend), Mantle.devicearray(backend, s.heights), s.radius, s.ramp)

@inline function (s::SurfaceStir)(coords, velocity, density, pressure, t)
    b, race = s.base.basin, s.base.race
    x, y, z = Float32.(coords)
    r = sqrt(x*x + y*y)
    h = clamp(r / s.radius * Float32(length(s.heights)-1), 0f0, Float32(length(s.heights)-1))
    i = min(floor(Int, h), length(s.heights)-2)
    η = @inbounds s.heights[i+1] + (h-Float32(i)) * (s.heights[i+2]-s.heights[i+1])
    relative = z - η
    k = clamp((r-(b.radius-b.band))/b.band, 0f0, 1f0)
    k = k*k*(3f0-2f0*k)
    onset = clamp(Float32(t) / s.ramp, 0f0, 1f0)
    onset = onset*onset*(3f0-2f0*onset)
    # Ease the stirring away at the axis; no pinned surface layer at the core.
    churn = clamp(r / (1.3f0*b.core), 0f0, 1f0)
    churn = churn*churn*(3f0-2f0*churn)
    v = swirl(b, r)
    turn = r > 1f-4 ? v/r : 0f0
    eddy = eddies(race, x, y, Float32(t))
    aim = Vec2f(-turn*y, turn*x) + onset*churn*eddy
    hold = (1f0-k) * clamp((-relative-race.free)/race.free, 0f0, 1f0)
    guide = (hold/race.guide) * (aim-Vec2f(velocity[1], velocity[2]))
    top = (1f0-k)*churn*exp(min(relative, 0f0)/race.free)*onset
    push = race.eddy > 0f0 ? (top*race.turbulence/race.eddy)*eddy : Vec2f(0)
    rising = (1f0-k)*churn*exp(min(relative, 0f0)/race.reach)*onset
    acc = Vec3f(guide[1]+push[1], guide[2]+push[2], rising*boils(race,x,y,Float32(t)))
    k == 0f0 && return acc
    sea, up = orbital(s.base.waves, s.base.origin+Vec2f(x,y), relative, Float32(t)-s.base.offset)
    target = Vec3f(-turn*y+sea[1], turn*x+sea[2], up)
    return acc + (k/b.relax) * (target-Vec3f(velocity[1],velocity[2],velocity[3]))
end
