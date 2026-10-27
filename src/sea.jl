# The sea round a basin: its waves, the Lamb-Oseen vortex's swirl and balanced surface,
# and the hash and value noise the stirring reads. A scene that draws the sea around a
# simulated basin uses the same `Wave`s the basin is fed.

"One Gerstner wave: direction, wave number, angular frequency, amplitude, steepness, phase."
struct Wave
    d::Vec2f
    k::Float32
    ω::Float32
    a::Float32
    q::Float32
    φ::Float32
end

"""
    waves(; n = 48, wind = 0.6, spread = 1.1, hs = 0.35, λmin = 0.5, λmax = 24, seed = 3)

Waves for a wind from angle `wind`, directions spread `±spread` around it,
wavelengths between `λmin` and `λmax`, amplitudes falling as the wavelength's
power 1.25 so the sea state has roughly significant height `hs`.
"""
function waves(; n = 48, wind = 0.6, spread = 1.1, hs = 0.35, λmin = 0.5, λmax = 24.0, seed = 3)
    rng = MersenneTwister(seed)
    λs = exp.(range(log(λmax), log(λmin); length = n))
    raw = λs .^ 1.25
    scale = hs / (4 * sqrt(sum(abs2, raw) / 2))
    return [begin
                θ = wind + spread * (2rand(rng) - 1)
                k = 2π / λ
                Wave(Vec2f(cos(θ), sin(θ)), k, sqrt(9.81 * k), scale * r, 0.55, 2π * rand(rng))
            end for (λ, r) in zip(λs, raw)]
end

"The sea's waves round the whirlpool: a wind sea of 1.4 m significant height, 1–10 m long."
narutowaves(; hs = 1.4, λmin = 1.0, λmax = 10.0, wind = 0.4, spread = 0.9, seed = 5) =
    waves(; n = 32, wind, spread, hs, λmin, λmax, seed)

"Azimuthal speed of the Lamb-Oseen vortex of circulation `Γ` and core radius `core` at radius `r`."
@inline oseen(Γ, core, r) = Γ / (2f0 * Float32(π) * max(Float32(r), 1f-4)) * (1f0 - exp(-(Float32(r) / core)^2))

"""
    balanced(Γ, core, g, r; steps = 2000)

The free surface over the Lamb-Oseen vortex in balance at radius `r`: the pressure
gradient that carries its swirl, integrated in from far away, where the sea is at rest
at height zero.
"""
function balanced(Γ, core, g, r; steps = 2000)
    # ∫_r^∞ v²/(g r') dr', in log r for the long tail.
    a, z = log(max(Float32(r), 1f-3)), log(50f0 * core + 50f0 * max(Float32(r), 1f0))
    h = (z - a) / steps
    s = 0f0
    for i in 0:steps
        ρ = exp(a + i * h)
        s += ((i == 0 || i == steps) ? 0.5f0 : 1f0) * oseen(Γ, core, ρ)^2 / g
    end
    return -s * h
end

"A smooth hash in [0, 1): the rim's wobble and the foam it brings in."
@inline function hash01(i::Int32, j::Int32)
    k = (i % UInt32) * 0x27d4eb2d ⊻ (j % UInt32) * 0x165667b1
    k = (k ⊻ (k >> 15)) * 0x85ebca6b
    k = (k ⊻ (k >> 13)) * 0xc2b2ae35
    return Float32(k ⊻ (k >> 16)) / 4.2949673f9
end
"Value noise: the hash, smoothly interpolated between integer points."
@inline function noise(x::Float32, y::Float32)
    xi = floor(x); yi = floor(y)
    fx = x - xi; fy = y - yi
    sx = fx * fx * (3f0 - 2f0fx); sy = fy * fy * (3f0 - 2f0fy)
    i = unsafe_trunc(Int32, xi); j = unsafe_trunc(Int32, yi)
    a = hash01(i, j); b = hash01(i + Int32(1), j)
    c = hash01(i, j + Int32(1)); d = hash01(i + Int32(1), j + Int32(1))
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy
end
