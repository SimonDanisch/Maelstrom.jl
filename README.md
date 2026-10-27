# Maelstrom.jl

Liquids on the GPU through [Mantle](https://github.com/SimonDanisch/Mantle.jl), simulated
the way film water is: particles where the surface is, a grid where it is not, and time
steps the flow sets.

<p align="center">
  <img src="docs/media/whirlpool_closeup.jpg" alt="Looking into the funnel of a simulated whirlpool" width="100%">
</p>
<p align="center"><sub>Frame 87 of a sparse FLIP whirlpool on a 5 cm grid, rendered with
<a href="https://github.com/MakieOrg/Makie.jl/tree/sd/lava-refactor/RayMakie">RayMakie</a>
at 1280×720 and 400 samples per pixel in 11 minutes on an AMD Radeon 8060S.</sub></p>

<table>
  <tr>
    <td width="50%"><img src="docs/media/rough_tidal_race.webp" alt="A whirlpool in a rough tidal race" width="100%"></td>
    <td width="50%"><img src="docs/media/crossing_seas.webp" alt="Two wave systems crossing over a whirlpool" width="100%"></td>
  </tr>
  <tr>
    <td><sub>A whirlpool in a rough tidal race: 2 s of consecutive simulation frames, none
    interpolated. 720p, 400 samples per pixel, shown here at 960 px.</sub></td>
    <td><sub>Two wave systems crossing over the vortex: 16 consecutive frames, 6.0–6.6 s.
    720p, 256 samples per pixel, shown here at 960 px.</sub></td>
  </tr>
</table>

## What it does

- **Narrow-band FLIP** (Ferstl et al. 2016) with APIC transfers (Jiang et al. 2015) and
  spatiotemporal steps (ST-FLIP, Braun et al. 2026), so a step can cross many cells.
- **Two grids:** a dense MAC grid (`FlipBasin`), and a sparse two-resolution block grid
  (`SparseBasin`, after Braun, Bender & Thuerey 2025): 16³ fine blocks where the surface
  is, a grid four times coarser elsewhere.
- **Pressure:** multigrid-preconditioned conjugate gradient, with flexible GMRES when the
  V-cycle is not a usable CG preconditioner.
- **Swirl that survives big steps:** advection–reflection (Zehnder et al. 2018) keeps a
  vortex turning at one step per frame.
- **Whitewater:** spray, foam and bubbles from the Ihmsen et al. 2012 model, made live
  from the solver's particles every frame.
- **Bakes and checkpoints:** particle, level-set and whitewater frames on disk, and
  portable checkpoints that restore into a larger block pool.

The scene it simulates today is a basin holding a whirlpool in a tidal race: `Basin`
holds the vortex and the forcing band at the rim, and `Stir` / `SurfaceStir` relax the
water to the swirl with the sea's waves in it and stir it with the race's eddies.

## Performance

The same 16 s whirlpool, simulated four ways on one AMD Radeon 8060S (the Strix Halo
integrated GPU) through Mantle and Vulkan. The funnel's depth checks that the faster
solvers still make the same whirlpool as the SPH reference.

| Solver | Time for 16 s | Deepest point at 16 s |
|---|---|---|
| SPH (TrixiParticles on Mantle), 10 cm, 11.2 M particles | 22,889 s (6.4 h) | −12.0 m |
| Dense FLIP, 5 cm | 13,220 s (3.7 h) | −10.7 m |
| Sparse ST-FLIP, 5 cm, one step per frame | 1,245 s (21 min) | −8.1 m |
| **Sparse ST-FLIP, 5 cm, advection–reflection (default)** | **1,877 s (31 min)** | **−11.4 m** |

One step per frame is fastest, but plain advect-then-project loses the throat's spin
and the funnel comes out too shallow. Advection–reflection keeps it for 1.5× the cost:
12× faster than the SPH reference at twice its resolution.

The SPH reference is not a slow one: on the SPHERIC dam break (3 M particles, RTX 4000
Ada) TrixiParticles on Mantle finishes in 17 minutes, DualSPHysics 5.4 in 25 minutes on
the same GPU.

## Rendering with RayMakie

The pictures above were path traced with
[RayMakie](https://github.com/MakieOrg/Makie.jl/tree/sd/lava-refactor/RayMakie), the GPU
path tracer for [Makie](https://github.com/MakieOrg/Makie.jl), built on
[Hikari](https://github.com/JuliaGraphics/Hikari.jl). The water surface is a NanoVDB level
set rebuilt from the bake's particles every frame, and the foam, bubbles and spray are
volumes inside physically based seawater, lit by a sun and sky. Any Makie scene renders
the same way, on the same Vulkan GPUs Maelstrom runs on.

Converting bakes for rendering is not part of this package yet; these frames went
through the converter of the film project Maelstrom came from.

## Use

```julia
using Mantle, Maelstrom
backend = Mantle.defaultbackend()

b = narutobasin(; radius = 18f0, depth = 12f0, Γ = 225f0, core = 2.5f0)
stir = SurfaceStir(b, narutowaves(); race = Race(; eddy = 3f0))
s = SparseBasin(b, backend; dx = 0.05f0, stir, cfl = 4f0)
initialize!(s)

white = Whitewater(wild_drawn_basin(s), backend; rules = WhitewaterRules())
advance_wild!(s, white, "bake"; tend = 10, snapshots = (8, 10))
```

## Notes

- Everything a step does runs on the device; there is no host backend.
- Mantle depends on GLFW, so precompiling needs a display (`DISPLAY`) even for headless
  simulation.
