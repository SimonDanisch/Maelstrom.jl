# Maelstrom.jl

Liquids on the GPU through [Mantle](https://github.com/SimonDanisch/Mantle.jl), simulated
the way film water is: particles where the surface is, a grid where it is not, and time
steps the flow sets.

- **Narrow-band FLIP** (Ferstl et al. 2016) with APIC transfers (Jiang et al. 2015) and
  spatiotemporal steps (ST-FLIP, Braun et al. 2026), so a step can cross many cells.
- **Two grids:** a dense MAC grid (`FlipBasin`), and a sparse two-resolution block grid
  (`SparseBasin`, after Braun, Bender & Thuerey 2025): 16³ fine blocks where the surface
  is, a grid four times coarser elsewhere.
- **Pressure:** multigrid-preconditioned conjugate gradient, with flexible GMRES when the
  V-cycle is not a usable CG preconditioner.
- **Whitewater:** spray, foam and bubbles from the Ihmsen et al. 2012 model, made live
  from the solver's particles every frame.
- **Bakes and checkpoints:** particle, level-set and whitewater frames on disk, and
  portable checkpoints that restore into a larger block pool.

The scene it simulates today is a basin holding a whirlpool in a tidal race: `Basin`
holds the vortex and the forcing band at the rim, and `Stir` / `SurfaceStir` relax the
water to the swirl with the sea's waves in it and stir it with the race's eddies.

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
- Rendering the bakes (NanoVDB level sets and whitewater volumes for Hikari) is not part
  of this package yet.
