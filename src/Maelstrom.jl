"""
    Maelstrom

Liquids on the GPU through Mantle, simulated the way film water is: narrow-band FLIP
(APIC transfers, spatiotemporal steps), dense (`FlipBasin`) or on a sparse two-resolution
block grid (`SparseBasin`), with the pressure solved by multigrid-preconditioned CG or
flexible GMRES; whitewater from the Ihmsen model; bake files and checkpoints.

The scene it simulates today is a basin holding a whirlpool, stirred by a tidal race
(`Basin`, `Race`, `Stir`, `SurfaceStir`).
"""
module Maelstrom

import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const, @localmem, @synchronize
import AcceleratedKernels as AK
import Mantle
import Adapt
using Adapt: adapt
using GeometryBasics: Vec, Vec2f, Vec3f, Point, Point2f, Point3f
using LinearAlgebra: ⋅, norm, UpperTriangular
using Random: MersenneTwister
using Statistics: mean, quantile
using Mmap, Serialization
import SHA
import JSON

include("sea.jl")
include("basin.jl")
include("interpolation.jl")
include("bakefiles.jl")
include("whitewater.jl")
include("flip.jl")
include("sflip.jl")
include("fgmres.jl")
include("checkpoint.jl")
include("live.jl")
include("run.jl")

# The scene.
export Basin, Race, Stir, SurfaceStir, Wave, waves, narutobasin, narutowaves
# The solvers.
export FlipBasin, SparseBasin, initialize!, step!, maxspeed, bakeflip
# Whitewater.
export Whitewater, WhitewaterRules, WhitewaterFrames, whitewater!, perparticle, SPRAY, FOAM, BUBBLE
export prepare_live_whitewater, update_live_whitewater!, write_live_whitewater_provenance
# Bakes and checkpoints.
export Particles, ParticleFrames, WhitewaterData, WhirlpoolBake, LevelSets, LevelSetFrames, BandParticles, frame, record!
export save_sparse_checkpoint, load_sparse_checkpoint
# Running a sparse basin through film frames.
export advance_wild!, snapshot_wild, wild_drawn_basin

end
