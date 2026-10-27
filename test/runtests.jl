using Test, Maelstrom
import Mantle

@testset "Maelstrom" begin
    @testset "the sea and the vortex" begin
        Γ, core = 100f0, 2f0
        @test Maelstrom.oseen(Γ, core, 0f0) == 0f0
        # Far out the Lamb-Oseen vortex is a free vortex.
        @test Maelstrom.oseen(Γ, core, 50f0) ≈ Γ / (2f0 * Float32(π) * 50f0) rtol = 1f-3
        b = narutobasin()
        # The balanced surface falls towards the centre and stays below the sea.
        @test Maelstrom.surface(b, 0f0) < Maelstrom.surface(b, 3f0) < Maelstrom.surface(b, 9f0) < 0f0
        w = waves(; n = 16, hs = 1.0)
        @test length(w) == 16 && eltype(w) == Wave
        stir = Stir(b, w; race = Race())
        # The basin carries only the waves at least `b.shortest` long.
        @test all(x -> 2π / x.k >= b.shortest, stir.waves)
    end

    backend = Mantle.defaultbackend()

    @testset "a sparse basin runs, keeps its water and bakes what it ran" begin
        b = narutobasin(; radius = 6f0, depth = 3f0, Γ = 30f0, core = 1.5f0, band = 1.5f0, headroom = 2f0)
        stir = SurfaceStir(b, narutowaves(); race = Race(; eddy = 2f0))
        s = SparseBasin(b, backend; dx = 0.2f0, stir, cfl = 4f0, blocks = 2000, capacity = 2_000_000)
        initialize!(s)
        @test s.particles.n > 0
        white = Whitewater(wild_drawn_basin(s), backend; rules = WhitewaterRules(), capacity = 500_000)
        root = mktempdir()
        advance_wild!(s, white, root; tend = 0.25, snapshots = (0.25,))
        @test s.t ≈ 0.25
        @test isfinite(maxspeed(s))
        @test 0.95 < Maelstrom.volume(s) / s.volume0 < 1.05

        bake = WhirlpoolBake(joinpath(root, "frame0006", "whirlpool"))
        @test length(bake.particles) == 1
        @test bake.levels isa LevelSetFrames
        @test bake.from ≈ 0.25

        path = joinpath(root, "state.jls")
        save_sparse_checkpoint(s, white, path)
        restored = load_sparse_checkpoint(path, backend)
        @test restored.simulation.t == s.t
        @test restored.simulation.particles.n == s.particles.n
        @test restored.whitewater.frames == white.frames
    end

    @testset "a dense basin steps" begin
        b = narutobasin(; radius = 4f0, depth = 2f0, Γ = 15f0, core = 1f0, band = 1f0, headroom = 1.5f0)
        s = FlipBasin(b, backend; dx = 0.2f0, capacity = 1_000_000)
        initialize!(s)
        for _ in 1:3
            step!(s, 1 / 48)
        end
        @test isfinite(maxspeed(s))
        @test 0.95 < Maelstrom.volume(s) / s.volume0 < 1.05
    end
end
