# Lookups on regular grids shared by the solvers and the whitewater.

"Trilinear interpolation of the 3D array `A` at index-space point `q` (element `a` at `a - 1`), clamped to the array."
@inline function trilinear(A, q::Vec3f)
    n1, n2, n3 = size(A)
    x = clamp(q[1], 0f0, Float32(n1 - 1)); y = clamp(q[2], 0f0, Float32(n2 - 1)); z = clamp(q[3], 0f0, Float32(n3 - 1))
    i = min(unsafe_trunc(Int, x), n1 - 2); j = min(unsafe_trunc(Int, y), n2 - 2); k = min(unsafe_trunc(Int, z), n3 - 2)
    fx = x - i; fy = y - j; fz = z - k
    i += 1; j += 1; k += 1
    @inbounds begin
        c00 = Float32(A[i, j, k]) * (1f0 - fx) + Float32(A[i + 1, j, k]) * fx
        c10 = Float32(A[i, j + 1, k]) * (1f0 - fx) + Float32(A[i + 1, j + 1, k]) * fx
        c01 = Float32(A[i, j, k + 1]) * (1f0 - fx) + Float32(A[i + 1, j, k + 1]) * fx
        c11 = Float32(A[i, j + 1, k + 1]) * (1f0 - fx) + Float32(A[i + 1, j + 1, k + 1]) * fx
    end
    return (c00 * (1f0 - fy) + c10 * fy) * (1f0 - fz) + (c01 * (1f0 - fy) + c11 * fy) * fz
end

"The cell coordinates (from 0) of `p` in a grid of cells `R` wide from `lo`, `dims` a side, clamped to the grid."
@inline function cellcoords(lo, R, dims, p)
    i = clamp(unsafe_trunc(Int32, floor((p[1] - lo[1]) / R)), Int32(0), dims[1] - Int32(1))
    j = clamp(unsafe_trunc(Int32, floor((p[2] - lo[2]) / R)), Int32(0), dims[2] - Int32(1))
    k = clamp(unsafe_trunc(Int32, floor((p[3] - lo[3]) / R)), Int32(0), dims[3] - Int32(1))
    return Vec{3,Int32}(i, j, k)
end

"The 1-based linear index of the cell at coordinates `c` (from 0) in a grid `dims` a side."
@inline gridcell(dims, c) = Int(c[1]) + Int(dims[1]) * (Int(c[2]) + Int(dims[2]) * Int(c[3])) + 1
