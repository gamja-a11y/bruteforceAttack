# v4 audit and continuous ROI proof

## Audit of v3

Proven: the two Padé identities and nonnegative coefficient inequalities are
checked symbolically. The tighter geometric bound uses the chord above the
convex reciprocal product on its certified q range. Its nonnegative
pointwise majorant can be squared and integrated with exact disk moments.
The independent v2 gamma/disk computation overlaps the v3 original bound.

Proven: GL brackets are disjoint, lie strictly inside (-1,1), and exhibit
sign changes. There are exactly n roots because the polynomial has degree n.
The GL formula encloses positive weights. On each analytic parameter disk
of twice the real half-width, the degree-(2n-1) Taylor tail is bounded using
Cauchy's formula. The two-axis telescoping error is 4 M 2^-(2n-1) times
cell area, since the quadrature weights are positive and sum to one.
Arb branch/pole tests are required for all enclosing boxes.

The signed boundary normal is (dy,-dx), with the determinant of each
reflection/swap applied to its parametric orientation. Writing H as half
the unit-interval integral of G(uq) makes the q=0 formula regular.

## Continuous extension

Proven: for real q in the certified interval [0,Q], the derivative of G
is bounded by applying the product and quotient rules to the exact Padé
remainders. The code uses s<=Q/2, s'<=1/2 and positive lower bounds for
both denominators. The phase exponential has modulus one on the real axis.
Differentiation under the compact area integral gives
|grad L2|<=area*sup|G2'|*2*max|r-r'|/(lambda*z^3).

The physical field at a cell center is within the certified GL disk
about the Arb-enclosed complex sum. For every point of its polar cell,
the additional radius is the spatial Lipschitz bound times a radial-plus-arc
distance. The octant is valid only for this AR=1 case. Radial limits include
the circular edge exactly; there are no clipped Cartesian cells.
The normalized cell weights are (r1²-r0²)/(25*nangular) and sum to one.
Squaring the cell magnitude bounds, summing with these exact weights and
taking square roots encloses the continuous norm. Add the independent
Arb-certified Taylor remainder norm to bound the raw RS-I–PP field norm.

Arb arithmetic errors are already included in the enclosures. The final
comparison is against a rigorously enclosed geometric bound. The stored
oscillatory reference is empirical. Taking the smaller of two independently
certified upper endpoints is valid, but does not imply a phase-preserving
improvement. A finite schedule is not a guaranteed adaptive algorithm.
