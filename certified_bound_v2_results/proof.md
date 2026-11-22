# Mathematical audit

- **Proven, kernel and Taylor.** Cross-multiplied Padé identities and denominator signs
  are checked symbolically. For real phase error x, the integral Taylor
  remainder of exp(ix) is at most |x|^3/6 because |exp(itx)|=1.
- **Proven, normal and continuity.** With q=|r'-r|²/z², div((r'-r)H)=2H+2qH'=G.
  For CCW orientation n ds=(dy,-dx). Every reflected or swapped sector is
  multiplied by its orientation determinant. Swapping x,y is an ROI symmetry
  only at AR=1. H(q)=1/2 integral_0^1 G(uq)du removes division by zero,
  gives H(0)=G(0)/2=0, and preserves analyticity. G=O(q^5) there.
- **Proven, v1 interpolation.** The Lobatto-series coefficient convention and
  affine antiderivative are consistent. A rectangle covering the complete
  Bernstein ellipse must avoid the square-root cut and all denominator zeros.
  Coefficient balls include rounding. However, use of q.mid() after an
  outward-rounded intersection does not explicitly prove the center remains in
  the exact piecewise support in edge cases. v2 bypasses that argument; it does
  not reissue the v1 composite certificate.
- **Proven, v1 enclosure design.** Pointwise interval bounds and positive
  normalized disk cell weights bound the continuous L2 integral; enclosing
  terms independently loses cancellation. A plotted decimal is not the
  certificate; use the outward endpoint of the stored ball.
- **Proven, v2 high-order remainder.** On each two-parameter rectangle, check
  holomorphy and |f|<=M on a complex polydisk with radii twice the real
  half-widths. Cauchy's estimate bounds the degree-n Taylor tail in either
  parameter by M*2^-n. The exact-rational interpolatory rule has weight
  absolute sum S. Telescoping the two one-axis errors gives
  M*(1+S)^2*2^-n times rectangle area. The complex quadrature sum is accumulated
  before adding the enclosure. This is a certified pointwise test only.
- **Proven, continuous geometric norm.** J5 and J15 are polynomials in x²,y²
  with positive coefficients from gamma moments. Square them algebraically
  and integrate each term using the exact normalized disk beta moment.
  The weights integrate the physical radius-5 disk divided by 25*pi.
  Both B0 and an independently valid composite B_cert bound the same raw
  ||U_RS-U_PP|| with prefactor 1/(i lambda z). Taking the minimum of two
  independently certified upper endpoints is valid; no numerical phase fit
  or relative RS normalization is used.
- **Unverified.** No phase-preserving continuous-ROI certificate from the
  high-order probes. The empirical B0_osc2 is a benchmark only. Interval GL
  would reduce node cost but requires verified nodes and weights; derivative
  Taylor models and analytic subdivision near stationary phase are alternatives.
  Oscillatory rules require a separate validated remainder.
