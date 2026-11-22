# v6 direct continuous L2 certificate

## Audit of v5
The quadratic Gram expression correctly uses the real part of c_i conj(c_j)
and a factor of two off the diagonal for real monomials. The polar antiderivative
includes the radial Jacobian; 8/(25*pi) normalizes the octant for AR=1.
For the Cauchy polydisk GL estimate the Taylor tail is 2^-(2n-1) M per variable,
and two telescoped quadratures with positive weights give the factor four.
The directional third derivative is G'''(q')³+3G''q'q'', with q'''=0.
The real interval derivative majorants and radial-plus-arc displacement
produce a valid but loose remainder. v6 does not reuse that remainder or
the v5 serialized center-field replay as inputs to any certificate.

## Factorization and symmetry (proven)
Let D denote the exact complex RS-I minus Padé kernel including 1/(i lambda z).
By Fubini the squared norm is int_ROI int_A int_A D(r,s) conj(D(r,t))
dA(s)dA(t)dA(r)/(25*pi). A modulus or triangle inequality inside either
source integral discards the cross-term cancellation. Hermitian exchange
permits a real-part integral on half the paired domain; reflections of source
and observation coordinates together permit symmetry reduction. Neither
reduces the number of independent continuous coordinates for generic p=12.
For the circular p=2 case, continuous rotation reduces the observation
dependence to radius; this is used only in the empirical direct comparison.

## Interval-certified polynomial method
The exact whole residual is used, with no Taylor phase truncation. Complex
Arb boxes check analyticity on the entire enclosing rectangle of the rho=2
Bernstein ellipse. Arc boxes cover the ellipse boundary and bound its
maximum; the maximum modulus principle handles the interior. The Lobatto
interpolant of degree n has uniform error at most 4*M*2^-n. Exact-node values,
discrete cosine coefficients and polynomial conversion are enclosed in Arb.
The effect on the normalized field L2 norm is at most A/(lambda*z) times
this uniform error. There is no boundary or ROI numerical quadrature in
the certified path.

Expand the polynomial in q=(distance/z)^2. Source reflection kills odd
powers, and the remaining coefficients use exact superellipse gamma moments.
This produces a global field polynomial in X=(x/5)^2 and Y=(y/5)^2.
The real and imaginary polynomials are each squared with a Kronecker stride
of 2n+1, so terms cannot alias; the two squares retain signed cross terms.
The normalized disk moment of X^i Y^j is
(1/2)_i (1/2)_j/(i+j+1)!. Summing the Gram polynomial encloses its exact
squared norm, including every rounding error. A negative lower endpoint is
replaced by zero using nonnegativity; a negative upper endpoint fails.
Add the validated uniform residual bound at norm level before squaring
to report the upper endpoint for the true squared norm.

The direct paired quadrature scales as n^6 before symmetry; the source
moment transform is O(n^4), and the disk Gram uses polynomial convolution
rather than six-dimensional quadrature. A certified low-rank decomposition
could reduce cost only with a uniform or operator-norm tail enclosure; it
is unverified here. Reported high-precision direct quadrature is empirical.
