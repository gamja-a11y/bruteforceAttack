# Independent RS-I–Padé mathematical audit

For q=|r-s|²/z² and K=2πz/λ all fields share unit aperture illumination,
the prefactor 1/(iλz), and the omitted carrier exp(iK). RS-I has
amplitude (1+q)^-1+i(1+q)^-3/2/K and phase sqrt(1+q)-1. In the
physical derivative kernel z/R is the obliquity factor and 1/R is
spherical spreading. Fresnel uses unit amplitude and phase q/2.
Padé uses phase 4q(q+2)/(16+12q+q²) and amplitude
1/(1+q)+i(1+q/4-q²/48)/(K(1+7q/4+35q²/48)).
The exact RS-I minus Padé kernel is not the RS-I minus Fresnel kernel.
The target is sqrt(int_disk |U_RS-U_PP|²/(25π)), an absolute norm.

The continuous kernels are bounded on the compact product domains, so
Fubini expands the norm into int_ROI int_A int_A D(r,s) conj(D(r,t)).
For AR=ax/ay, the beta change of variables u=(|x|/ax)^p,v=(|y|/ay)^p
gives mu_2i,2j = 4 ax^(2i+1) ay^(2j+1) Gamma((2i+1)/p) Gamma((2j+1)/p) /
[p² Gamma(1+(2i+2j+2)/p)]. Set mu_00=100π. Independent reflection
eliminates odd source moments for all AR. After expanding q^d, apply
the translation/moment operator sum mu_ab ∂x^a ∂y^b/(a!b!).
The resulting polynomial has even independent powers and does not
require x/y exchange, including AR=2 or 4. Disk monomial moments of
X=(x/5)^2,Y=(y/5)^2 are (1/2)_i(1/2)_j/(i+j+1)! via beta integration.
Ordinary polynomial products at each X degree compute the real and
imaginary squares separately; signed cross terms are retained.
For a unit disk source and an axial observer, integrating q^d gives
π/(d+1) at z=1 for d=1,2,3. The corresponding disk checks for X^d
use elementary beta moments. Only at the end use Minkowski to add
the uniform kernel remainder times 100π/(λz). The exact squared norm
is nonnegative; a negative interval lower endpoint is safely replaced by zero.

For p>2 the enclosing source radius is the dual-norm expression
(ax^(2p/(p-2))+ay^(2p/(p-2)))^((p-2)/(2p)); p=2 uses max(ax,ay).
Adding ROI radius then dividing by z yields a safe q bound.
For Bernstein rho=2, Arb checks the enclosing rectangle is free of
the principal sqrt cut and poles. The denominator polynomials in q
also account for the Padé amplitude denominator. Full closed angular
intervals cover the ellipse boundary; the maximum modulus principle
extends the Arb maximum to the interior. The Chebyshev–Lobatto
interpolation error is <=4M rho^-n/(rho-1) (Trefethen, Approximation
Theory and Approximation Practice, Theorem 8.2; the analytic function
and magnitude bounds hold on the closed ellipse and in a neighborhood).
Source: https://people.maths.ox.ac.uk/trefethen/trefethen_sample.pdf
For complex values the same contour/aliasing proof uses the modulus.
This is interpolation, not truncation. Newton divided differences,
basis conversion, gamma and Gram products all use Arb enclosures.

Direct quadrature is empirical. The polar ROI uses octant symmetry
only at AR=1; the independent Cartesian comparison uses quadrant
reflection and the map x=5(1-u²), y=5u sqrt(2-u²) v. The 35-digit
calculation uses a separate implementation and its finite-order error
is not removed by precision. The original oscillatory value is
||L_2|| + ||rho_3||, a numerical estimate of a theoretical upper bound,
whereas the exact residual norm contains the complete exponential.

Reproduce: python fresnel_independent_audit.py --out NEW_DIRECTORY
