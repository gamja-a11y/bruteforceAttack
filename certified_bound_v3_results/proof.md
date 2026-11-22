# v3 proof and scope

**Proven.** The exact Padé identities and their signs are checked algebraically.
For real q>=0, |A(q)|<=sqrt(1+K^-2)/(1+q). Also
delta/q^5=1/[(sqrt(1+q)+1)^5(16+12q+q^2)]<=1/[512(1+3q/4)].
The product reciprocal f(q)=1/[(1+q)(1+3q/4)] is convex on q>=0.
Its chord on [0,Q] is 1-dq and bounds it above. Therefore the
pointwise residual is bounded by (C q^5 - sqrt(1+K^-2) K d q^6/512),
a nonnegative polynomial on the physical domain. Its continuous field L2
bound uses exact superellipse gamma moments and disk beta moments of the
squared polynomial, with outward rounding. The disk monomial moment is
R^(2i+2j) B(i+1/2,j+1/2)/(pi(i+j+1)). The normalized constant moment is one.
Both geometric bounds apply to the same unfit raw absolute field norm.

**Proven.** The boundary sweep uses eight reflected/swapped sectors. Their
orientation determinants multiply n ds=(dy,-dx). The q=0-safe substitution
H(q)=1/2 integral_0^1 G(uq)du preserves analyticity and gives H(0)=0.
The implementation uses the Padé residual, not a Fresnel phase approximation.
At each real cell the Cauchy contour is contained in a rectangular Arb box
around the complex disk with twice the real half-width. All branch/pole
conditions are checked on that box. A degree-d Taylor tail is <=M*2^-d.
On [0,1], a rule of degree d and absolute-weight sum S has error <=
(1+S) M*2^-d. Telescoping over both coordinates gives the factor (1+S)^2.

**Interval-certified.** SciPy supplies candidate locations only. Disjoint
rational root brackets with Arb-certified sign changes account for all n
Legendre roots. The exact Gauss weight formula on each bracket certifies
positive weights. Complex contributions are summed before adding remainder
to each component separately. No experimental convergence is used as a bound.

**Unverified.** The high-order boundary probe is at (3,4); continuous ROI
phase-preserving extension, a Taylor-model fallback and automatic tolerance-based
selection are not implemented. The continuous improvement is geometric.
The empirical oscillatory reference is only a comparator.
