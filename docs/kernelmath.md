# Notes on 1D Kernel Computation

Here we summarize some ways that accelerate the kernel computation. The closed-form expression for the one-dimensional signal is:

$$S(y) = \frac{\mu_0}{4\pi} \int dy'' j(y'') \underbrace{\int\!\!\int d\rho\,d\alpha\, \rho W(\rho) 
		\frac{2(p_y'' \cos\phi - \cos\gamma \sin\phi p_z)}{p_y''^2+p_z^2}}_{K(u)}$$

Here $u$ is the kernel coordinate. The window function is denoted $W$. Furthermore, 

$$p_y'' = u - \rho(\cos\alpha \sin\gamma + \cos\gamma\sin\alpha\cos\phi)$$

$$p_z = h+\rho \sin \alpha \sin \phi$$

The integral over $\rho$ can actually be solved completely analytically, although this is not always the most accurate numerical method. The first step is to introduce some constants to get rid of the noise of trig functions:

$$K(u) = \int d\alpha \int d\rho \rho W(\rho) \frac{2(c_1 p_y'' - c_2 p_z)}{p_y''^2+p_z^2}$$

Then writing $p_y''=u-A\rho$ and $p_z = h+B\rho$ we can perform a fractional decomposition and obtain

$$K(u) = \int d\alpha  2 \mathrm{Re} \int d\rho \rho W(\rho) \cdot  \frac{ic_1-c_2}{h+iu + (B-iA)\rho}$$

Upon introducing the simple linearly tapered $W(\rho)$ between $\rho_1$ and $\rho_2$, everything becomes solvable. From here, there are two roads. The first is directly integrating with something like Mathematica, which yields:

$$K(u) = \int d\alpha\, 2\,\mathrm{Re}\left\{\kappa\left[\frac{s(\rho_1+\rho_2)+2w}{2s^2} + \frac{w\ln w}{s^2} + \frac{w(w+s\rho_1)\ln(w+s\rho_1)}{(\rho_2-\rho_1)\,s^3} - \frac{w(w+s\rho_2)\ln(w+s\rho_2)}{(\rho_2-\rho_1)\,s^3}\right]\right\}$$

with $w = h+iu$, $s = B-iA$ and $\kappa = ic_1-c_2$. Note that this integration is only valid when the full SQUID loop is above the sample ($h > \rho_2 |\sin{\phi}|$). This might seem a trivial point, but it is essential when co-optimizing the SQUID height and radius.

Far from the sensor (small $|s\rho_2/w|$), this expression is a difference of very large, nearly equal terms, so it loses most of its precision to rounding, catastrophically so in float32.

The second road avoids this: expanding $1/(w+s\rho)$ as a geometric series and integrating term by term gives, with $q = \rho_1/\rho_2$,

$$K(u) = \int d\alpha\, 2\,\mathrm{Re}\left\{\frac{\kappa\,\rho_2^2}{w}\sum_{n=0}^{\infty}\left(-\frac{s\rho_2}{w}\right)^n \frac{1-q^{n+3}}{(n+2)(n+3)(1-q)}\right\},$$

which converges for $|s\rho_2/w|<1$ and has no cancellation. For proper accuracy, a truncated series expansion must applied to those points where $|s\rho_2/w|<0.7$.

These methods can, in theory, give great accuracy. However, the geometric sum needs many terms, especially for small SQUIDs far away from the surface, and the closed-form expression struggles with machine accuracy due to the divisions by $s^2$, $s^3$ and $\rho_2-\rho_1$. Sufficiently accurate, more robust, and faster is a Gauss-Legendre quadrature over $\rho$:

$$\int_0^{\rho_2} d\rho \rho W(\rho) \frac{2(c_1 p_y'' - c_2 p_z)}{p_y''^2 + p_z^2}$$

$$= (\frac{\rho_1}{2})^2 \sum_{i=1}^N w_i (x_i + 1) g(\frac{\rho_1}{2}(x_i+1))$$

$$+ \frac{\rho_2-\rho_1}{2} \sum_{i=1}^N w_i (\frac{\rho_2-\rho_1}{2} (1+x_i) + \rho_1) \frac{1-x_i}{2} g(\frac{\rho_2-\rho_1}{2}(1+x_i)+\rho_1)$$

Here, $w_i$ is the $i$-th weight and $x_i$ is the $i$-th node of the Legendre polynomial of order $N$. The fraction $g(\rho) = \frac{2(c_1 p_y'' - c_2 p_z)}{p_y''^2 + p_z^2}$ is evaluated at the quadrature nodes.
The LG quadrature is sufficiently accurate for this particular kernel and window, that order $N=6$ is generally sufficient. The exact order is exposed as an optional parameter in Forward1D. H