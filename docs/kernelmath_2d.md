# Notes on 2D Kernel Computation

Here we summarize some ways that accelerate the kernel computation in 2D. We follow the structure outlined in the notes on 1D kernel computation. 

Again using the displacement vector $\mathbf{p} = (x + \rho \cos{\alpha}, y - \rho\sin{\alpha}\cos{\phi}, h+\rho\sin{\alpha}\sin{\phi})$, we have a kernel for the $x$-component of the sheet current density:

$$K_1(x,y) = \int d\alpha\int d\rho \rho W(\rho) \frac{\cos{\phi}p_y-\sin{\phi}p_z}{p^3}$$

We first introduce some easier notation, since most of the integrand we actually do not need:

$$a(\rho,\alpha) = \frac{\cos{\phi}p_y-\sin{\phi}p_z}{p^3}$$

If we then also use the same tapered definition of the window function as before, the integral splits:

$$K_1(x,y) = \int d\alpha \int_0^{\rho_1} \rho a(\rho,\alpha)d\rho + \int d\alpha \int_{\rho_1}^{\rho_2} \rho \frac{\rho_2-\rho}{\rho_2-\rho_1}a(\rho,\alpha)d\rho$$

To be able to, once again, use $N$th order Gauss-Legendre quadrature over $\rho$ we introduce two auxiliary variables

$$u = \frac{2\rho}{\rho_1}-1$$

$$u' = -2\frac{\rho-\rho_2}{\rho_1-\rho_2}+1$$

The integrals become a little longer, but all the more computable:

$$K_1(x,y) = (\frac{\rho_1}{2})^2\int d\alpha \int_{-1}^1 du (u+1) a(\rho(u), \alpha) +$$ $$\frac14(\rho_2-\rho_1)\int d\alpha \int_{-1}^1 du' (\frac12(\rho_2-\rho_1)(u'-1)+\rho_2)(1-u') a(\rho(u'),\alpha)$$

Now introducing the quadrature at points $u_i$ and $u'_i$ with weights $w_i$ we have

$$K_1(x,y) = (\frac{\rho_1}{2})^2\int d\alpha \sum_{i=1}^N w_i (u_i+1) a(\rho(u_i), \alpha) +$$ $$\frac14(\rho_2-\rho_1)\int d\alpha \sum_{i=1}^N w_i (\frac12(\rho_2-\rho_1)(u'_i-1)+\rho_2)(1-u'_i) a(\rho(u'_i),\alpha)$$

We find that, in practice, $N=6$ is sufficient for the 2D kernel as well. Expanding the integral over $\alpha$ as a standard Riemann sum then gives the code implemented in Forward.py. For this, of course, the $y$-component kernel $K_2$ is required too. Using the following auxiliary function

$$b(\rho, \alpha) = -\frac{\cos{\phi}p_x}{p^3},$$

we find a similar expression for $K_2$:

$$K_2(x,y) = (\frac{\rho_1}{2})^2\int d\alpha \sum_{i=1}^N w_i (u_i+1) b(\rho(u_i), \alpha) +$$ $$\frac14(\rho_2-\rho_1)\int d\alpha \sum_{i=1}^N w_i (\frac12(\rho_2-\rho_1)(u'_i-1)+\rho_2)(1-u'_i) b(\rho(u'_i),\alpha)$$
