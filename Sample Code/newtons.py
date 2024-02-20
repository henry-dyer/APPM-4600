import numpy as np
import matplotlib.pyplot

class main:
    def Newton_explore(self):
        f = lambda x: (x-2)**3
        fp = lambda x: 3*(x-2)**2

        p0 = 1.2
        Nmax = 100
        tol = 1.e-9

        [p,pstar,info,it] = self.newton(f,fp,p0,tol, Nmax)

        print('%16.16e  %5.5e ' % (pstar,info))
        
        r1 = abs(p[0:it-1]-2)/abs(p[1:it]-2)
        r2 = abs(p[0:it-1]-2)/(abs(p[1:it]-2)**2)

        matplotlib.pyplot.figure(1)
        matplotlib.pyplot.semilogy(r1)
        matplotlib.pyplot.figure(2)
        matplotlib.pyplot.semilogy(r2)
        matplotlib.pyplot.show()

        return
        
    def newton(self,f,fp,p0,tol, Nmax):
        p = np.zeros(Nmax+1)
        p[0] = p0
        it = 0

        for it in range(Nmax):
            p1 = p0-f(p0)/fp(p0)
            p[it+1] = p1
            if abs(p1 - p0) < tol:
                pstar = p1
                info = 0
                return [p,pstar,info,it]
            p0 = p1

        pstar = p1
        info = 1

        return [p,pstar,info,it]


f = lambda x: np.exp(x**2 + 7*x - 30) - 1
fp = lambda x: (2*x + 7) * np.exp(x**2 + 7*x - 30)

x0 = 4.5
tol = 10**-8
Nmax = 100

print(main().newton(f, fp, x0, tol, Nmax))
