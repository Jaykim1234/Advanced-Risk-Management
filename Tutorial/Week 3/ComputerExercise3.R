library("QRM")

T = 10000
d = 4.5
x = rt(T,d)*sqrt((d-2)/d)       # generated T obs of standardized t(d)
x = sort(x,decreasing=TRUE)     # sort in descending order (largest first)

tfit = fit.st(x)                # ML estimation
d_ML = unname(tfit$par.ests[1]) # MLE of d
Tu = 500                        # take Tu as 0.05*T, so use 5% largest values of x
u = x[Tu+1]                     # threshold for EVT
p = ((1:T)-0.5)/Tu              # vector of p_i values