import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

"""
Data reading and transformation
"""
S = pd.read_csv('c:/work/ARM2019/week4/Chapter7Data.csv', parse_dates=True, index_col='Date')
S1 = S['SP500']
S2 = S['Note10y']
R1 = 100*np.log(1 + S1.pct_change().dropna())
R2 = 100*np.log(1 + S2.pct_change().dropna())
R1.name = 'R1'
R2.name = 'R2'

"""
RiskMetrics
"""
R1sq = R1**2
R2sq = R2**2
R1R2 = R1*R2
sigma11 = R1sq.ewm(alpha=0.06).mean()
sigma22 = R2sq.ewm(alpha=0.06).mean()
sigma12 = R1R2.ewm(alpha=0.06).mean()
sigma1 = np.sqrt(sigma11)[20:]
sigma2 = np.sqrt(sigma22)[20:]
rho12 = sigma12/(sigma1*sigma2)[20:]
sigma12 = sigma12[20:] # remove first 20 startup observations

plt.figure(figsize=(8,6))
plt.subplot(221)
sigma1.plot()
plt.legend(['$\sigma_{1,t}$'])
plt.subplot(222)
sigma12.plot()
plt.legend(['$\sigma_{12,t}$'])
plt.subplot(223)
rho12.plot()
plt.legend(['$\\rho_{12,t}$'])
plt.subplot(224)
sigma2.plot()
plt.legend(['$\sigma_{2,t}$'])
plt.savefig('EWMA.pdf')
plt.show()

"""
GARCH
"""
beta = 0.969832**2
alpha = 0.232871**2     # estimates obtained from G@RCH package under Ox
s11 = R1sq.mean()
s22 = R2sq.mean()
s12 = R1R2.mean()

sigma11 = s11*(R1sq>0)
sigma22 = s22*(R1sq>0)
sigma12 = s12*(R1sq>0)
for t in range(1,len(sigma11)):
    sigma11[t] = s11*(1-alpha-beta) + alpha*R1sq[t-1] + beta*sigma11[t-1]
    sigma22[t] = s22*(1-alpha-beta) + alpha*R2sq[t-1] + beta*sigma22[t-1]
    sigma12[t] = s12*(1-alpha-beta) + alpha*R1R2[t-1] + beta*sigma12[t-1]

sigma1 = np.sqrt(sigma11)[20:]
sigma2 = np.sqrt(sigma22)[20:]
rho12 = sigma12/(sigma1*sigma2)[20:]
sigma12 = sigma12[20:] # remove first 20 startup observations

plt.figure(figsize=(8,6))
plt.subplot(221)
sigma1.plot()
plt.legend(['$\sigma_{1,t}$'])
plt.subplot(222)
sigma12.plot()
plt.legend(['$\sigma_{12,t}$'])
plt.subplot(223)
rho12.plot()
plt.legend(['$\\rho_{12,t}$'])
plt.subplot(224)
sigma2.plot()
plt.legend(['$\sigma_{2,t}$'])
plt.savefig('GARCH.pdf')
plt.show()

"""
DCC
"""
from arch import arch_model
am1 = arch_model(R1,p=1,o=1,q=1)
res1 = am1.fit()
sigma1 = res1.conditional_volatility
z1 = R1/sigma1
z1 = (z1-z1.mean())/z1.std()
am2 = arch_model(R2,p=1,o=1,q=1)
res2 = am2.fit()
sigma2 = res2.conditional_volatility
z2 = R2/sigma2
z2 = (z2-z2.mean())/z2.std()
z1sq = z1**2
z2sq = z2**2
z1z2 = z1*z2
r12 = z1z2.mean()
alpha = 0.053930      # estimates obtained from G@RCH package under Ox
beta = 0.919449
q11 = 1.0*(z1sq>0)
q22 = 1.0*(z1sq>0)
q12 = r12*(z1sq>0)
for t in range(1,len(sigma1)):
    q11[t] = (1-alpha-beta) + alpha*z1sq[t-1] + beta*q11[t-1]
    q22[t] = (1-alpha-beta) + alpha*z2sq[t-1] + beta*q22[t-1]
    q12[t] = r12*(1-alpha-beta) + alpha*z1z2[t-1] + beta*q12[t-1]
rho12 = q12/np.sqrt(q11*q22)
sigma12 = sigma1*sigma2*rho12

sigma1 = sigma1[20:]
sigma2 = sigma2[20:]
rho12 = rho12[20:]
sigma12 = sigma12[20:] # remove first 20 startup observations


plt.figure(figsize=(8,6))
plt.subplot(221)
sigma1.plot()
plt.legend(['$\sigma_{1,t}$'])
plt.subplot(222)
sigma12.plot()
plt.legend(['$\sigma_{12,t}$'])
plt.subplot(223)
rho12.plot()
plt.legend(['$\\rho_{12,t}$'])
plt.subplot(224)
sigma2.plot()
plt.legend(['$\sigma_{2,t}$'])
plt.savefig('DCC.pdf')
plt.show()

"""
MC simulation
"""
MC = 10000
K = 22
coefs = res1.params
mu = coefs[0]
omega = coefs[1]
alpha = coefs[2]
gamma = coefs[3]
beta = coefs[4]

RK = np.zeros(MC)
sig0 = sigma1['2005-01-03']
for i in range(0,MC):
    sig = sig0
    for k in range(0,K):
        Rk = mu + sig*np.random.normal(0,1)
        sig = np.sqrt(omega + alpha*(Rk-mu)**2 + gamma*(Rk-mu<0)*(Rk-mu)**2 + beta*sig**2)
        RK[i] += Rk
plt.figure(figsize=(8,6))
sns.distplot(RK,hist=True,kde=False)
plt.legend(['$R_{t+1:t+K}$'])
plt.show()
print("22-day 1% VaR, 31 December 2004:")
print("Monte Carlo simulation:", str.format('{:.3f}', -np.percentile(RK,1)))
print("RiskMetrics:           ", str.format('{:.3f}', -np.sqrt(K)*sig0*stats.norm.ppf(0.01)))
print("Skewness and excess kurtosis: SK = ", str.format('{:.3f}', stats.skew(RK)), ", EK = ", str.format('{:.3f}', stats.kurtosis(RK)))

RK = np.zeros(MC)
sig0 = sigma1['2009-01-02']
for i in range(0,MC):
    sig = sig0
    for k in range(0,K):
        Rk = mu + sig*np.random.normal(0,1)
        sig = np.sqrt(omega + alpha*(Rk-mu)**2 + gamma*(Rk-mu<0)*(Rk-mu)**2 + beta*sig**2)
        RK[i] += Rk
plt.figure(figsize=(8,6))
sns.distplot(RK,hist=True,kde=False)
plt.legend(['$R_{t+1:t+K}$'])
plt.savefig("MC_hist.pdf")
plt.show()
print("22-day 1% VaR, 31 December 2008:")
print("Monte Carlo simulation:", str.format('{:.3f}', -np.percentile(RK,1)))
print("RiskMetrics:           ", str.format('{:.3f}', -np.sqrt(K)*sig0*stats.norm.ppf(0.01)))
print("Skewness and excess kurtosis: SK = ", str.format('{:.3f}', stats.skew(RK)), ", EK = ", str.format('{:.3f}', stats.kurtosis(RK)))


"""
FH simulation
"""
MC = 10000
K = 22

RK = np.zeros(MC)
sig0 = sigma1['2005-01-03']
z = z1['2001-01-02':'2004-12-31']
z = (z - z.mean())/z.std()
m = len(z)
for i in range(0,MC):
    sig = sig0
    for k in range(0,K):
        Rk = mu + sig*z[np.random.randint(0,m)]
        sig = np.sqrt(omega + alpha*(Rk-mu)**2 + gamma*(Rk-mu<0)*(Rk-mu)**2 + beta*sig**2)
        RK[i] += Rk
plt.figure(figsize=(8,6))
sns.distplot(RK,hist=True,kde=False)
plt.legend(['$R_{t+1:t+K}$'])
plt.show()
print("22-day 1% VaR, 31 December 2004:")
print("FHS        :", str.format('{:.3f}', -np.percentile(RK,1)))
print("RiskMetrics:", str.format('{:.3f}', -np.sqrt(K)*sig0*stats.norm.ppf(0.01)))
print("Skewness and excess kurtosis: SK = ", str.format('{:.3f}', stats.skew(RK)), ", EK = ", str.format('{:.3f}', stats.kurtosis(RK)))

RK = np.zeros(MC)
sig0 = sigma1['2009-01-02']
z = z1['2005-01-03':'2008-12-31']
z = (z - z.mean())/z.std()
m = len(z)
for i in range(0,MC):
    sig = sig0
    for k in range(0,K):
        Rk = mu + sig*z[np.random.randint(0,m)]
        sig = np.sqrt(omega + alpha*(Rk-mu)**2 + gamma*(Rk-mu<0)*(Rk-mu)**2 + beta*sig**2)
        RK[i] += Rk
plt.figure(figsize=(8,6))
sns.distplot(RK,hist=True,kde=False)
plt.legend(['$R_{t+1:t+K}$'])
plt.savefig("FHS_hist.pdf")
plt.show()
print("22-day 1% VaR, 31 December 2008:")
print("FHS        :", str.format('{:.3f}', -np.percentile(RK,1)))
print("RiskMetrics:", str.format('{:.3f}', -np.sqrt(K)*sig0*stats.norm.ppf(0.01)))
print("Skewness and excess kurtosis: SK = ", str.format('{:.3f}', stats.skew(RK)), ", EK = ", str.format('{:.3f}', stats.kurtosis(RK)))

