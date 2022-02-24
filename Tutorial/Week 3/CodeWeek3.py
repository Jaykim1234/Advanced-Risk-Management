import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm

"""
Data reading and transformation
"""
st = dt.datetime(2000, 12, 31)
en = dt.datetime(2010, 12, 31)
data = web.DataReader('^GSPC', 'yahoo', start=st, end=en)
S = data['Adj Close']
R = 100 * np.log(1+ S.pct_change().dropna())
R.name = 'R'

"""
Fit GJR-GARCH model, keep shocks z, and standardize Returns
"""
from arch import arch_model
am = arch_model(R,p=1,o=1,q=1)
res = am.fit(disp='off')
sigma = res.conditional_volatility
z = res.std_resid
Return = (R - R.mean())/R.std()
z = (z-z.mean())/z.std()
print("Skewness and excess kurtosis of return R: SK = {0:.3f}, EK = {1:.3f}".format(Return.skew(), Return.kurtosis()))
print("Skewness and excess kurtosis of shocks z: SK = {0:.3f}, EK = {1:.3f}".format(z.skew(), z.kurtosis()))

""" 
Histograms and normal densities
"""
plt.figure(figsize=(6, 4))
plt.subplot(211)
sns.distplot(Return,hist=True,kde=False,fit=stats.norm)
plt.legend(['N(0,1)','Return'])
plt.subplot(212)
sns.distplot(z,hist=True,kde=False,fit=stats.norm)
plt.legend(['N(0,1)','z'])
#plt.savefig('histograms.pdf')
#plt.close()

""" 
QQ-plots
"""
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(1, 2, 1)
sm.qqplot(Return, stats.norm, line='45', ax=ax)
plt.legend(['Return','N(0,1)'])
ax = fig.add_subplot(1, 2, 2)
sm.qqplot(z, stats.norm, line='45', ax=ax)
plt.legend(['z','N(0,1)'])
#plt.savefig('qqplots.pdf')
#plt.close()

"""
Filtered historical simulation
"""
m = 250
VaR_FHS = pd.Series(0*R, name='VaR FHS')
for t in range(m,len(R)):
    VaR_FHS[t] = -sigma[t]*np.percentile(z[t-m:t],1)
VaR_FHS = VaR_FHS['1/7/2002':'12/31/2010']
VaR_FHS.plot()
VaR_RM = 2.33*sigma['1/7/2002':'12/31/2010']
VaR_RM.plot()
plt.margins(x=0)
plt.legend(["VaR FHS","VaR GARCH-N(0,1)"])
#plt.savefig('FHS.pdf')
#plt.close()

"""
Standardized Student's t(d) densities
"""
plt.figure(figsize=(6, 4))
x = np.arange(-6,4,0.01)
d = 3.0
s = np.sqrt((d-2)/d)
f1 = stats.t.pdf(x,d,0,s)
d = 5.0
s = np.sqrt((d-2)/d)
f2 = stats.t.pdf(x,d,0,s)
d = 8.0
s = np.sqrt((d-2)/d)
f3 = stats.t.pdf(x,d,0,s)
f4 = stats.norm.pdf(x)
plt.plot(x[200:],f1[200:],x[200:],f2[200:],x[200:],f3[200:],x[200:],f4[200:])
plt.legend(['$d=3$','$d=5$','$d=8$','$d=\infty$'])
#plt.savefig('tdensities.pdf')
#plt.close()
plt.plot(x[:350],f1[:350],x[:350],f2[:350],x[:350],f3[:350],x[:350],f4[:350])
plt.legend(['$d=3$','$d=5$','$d=8$','$d=\infty$'])
#plt.savefig('tdenstails.pdf')
#plt.close()

"""
ML estimation of GARCH-t(d) model
"""
amt = arch_model(R,p=1,o=1,q=1,dist='StudentsT')
# amt = arch_model(R,p=1,o=1,q=1,dist='SkewStudent')
rest = amt.fit()
print(rest.summary())
z = rest.std_resid
z = (z-z.mean())/z.std()
d = rest.params[5]
s = np.sqrt((d-2)/d)
fig, ax = plt.subplots(figsize=(4, 4))
sm.qqplot(z, stats.t, distargs=[d], loc=0, scale=s, fit=False, line='45', ax=ax)
plt.legend(['z','t(d)'])
#plt.savefig('qqplotst.pdf')
#plt.close()

"""
EVT estimation, from GARCH shocks
"""
z = res.std_resid
z = z.sort_values()
Tu = 50
u = -z[Tu]
x = -z[0:Tu]
xi = np.log(x/u).mean()

fig, ax = plt.subplots(figsize=(4, 4))
sm.qqplot((x/u), stats.pareto, distargs=[1/xi], line='45', ax=ax)
plt.show()

"""
VaR and ES from t(d) and EVT
"""
from scipy.special import gamma
tp = stats.t.ppf(0.99,d)
C = gamma((d+1)/2)/(gamma(d/2)*np.sqrt(np.pi*(d-2)))
VaR_t = (tp*np.sqrt((d-2)/d))
ES_t = (C/0.01)*((d-2)/(d-1))*(1+(tp**2)/d)**((1-d)/2)
hit_t = 1*(z<-VaR_t)
print("VaR from t(d) distribution: {0:.4f}".format(VaR_t))
print("ES from t(d) distribution:  {0:.4f}".format(ES_t))
print("Percentage of exceedances:  {0:.4f}".format(hit_t.mean()*100))

T = len(z)
VaR_EVT = u*(0.01*T/Tu)**(-xi)
ES_EVT = VaR_EVT/(1-xi)
hit_EVT = 1*(z<-VaR_EVT)
print("VaR from EVT distribution:  {0:.4f}".format(VaR_EVT))
print("ES from EVT distribution:   {0:.4f}".format(ES_EVT))
print("Percentage of exceedances:  {0:.4f}".format(hit_EVT.mean()*100))
