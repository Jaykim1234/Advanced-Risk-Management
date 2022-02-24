import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import seaborn as sns

T = 10000
d = 4.5
x = stats.t.rvs(d,size=T)*np.sqrt((d-2)/d)      # generated T obs of standardized t(d) 
x = np.flipud(np.sort(x))                       # sort in descending order (largest first)
x = pd.Series(x,name='x')                       # make into Pandas series

params = stats.t.fit(x,d,floc=0)        # ML estimation, fixing mean=0
d_ML = params[0]                        # MLE of d

Tu = 500                                # take Tu as 0.05*T, so use 5% largest values of x
u = x[Tu]                               # threshold for EVT
p = (np.arange(1,T+1)-0.5)/T            # vector of p_i values

