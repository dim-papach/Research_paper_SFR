import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

df = pd.read_csv('Karachentsev_updated.txt', delimiter= '\s+', header=None)
df.columns=["log_SFR_Ha","log_SFR_FUV","log_K", "log_MHI"]

print(df)

df = df.replace(99999,np.nan)
print(df)

df['SFR_Ha']=10**df['log_SFR_Ha']

df['SFR_FUV']=10**df['log_SFR_FUV']

df['K']=10**df['log_K']

df['MHI']=10**df['log_MHI']

print(df)

df['SFR_0']=df[ ['SFR_Ha','SFR_FUV']].mean(axis=1, skipna=True)
df['log_SFR_0']= np.log10(df['SFR_0'])
print(df)

df=df[(df.SFR_0>=10**(-3))]

df['Mass']=0.6*df['K']
df=df[ ['log_SFR_Ha', 'log_SFR_FUV', 'log_SFR_0', 'log_K', 'log_MHI', 'SFR_Ha', 'SFR_FUV','SFR_0', 'K', 'MHI', 'Mass']]
print(df)

df['av_SFR']=df['Mass']*1.3/(12.5*10**9)
df['ratio']=df['av_SFR']/df['SFR_0']
df['log_ratio']=np.log10(df['ratio'])
print(df)

def func(zGuess,*Params):
    A,t = zGuess
    avS,S = Params
    x=12.5*10**9/t

    eq_1 = S-A*x*np.exp(-x)
    eq_2 = avS-A*(1-(1+x)*np.exp(-x))/12.5*10**(-9)
    return eq_1,eq_2


zGuess = np.array([2.6,0.92])

df['output_a'],df['output_b'] = zip(*df.apply(lambda df: fsolve(func,zGuess,args=(df['av_SFR'],df['SFR_0']))))
