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

for i in df.index:
    def sfrx(z):
        A = z[1]
        x = z[0]
        tsf=12.5*10**9
        a=np.exp(np.log(A))
        sfr=df.loc[i]['SFR_0']
        asfr=df.loc[i]['av_SFR']
        ratio=df.loc[i]['ratio']
        f=np.zeros(2)

        f[0]=asfr-a*(1-(1+x))
        #f[0]=ratio-(np.exp(x)-x-1)/x**2
        f[1]=sfr-a*x**2*np.exp(-x)/tsf
        return f

    #for i in df.index:
    z = fsolve(sfrx,[3.0,4.0])
    df.at[i,'A_del']=np.exp(z[1])
    df.at[i,'x']=np.exp(z[0])
    

print(df)

plt.plot(df['x'], df['A_del'])
plt.show
