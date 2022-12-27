import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns

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

df=df[(df.SFR_0>=10**(-3))]

df['Mass']=0.6*df['K']
#df['Mass']=df['MHI']
df=df[ ['log_SFR_Ha', 'log_SFR_FUV', 'log_SFR_0', 'log_K', 'log_MHI', 'SFR_Ha', 'SFR_FUV','SFR_0', 'K', 'MHI', 'Mass']]
print(df)

df['av_SFR']=df['Mass']*1.3/(12.5*10**9)
df['ratio']=df['av_SFR']/df['SFR_0']
df['log_ratio']=np.log10(df['ratio'])

print(df)
print(df[ ["SFR_0","av_SFR"]].describe(include="all"))

df.plot(kind='scatter', y='av_SFR', x='SFR_0')
plt.xscale('log')
plt.yscale('log')
plt.savefig('graphs/av_SFR-SFR_0')
df.to_csv("out", sep="\t", columns=['SFR_0','av_SFR','Mass','ratio'])

sns.lmplot(x='K',y='MHI',data=df,fit_reg=True).savefig("graphs/K_M.png")

g=sns.lmplot(x='SFR_0',y='av_SFR',data=df,fit_reg=True)
g.savefig("graphs/AAA.png")

print(df[ ["ratio","log_ratio"]].describe(include="all"))

#we can choose the number of bins acording to the Square-root choice (https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width)
k=int(np.ceil(np.sqrt(df.shape[0])))

plt.hist('log_ratio',bins='fd')
plt.xlabel('ratio')
plt.ylabel('# of event')

plt.savefig('graphs/histogram.png')

for i in df.index:
    def sfrx(z):
        A = z[1]
        x = z[0]

        tsf=12.5*10**9

        #a=np.exp(np.log(A))

        sfr=df.loc[i]['SFR_0']
        asfr=df.loc[i]['av_SFR']
        ratio=df.loc[i]['ratio']

        f=np.zeros(2)

       # f[0]=asfr-A*(1-(1+x)*np.exp(-x))/tsf
        f[0]=ratio-(np.exp(np.exp(np.log(x)))-np.exp(np.log(x))-1)/np.exp(np.log(x))**2
        f[1]=sfr-A*x**2*np.exp(-x)/tsf
        return f

    #for i in df.index:
    z = fsolve(sfrx,[3.0,4.0])
    df.at[i,'A_del']=(z[1])
    df.at[i,'x']=(z[0])


print(df)
print(df["x"].describe(), "\n")
print(df["A_del"].describe())

df.plot(kind='scatter', x='x', y='A_del')
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/x-A")
