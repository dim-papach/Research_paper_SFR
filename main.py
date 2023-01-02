import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Karachentsev_updated.txt', delimiter= '\s+', header=None)
#name the columns
df.columns=["log_SFR_Ha","log_SFR_FUV","log_K", "log_MHI"]


df = df.replace(99999,np.nan)

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

df['av_SFR']=df['Mass']*1.3/(12.5*10**9)
df['ratio']=df['av_SFR']/df['SFR_0']
df['log_ratio']=np.log10(df['ratio'])

print(df)
print(df[ ["SFR_0","av_SFR",'MHI','Mass']].describe(include="all"))

df.plot(kind='scatter',y='av_SFR', x='SFR_0')
plt.xscale('log')
plt.yscale('log')
plt.savefig('graphs/av_SFR-SFR_0')
df.to_csv("out", sep="\t", columns=['SFR_0','av_SFR','Mass','ratio'])

#sns.lmplot(x='K',y='MHI',data=df,fit_reg=True).savefig("graphs/K_M.png")

g=sns.lmplot(x='SFR_0',y='av_SFR',data=df,fit_reg=True)
plt.xscale('log')
plt.yscale('log')

g.savefig("graphs/AAA.png")


t=sns.lmplot(x='log_MHI',y='log_K',data=df,fit_reg=True)
#df.plot(kind='scatter', y='K', x='MHI')
#plt.xscale('log')
#plt.yscale('log')
#plt.savefig('graphs/K_M')

t.savefig("graphs/bAA.png")

print(df[ ["ratio","log_ratio"]].describe(include="all"))

#we can choose the number of bins acording to the Square-root choice (https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width)
k=int(np.ceil(np.sqrt(df.shape[0])))

df.hist(column='log_ratio', bins=k)

plt.xlabel('log_ratio')
plt.ylabel('# of event')


plt.savefig('graphs/histogram_ratio.png')

for i in df.index:
    def sfrx(z):
        A = z[1]
        x = z[0]

        tsf=12.5e09

        #a=np.exp(np.log(A))

        sfr=df.loc[i]['SFR_0']
        asfr=df.loc[i]['av_SFR']
        ratio=df.loc[i]['ratio']

        f=np.zeros(2)

        f[0]=asfr-A*(1-(1+np.exp(np.log(x)))*np.exp(-x))/tsf
        #f[1]=ratio-(np.exp(np.exp(np.log(x)))-np.exp(np.log(x))-1)/np.exp(np.log(x))**2
        f[1]=sfr-A*x**2*np.exp(-x)/tsf
        return f

    #for i in df.index:
    z = fsolve(sfrx,[3.0,4.0])
    df.at[i,'A_del']=(z[1])
    df.at[i,'x']=(z[0])


df['tau']=12.5*10**9/df['x']

print(df)
print(df[ ["x", 'tau', 'A_del']].describe(include='all' ), "\n")

df.plot(kind='scatter', x='x', y='A_del')
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/x-A")

df.plot(kind='scatter', x='tau', y='A_del')
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/T-A")

#we can choose the number of bins acording to the Square-root choice (https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width)
k=int(np.ceil(np.sqrt(df.shape[0])))

df.hist(column='x', bins=k)

plt.xlabel('x_1')
plt.ylabel('# of event')


plt.savefig('graphs/histogram_x.png')

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

        #f[0]=asfr-A*(1-(1+np.exp(np.log(x)))*np.exp(-np.exp(np.log(x))))/tsf
        f[0]=ratio-(np.exp(x)-np.exp(np.log(x))-1)/np.exp(np.log(x))**2
        f[1]=sfr-A*x**2*np.exp(-x)/tsf
        return f

    #for i in df.index:
    z = fsolve(sfrx,[3.0,4.0])
    df.at[i,'A_del']=(z[1])
    df.at[i,'x']=(z[0])


df['tau']=12.5*10**9/df['x']

print(df)
print(df[ ["x", 'tau', 'A_del']].describe(include='all' ), "\n")

df.plot(kind='scatter', x='x', y='A_del')
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/x-A_2")

df.plot(kind='scatter', x='tau', y='A_del')
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/T-A_2")

#we can choose the number of bins acording to the Square-root choice (https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width)
k=int(np.ceil(np.sqrt(df.shape[0])))

df.hist(column='x', bins=k)

plt.xlabel('x2')
plt.ylabel('# of event')


plt.savefig('graphs/histogram_x2.png')

for i in df.index:
    def sfrx(z):
        x = z

        tsf=12.5*10**9


        ratio=df.loc[i]['ratio']


        #f=ratio-(np.exp(x)-x-1)/x**2
        f=ratio-(np.exp(x)-np.exp(np.log(x))-1)/x**2
        return f

    #for i in df.index:
    z = fsolve(sfrx,3.0)
    df.at[i,'x']=(z)


df['tau']=12.5*10**9/df['x']

df['A_del']=df['SFR_0']*df['tau']*np.exp(df['x'])/df['x']

print(df)
print(df[ ["x", 'tau', 'A_del']].describe(include='all' ), "\n")

df.plot(kind='scatter', x='x', y='A_del')
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/x-A_3")

df.plot(kind='scatter', x='tau', y='A_del')
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/T-A_3")

#we can choose the number of bins acording to the Square-root choice (https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width)
k=int(np.ceil(np.sqrt(df.shape[0])))

df.hist(column='x', bins=k)

plt.xlabel('x3')
plt.ylabel('# of event')


plt.savefig('graphs/histogram_x3.png')

for i in df.index:
    def tsfs(z):
        tsf = z

        tau=df.loc[i]['tau']
        ratio=df.loc[i]['ratio']
        SFR=df.loc[i]['SFR_0']
        A=df.loc[i]['A_del']
        x=tsf/tau
        asfr=df.loc[i]['av_SFR']

        f=SFR-A*x*np.exp(-x)/tau

        #f=asfr-A*(1-(1+np.exp(np.log(x)))*np.exp(-x))/tsf
        #f=ratio-(np.exp(x)-x-1)/x**2
        return f

    #for i in df.index:
    z = fsolve(tsfs,3.0)
    df.at[i,'tsf']=(z)

df['tsf1']=df['A_del']*(1-(1+df['x'])*np.exp(-df['x']))/df['av_SFR']

print(df[ ['tsf','tsf1']].describe(include='all'))

df.plot(kind='scatter', x='Mass', y='tsf')
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/M-tsf")

df.plot(kind='scatter', x='Mass', y='tsf1')
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/M-tsf1")
