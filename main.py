import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def simple_regplot(
    X, Y, n_std=2, n_pts=100, ax=None, scatter_kws=None, line_kws=None, ci_kws=None):
    x=df[X]
    y=df[Y]
    """ Draw a regression line with error interval. """
    ax = plt.gca() if ax is None else ax

    # calculate best-fit line and interval
    x_fit = sm.add_constant(x)
    fit_results = sm.OLS(y, x_fit,missing='drop').fit()

    xconst=f'{fit_results.params[1]:.3f}'
    const=f'{fit_results.params[0]:.3f}'
    eval_x = sm.add_constant(np.linspace(np.min(x), np.max(x), n_pts))
    pred = fit_results.get_prediction(eval_x)

    # draw the fit line and error interval
    ci_kws = {} if ci_kws is None else ci_kws
    ax.fill_between(
        eval_x[:, 1],
        pred.predicted_mean - n_std * pred.se_mean,
        pred.predicted_mean + n_std * pred.se_mean,
        facecolor ='red',
        alpha=0.5,
        **ci_kws,
    )
    line_kws = {} if line_kws is None else line_kws
    h = ax.plot(eval_x[:, 1], pred.predicted_mean, **line_kws,color='red',linestyle='dashed',label= Y+"="+xconst+X+"+"+const)

    # draw the scatterplot
    scatter_kws = {} if scatter_kws is None else scatter_kws
    ax.scatter(x, y, **scatter_kws,label="Data")

    ax.set_title(Y+ "=f("+X+ ")")
    ax.grid()
    ax.set_xlabel(X)
    ax.set_ylabel(Y)
    ax.legend(loc= "best")
    plt.savefig("graphs/"+X+"-"+Y)
    plt.show()
    return fit_results

df=pd.read_csv("Karachentsev_list.csv")

df["TType"]=df["TType"].astype('category')
df["Tdw1"]=df["Tdw1"].astype('category')
df["Tdw2"]=df["Tdw2"].astype('category')

df['SFR_Ha']=10**df['log_SFR_Ha']

df['SFR_FUV']=10**df['log_SFR_FUV']

df['K']=10**df['logKLum']

df['MHI']=10**df['logMHI']

df['SFR_0']=df[ ['SFR_Ha','SFR_FUV']].mean(axis=1, skipna=True)
df['log_SFR_0']= np.log10(df['SFR_0'])

df=df[(df.SFR_0>=10**(-3))]

df['Mass']=0.6*df['K']
df["logMass"]=np.log10(df['Mass'])

df['av_SFR']=df['Mass']*1.3/(12.5*10**9)
df['ratio']=df['av_SFR']/df['SFR_0']
df['log_ratio']=np.log10(df['ratio'])

df["Mg"]=1.33*df["MHI"]
df["logMg"]=np.log10(df["Mg"])

df["Mt"]=df["Mg"]+df["Mass"]
df['logMt']=np.log10(df['Mt'])

print(df[ ["SFR_0","av_SFR","ratio","log_ratio",'Mt','MHI','Mass','Mg']].describe(include="all"))

df.plot(kind='scatter',y='av_SFR', x='SFR_0')
plt.xscale('log')
plt.yscale('log')
plt.savefig('graphs/av_SFR-SFR_0')
df.to_csv("out", sep="\t", columns=['SFR_0','av_SFR','Mass','ratio'])
plt.show()



for i in df.index:
    def sfrx(z):
        x = z

        ratio=df.loc[i]['ratio']

        #f=ratio-(np.exp(x)-np.abs(x)-1)/x**2
        f=ratio-(np.exp(x)-np.exp(np.log(x))-1)/x**2
        return f

    #for i in df.index:
    z = fsolve(sfrx,3.0)
    df.at[i,'x']=(z)

tsf=np.random.normal(12.5*10**9,1,1)
df['tau']=tsf/df['x']

df['A_del']=df['SFR_0']*df['tau']*np.exp(df['x'])/df['x']

print(df[ ["x", 'tau', 'A_del','Mass']].describe(include='all' ), "\n")

df.plot(kind='scatter', x='x', y='A_del',c= "logMass")
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/x-A_3")

df.plot(kind='scatter', x='tau', y="A_del", c= "logMass")
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/T-A_3")

df.plot(kind='scatter', x='tau', y='A_del', c= "TType")
plt.xscale('log')
plt.yscale('log')
plt.show()

#we can choose the number of bins acording to the Square-root choice (https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width)
k=int(np.ceil(np.sqrt(df.shape[0])))

df.hist(column='x', bins=k)

plt.xlabel('x3')
plt.ylabel('# of event')


plt.savefig('graphs/histogram_x3.png')

plt.show()

for i in df.index:
    def sfrx(z):
        x = z

        ratio=df.loc[i]['ratio']

        #f=ratio-(np.exp(x)-np.abs(x)-1)/x**2
        f=ratio-(np.exp(x)-np.exp(np.log(x))-1)/x**2
        return f

    #for i in df.index:
    z = fsolve(sfrx,3.0)
    df.at[i,'x_i']=(z)

tau=np.random.normal(4*10**9,1,1)
df['tsf']=tau*df['x_i']

df['A_del_i']=df['SFR_0']*tau*np.exp(df['x'])/df['x']

print(df[ ["x_i", 'tsf', 'A_del_i','A_del','Mass']].describe(include='all' ), "\n")

df.plot(kind='scatter', x='x_i', y='A_del_i',c= "logMass")
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/x-A_i")

df.plot(kind='scatter', x='tsf', y="A_del_i", c= "logMass")
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/T-A_i")

df.plot(kind='scatter', x='tsf', y='A_del_i', c= "TType")
plt.xscale('log')
plt.yscale('log')
plt.show()

df['TType'].value_counts(sort=False).plot(kind='bar')
plt.savefig("graphs/hist-Type")
plt.show()
df['Tdw1'].value_counts(sort=False).plot(kind='bar', logy=True)
plt.savefig("graphs/hist-Tdw1")
plt.show()
df['Tdw2'].value_counts(sort=False).plot(kind='bar')
plt.savefig("graphs/hist-Tdw2")
plt.show()

#we can choose the number of bins acording to the Square-root choice (https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width)
k=int(np.ceil(np.sqrt(df.shape[0])))

df.hist(column='x', bins=k)

plt.xlabel('x3')
plt.ylabel('# of event')


plt.savefig('graphs/histogram_x3.png')

plt.show()

mhi_mass=simple_regplot("logMass","logMHI")
print(mhi_mass.summary())

mg_mass=simple_regplot("logMg","logMass")
print(mg_mass.summary())

mass_m26=simple_regplot("logMass","logM26")
print(mass_m26.summary())

mg_m26=simple_regplot("logMg","logM26")
print(mg_m26.summary())

mhi_m26=simple_regplot("logMHI","logM26")
print(mhi_m26.summary())

mass_mt=simple_regplot("logMass","logMt")
print(mass_mt.summary())

mg_mt=simple_regplot("logMg","logMt")
print(mg_mt.summary())

mhi_mt=simple_regplot("logMHI","logMt")
print(mhi_mt.summary())

m26_mt=simple_regplot("logM26","logMt")
print(m26_mt.summary())

plt.scatter(10**df["logM26"], df["Mass"],label= "M26")
plt.scatter(df["MHI"], df["Mass"], label= "MHI")
plt.scatter(df["Mt"], df["Mass"], label= "MHI")
plt.legend(loc= "upper left")
plt.xlabel("Mass")
plt.xscale("log")
plt.yscale("log")
plt.title("M26, MHI=f(Mass), log")
plt.savefig('graphs/M-MHI-M26')
plt.show()
