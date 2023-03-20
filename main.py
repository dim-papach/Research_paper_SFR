import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf


def simple_regplot(data ,
    X, Y, n_std=2, n_pts=100, ax=None, scatter_kws=None, line_kws=None, ci_kws=None):
    df=data
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
    ax.legend(loc = "best")
    if scatter_kws == {}:
        plt.savefig("graphs/"+X+"-"+Y)
        plt.show()
    else:
        col = scatter_kws["c"].name
        PCM=ax.get_children()[2] #get the mappable, the 1st and the 2nd are the x and y axes
        plt.colorbar(PCM, ax=ax).set_label(col)
        plt.savefig("graphs/"+X+"-"+Y+"color_"+col)
        plt.show()
    return fit_results

# Calculate the bin width using the Freedman-Diaconis rule
def fd_bins(x):
    iqr = df[x].quantile(0.75) - df[x].quantile(0.25)
    bin_width = 2 * iqr / df[x].count()**(1/3)

    # Calculate the number of bins using the bin width
    return int((df[x].max() - df[x].min()) / bin_width)

df=pd.read_csv("Karachentsev_list.csv")

df["TType"]=df["TType"].astype('category')
df["Tdw1"]=df["Tdw1"].astype('category')
df["Tdw2"]=df["Tdw2"].astype('category')

df['SFR_Ha']=10**df['log_SFR_Ha']

df['SFR_FUV']=10**df['log_SFR_FUV']

df['K']=10**df['logKLum']

df['MHI']=10**df['logMHI']

df["color"]=df['Bmag']-df['Kmag']

df['SFR_0']=df[ ['SFR_Ha','SFR_FUV']].mean(axis=1, skipna=True)
df['log_SFR_0']= np.log10(df['SFR_0'])

df=df[(df.SFR_0>=10**(-3))]

df['StellarMass']=0.6*df['K']
df["logStellarMass"]=np.log10(df['StellarMass'])

df['av_SFR']=df['StellarMass']*1.3/(12.5*10**9)
df['log_av_SFR']=np.log10(df['av_SFR'])

df['ratio']=df['av_SFR']/df['SFR_0']
df['log_ratio']=np.log10(df['ratio'])

data["log_ratio"]=df["log_ratio"]

df["Mg"]=1.33*df["MHI"]
df["logMg"]=np.log10(df["Mg"])

df["Mt"]=df["Mg"]+df["StellarMass"]
df['logMt']=np.log10(df['Mt'])

df["Mass_ratio"]=df["StellarMass"]/df["Mg"]
df["log_Mass_ratio"]=np.log10(df["Mass_ratio"])

print(df[ ["SFR_0","av_SFR","ratio","log_ratio",'Mt','MHI','StellarMass','Mg']].describe(include="all"))

df.plot(kind='scatter',y='av_SFR', x='SFR_0', grid="True")
plt.xscale('log')
plt.yscale('log')
plt.savefig('graphs/av_SFR-SFR_0')
plt.show()
simple_regplot('log_SFR_0','log_av_SFR')

da=df["log_ratio"]
print(da.describe())
# Calculate the mean and standard deviation of the da
mean = da.mean()
std = da.std()

# Define the lower and upper bounds for the 2-sigma range
lower = mean - 2*std
upper = mean + 2*std

# Filter out the values outside of the 2-sigma range
df = df[(da >= lower) & (da <= upper)]
print(da.describe())
iqr = data['log_ratio'].quantile(0.75) - data['log_ratio'].quantile(0.25)
bin_width = 2 * iqr / data['log_ratio'].count()**(1/3)

# Calculate the number of bins using the bin width
binss= int((data['log_ratio'].max() - data['log_ratio'].min()) / bin_width)

binsss= int((da.max() - da.min()) / bin_width)

data["log_ratio"].hist(bins=binss,edgecolor= "blue")
da.hist(bins=binsss,alpha=0.5,edgecolor='red')
plt.show()


df = df[(df["log_ratio"] >= lower) & (df["log_ratio"] <= upper)]
print(df.count() , df['ratio'].describe())

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

df.hist(column='x', bins=fd_bins('x'))

plt.xlabel('x')
plt.ylabel('# of event')

plt.savefig('graphs/histogram_x3.png')

plt.show()

tsf=12.5*10**9

df['tau']=tsf/df['x']
df["log_tau"]=np.log10(df["tau"])

df['A']=df['SFR_0']*df['tau']*np.exp(df['x'])/df['x']

df["a"]=df["av_SFR"]*tsf/(1-(1+df["x"])*np.exp(-df['x']))

df["A_del"]=df[ ["a","A"]].mean(axis=1, skipna=True)

print(df[ ["x", 'tau','A', 'A_del','StellarMass','a']].describe(include='all' ), "\n")

print(df[ ["x", 'tau','A_del']].describe(include='all' ), "\n")

df.plot(kind='scatter', x='x', y='A_del',c= "logMt")
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/x-A_3")

df.plot(kind='scatter', x='tau', y="A_del", c= "logMt")
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/T-A_3")
plt.show()

dp=df[["Name","TType","SFR_0","logMt","StellarMass","A_del","tau","x","color","Mass_ratio","log_Mass_ratio"]].copy()
dp['log_tau'] = np.log10(dp.tau)
tau=3.5*10**9
dp["z"]=1.3*dp["StellarMass"]/tau


for i in df.index:
    def sfrx(z):
        x = z

        cons=dp.loc[i]['z']
        SFR=dp.loc[i]['SFR_0']


        #f=ratio-(np.exp(x)-np.abs(x)-1)/x**2
        f=cons/SFR-(np.exp(x)-np.exp(np.log(x))-1)/x
        return f

    #for i in df.index:
    z = fsolve(sfrx,3.0)
    dp.at[i,'x_i']=(z)
dp["tsf"]=dp['x_i']*tau
dp["log_tsf"]=np.log10(dp.tsf)
dp["av_SFR"]=dp.z/dp.x_i
dp['ratio']=dp.av_SFR/dp.SFR_0
dp['A']=tau*dp['SFR_0']*np.exp(dp.x_i)/dp.x_i

print(dp[['A','tsf','x_i']].describe())

dp.plot(kind='scatter', x='x_i', y='A',c='logMt')
plt.xscale('log')
plt.yscale('log')
plt.savefig("graphs/x-A_tau")
plt.show()

dp['af']=dp[["A_del","A"]].mean(axis=1, skipna=True)
dp.plot(kind='scatter', x='x_i', y='af',c='logMt')
plt.xscale('log')
plt.yscale('log')
plt.show()

dp['log_A_del']=np.log10(dp.A_del)
dp['log_A']=np.log10(dp.A)
print(dp[['A','A_del','af']].describe())

print(dp[['x','x_i']].describe())

simple_regplot(dp,'log_A_del','log_A',scatter_kws={"c":dp["logMt"]})

dp.plot(kind='scatter', x='log_tau', y='tsf',c='logMt')
plt.xscale('log')
plt.yscale('log')
plt.show()

x=simple_regplot(dp,'x','x_i',scatter_kws={"c":dp["logMt"]})
x=simple_regplot(dp,'x','x_i',scatter_kws={"c":dp["TType"]})
x=simple_regplot(dp,'x','x_i',scatter_kws={"c":dp["color"]})

simple_regplot(dp,'logMt','x_i',scatter_kws={"c":dp["color"]})

print(x.summary())

temp_dataf=dp[['Name','tsf']]
df = pd.merge(df, temp_dataf , on = 'Name', how = 'outer')
print(df['tsf'])

df["tau_g"]=df["Mg"]/df["SFR_0"]
df["log_tau_g"]=np.log10(df["tau_g"])
print(df["tau_g"].describe())

df.plot(kind="scatter",x="Mg",y="tau_g", c = 'logStellarMass')
plt.xscale('log')
plt.yscale('log')
plt.show()

df.plot(kind="scatter",x="x",y="tau_g", c = 'logStellarMass')
plt.xscale('log')
plt.yscale('log')
plt.show()

df.plot(kind="scatter",x="tau",y="tau_g", c = 'logStellarMass')
plt.xscale('log')
plt.yscale('log')
plt.show()

df.plot(kind="scatter",x="tsf",y="tau_g", c = 'logStellarMass')
plt.xscale('log')
plt.yscale('log')
plt.show()

typ=pd.read_csv("Karachentsev_list_flags.csv")

typ["TType"]=typ["TType"].astype('category')
typ["Tdw1"]=typ["Tdw1"].astype('category')
typ["Tdw2"]=typ["Tdw2"].astype('category')
print(typ.count())
typ['TType'].value_counts(sort=False).plot(kind='bar',logy=True,grid = 'True')
plt.xlabel("Morphology")
plt.ylabel("Number of Galaxies")
plt.savefig("graphs/hist-Type")
plt.show()

typ['Tdw1'].value_counts(sort=False).plot(kind='bar', logy=True,grid = 'True')
plt.xlabel("Dwarf galaxy morphology")
plt.ylabel("Number of Galaxies")
plt.savefig("graphs/hist-Tdw1")
plt.show()

typ['Tdw2'].value_counts(sort=False).plot(kind='bar', logy=True,grid = 'True')
plt.xlabel("Dwarf galaxy surface brightness morphology")
plt.ylabel("Number of Galaxies")
plt.savefig("graphs/hist-Tdw2")
plt.show()

#we can choose the number of bins acording to the Square-root choice (https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width)
k=int(np.ceil(np.sqrt(df.shape[0])))

df.hist(column='x', bins=k)

plt.xlabel('x3')
plt.ylabel('# of event')


plt.savefig('graphs/histogram_x3.png')

plt.show()

mhi_mass=simple_regplot(df,"logStellarMass","logMHI",scatter_kws={"c": df["log_tau_g"]})
print(mhi_mass.summary())

simple_regplot(df,"logStellarMass","logMg",scatter_kws={"c": df["color"]})

mg_mass=simple_regplot(df,"logMg","logStellarMass")
print(mg_mass.summary())

mass_m26=simple_regplot(df,"logStellarMass","logM26",scatter_kws={"c": df["log_tau_g"]})
print(mass_m26.summary())

mg_m26=simple_regplot(df,"logMg","logM26")
print(mg_m26.summary())

mhi_m26=simple_regplot(df,"logMHI","logM26")
print(mhi_m26.summary())

mass_mt=simple_regplot(df,"logStellarMass","logMt",scatter_kws={"c": df["log_tau_g"]})
mass_mt=simple_regplot(df,"logStellarMass","logMt",scatter_kws={"c": df["log_SFR_0"]})
mass_mt=simple_regplot(df,"logStellarMass","logMt",scatter_kws={"c": df["logMg"]})
mass_mt=simple_regplot(df,"logStellarMass","logMt",scatter_kws={"c": df["log_Mass_ratio"]})
mass_mt=simple_regplot(df,"logStellarMass","logMt",scatter_kws={"c": df["log_Mass_ratio"]})
mass_mt=simple_regplot(df,"logStellarMass","logMt",scatter_kws={"c": df["color"]})

print(mass_mt.summary())

mg_mt=simple_regplot(df,"logMg","logMt",scatter_kws={"c":df['log_SFR_0']})
mg_mt=simple_regplot(df,"logMg","logMt",scatter_kws={"c":df['log_tau_g']})
mg_mt=simple_regplot(df,"logMg","logMt",scatter_kws={"c":df['logStellarMass']})
print(mg_mt.summary())

mhi_mt=simple_regplot(df,"logMHI","logMt")
print(mhi_mt.summary())

m26_mt=simple_regplot(df,"logM26","logMt")
print(m26_mt.summary())

plt.scatter(10**df["logM26"], df["StellarMass"],label= "M26")
plt.scatter(df["MHI"], df["StellarMass"], label= "MHI")
plt.scatter(df["Mt"], df["StellarMass"], label= "MHI")
plt.legend(loc= "upper left")
plt.xlabel("StellarMass")
plt.xscale("log")
plt.yscale("log")
plt.title("M26, MHI=f(StellarMass), log")
plt.savefig('graphs/M-MHI-M26')
plt.show()

simple_regplot(dp,"log_tsf","log_Mass_ratio",scatter_kws={"c": df["color"]})
