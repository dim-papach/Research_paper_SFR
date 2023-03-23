import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tabulate import tabulate


pd.set_option('display.float_format', lambda x: '%.2E' % x)
######Print images with caption
def caption(fname,caption,name = None):
    if name == None:
        name = caption
    return "#+caption:{} \n#+name: fig:{} \n#+label: fig:{} \n[[./{}]]".format(caption,name,name,fname)

def simple_regplot(data ,
    X, Y, cap = None, name = None, n_std=2, n_pts=100, ax=None, scatter_kws=None, line_kws=None, ci_kws=None):
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
        lab = X+"-"+Y
        fname = "graphs/"+lab
        plt.savefig(fname)
        plt.close()
    else:
        col = scatter_kws["c"].name
        PCM=ax.get_children()[2] #get the mappable, the 1st and the 2nd are the x and y axes
        plt.colorbar(PCM, ax=ax).set_label(col)
        lab = X+"-"+Y+"-color_"+col
        fname = "graphs/"+lab
        plt.savefig(fname)
        plt.close()

    if name == None:
        name = cap

    Cap = "#+caption:{} \n#+name: fig:{} \n#+label: fig:{} \n[[./{}]]".format(cap,name,name,fname)
    return fit_results, Cap , lab

# Calculate the bin width using the Freedman-Diaconis rule
def fd_bins(x):
    iqr = df[x].quantile(0.75) - df[x].quantile(0.25)
    bin_width = 2 * iqr / df[x].count()**(1/3)

    # Calculate the number of bins using the bin width
    return int((df[x].max() - df[x].min()) / bin_width)
#
## Format the table as a string using the tabulate library
def no_col_str(df,string):
    return df.drop(df.filter(regex=string).columns, axis=1)
#######


#### PRINTING THE OLS PARAMETERS AS LATEX
def inline_ols(y,x,ols):
        return "${} = ({:.2E}\pm {:.2E})\cdot {} + ({:.2E}\pm {:.2E})$".format(y, ols[0].params[1], ols[0].bse[1], x, ols[0].params[0], ols[0].bse[0])

def inline_r2(ols):
        return r"$R^2 = {}\%$".format(round(x_comp[0].rsquared*100))

def eq_ols(y,x,ols):
    return (r"\begin{"+"equation"+"}\label{eq:"+"{}".format(ols[2])+"}\n"+
            r"\begin{"+"align"+"}\n"+
            "& {} = ({:.2E}\pm {:.2E})\cdot {} + ({:.2E}\pm {:.2E})\\\ \n".format(y, ols[0].params[1], ols[0].bse[1], x, ols[0].params[0], ols[0].bse[0])+
            r"& \textrm"+"{"+"with correlation "+"}"+" R^2={}\%\n".format(round(ols[0].rsquared*100))+
            r"\end{"+"align"+"}\n"+
            "\end{"+"equation"+"}\n")

flag=pd.read_csv("Karachentsev_list_flags.csv")
colorcol=flag[["Name","Kmag"]].copy()

colorcol=colorcol[colorcol["Kmag"].astype(str).str.contains(">|<|\*")==False]

colorcol["Kmag"] = colorcol["Kmag"].astype('float')

data = pd.read_csv("Karachentsev_list.csv")
data["TType"] = data["TType"].astype('category')
data["Tdw1"] = data["Tdw1"].astype('category')
data["Tdw2"] = data["Tdw2"].astype('category')

df = data.copy()
df = pd.merge(colorcol[["Name", "Kmag"]], df, on = "Name", how = 'right')
df['SFR_Ha']=10**df['log_SFR_Ha']

df['SFR_FUV']=10**df['log_SFR_FUV']

df['K']=10**df['logKLum']

df['MHI']=10**df['logMHI']

df["color"] = df["Bmag"] - df['Kmag']

no_col_str(df,'log').count().to_latex(position = "hc")

##Average SFR
df['SFR_0']=df[ ['SFR_Ha','SFR_FUV']].mean(axis=1, skipna=True)
df['log_SFR_0']= np.log10(df['SFR_0'])
##Clipping
df=df[(df.SFR_0>=10**(-3))]

#Masses
df['StellarMass']=0.6*df['K']
df["logStellarMass"]=np.log10(df['StellarMass'])

df["Mg"]=1.33*df["MHI"]
df["logMg"]=np.log10(df["Mg"])

df["Mt"]=df["Mg"]+df["StellarMass"]
df['logMt']=np.log10(df['Mt'])

df["Mass_ratio"]=df["StellarMass"]/df["Mg"]
df["log_Mass_ratio"]=np.log10(df["Mass_ratio"])

df["logcolor"] = np.log10(df["color"])

typ=pd.read_csv("Karachentsev_list_flags.csv")
typ["TType"]=typ["TType"].astype('category')
typ["Tdw1"]=typ["Tdw1"].astype('category')
typ["Tdw2"]=typ["Tdw2"].astype('category')
print(typ.count())
typ['TType'].value_counts(sort=False).plot(kind='bar',logy=True,grid = 'True')
plt.xlabel("Morphology")
plt.ylabel("Number of Galaxies")
plt.savefig("graphs/hist-Type")
plt.close()

typ['Tdw1'].value_counts(sort=False).plot(kind='bar', logy=True,grid = 'True')
plt.xlabel("Dwarf galaxy morphology")
plt.ylabel("Number of Galaxies")
plt.savefig("graphs/hist-Tdw1")
plt.close()

typ['Tdw2'].value_counts(sort=False).plot(kind='bar', logy=True,grid = 'True')
plt.xlabel("Dwarf galaxy surface brightness morphology")
plt.ylabel("Number of Galaxies")
plt.savefig("graphs/hist-Tdw2")
plt.close()

###Constant tsf
dts=df.copy()
tsf=12.5*10**9
zeta=1.3

dts['av_SFR']=dts['StellarMass']*1.3/(12.5*10**9)
dts['log_av_SFR']=np.log10(dts['av_SFR'])

dts['ratio']=dts['av_SFR']/dts['SFR_0']
dts['log_ratio']=np.log10(dts['ratio'])

for i in dts.index:
    def sfrx(z):
        x = z

        ratio=dts.loc[i]['ratio']

        #f=ratio-(np.exp(x)-np.abs(x)-1)/x**2
        f=ratio-(np.exp(x)-np.exp(np.log(x))-1)/x**2
        return f

    #for i in dts.index:
    z = fsolve(sfrx,3.0)
    dts.at[i,'x_tsf']=(z)

dts['tau']=tsf/dts['x_tsf']
dts["log_tau"]=np.log10(dts["tau"])

dts["A_tsf"]=dts["av_SFR"]*tsf/(1-(1+dts["x_tsf"])*np.exp(-dts['x_tsf']))

dts[["A_tsf","tau","x_tsf"]].describe(include='all').to_latex(position = "hc")

fname = "graphs/x-A_tsf.png"
dts.plot(kind='scatter', x='x_tsf', y='A_tsf',c= "logMt")
plt.xscale('log')
plt.yscale('log')
plt.savefig(fname)
plt.close()
caption(fname,"$A_{del} = f(x)$ for constant t_{sf}")

fname = "graphs/T-A_tsf.png"
dts.plot(kind='scatter', x='tau', y="A_tsf", c= "logMt")
plt.xscale('log')
plt.yscale('log')
plt.savefig(fname)
plt.close()
fname

###Constant tau
dtau=df.copy()
tau=3.5*10**9
zeta=1.3

dtau["z"]=zeta*dtau["StellarMass"]/tau

for i in df.index:
    def sfrx(z):
        x = z

        cons=dtau.loc[i]['z']
        SFR=dtau.loc[i]['SFR_0']


        #f=ratio-(np.exp(x)-np.abs(x)-1)/x**2
        f=cons/SFR-(np.exp(x)-np.exp(np.log(x))-1)/x
        return f

    #for i in df.index:
    z = fsolve(sfrx,3.0)
    dtau.at[i,'x_tau']=(z)

dtau["tsf"]=dtau['x_tau']*tau
dtau["log_tsf"]=np.log10(dtau.tsf)
dtau["av_SFR"]=dtau.z/dtau.x_tau
dtau['ratio']=dtau.av_SFR/dtau.SFR_0
dtau['A_tau']=tau*dtau['SFR_0']*np.exp(dtau.x_tau)/dtau.x_tau
dtau=dtau.drop(["z"],axis=1)

dtau[["A_tau","x_tau","tsf"]].describe(include='all').to_latex(position = "hc")

fname = "graphs/x-A_tau.png"
dtau.plot(kind='scatter', x='x_tau', y='A_tau',c= "logMt")
plt.xscale('log')
plt.yscale('log')
plt.savefig(fname)
plt.close()

caption(fname,r"$A_{del} = f(x)$ for constant $\tau$")

fname = "graphs/T-A_tau.png"
dtau.plot(kind='scatter', x='tsf', y='A_tau',c= "logMt")
plt.xscale('log')
plt.yscale('log')
plt.savefig(fname)
plt.close()

fname

dp=pd.merge(dtau[["Name","A_tau", "x_tau", "tsf"]], dts, on = 'Name')

dp[["x_tau","x_tsf"]].describe(include = 'all').to_latex(position = "hc")

x_comp=simple_regplot(dp,'x_tsf','x_tau',caption = "Comparing the two x")
x_comp_Mt=simple_regplot(dp,'x_tsf','x_tau',scatter_kws={"c":dp["logMt"]},caption = "Comparing the two x, according to their total mass")
x_comp_tt=simple_regplot(dp,'x_tsf','x_tau',scatter_kws={"c":dp["TType"]},caption = "Comparing the two x, according to their type")
x_comp_col=simple_regplot(dp,'x_tsf','x_tau',scatter_kws={"c":dp["logcolor"]},caption = "Comparing the two x, according to their color index")

x_comp_Mt[1]

x_comp_tt[1]

x_comp_col[1]

eq_ols(r"x|_\tau", "x|_{tsf}" , x_comp)

#Comparing the 2 results
fname="Comparing_the_A.png"
plt.scatter(data = dtau, x = "x_tau", y = "A_tau", label=r"$\tau$=3.5 Gyr")
plt.scatter(data = dts, x = "x_tsf", y = "A_tsf",alpha=0.5,label="$t_{sf}$=12.5 Gyr")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('A_del')
plt.legend(loc='upper right')
plt.grid()
plt.savefig(fname)
plt.show()
caption(fname,"Comparing the two A_{del}")

fname = "A_tau-A_tsf_X.png"
dp.plot.scatter(x = "A_tsf",
                y = "A_tau",
                c = "x_tsf", grid = True)
plt.xscale('log')
plt.yscale('log')
plt.savefig(fname)
plt.close()
caption(fname, "Comparison of the 2 A_{del}s according to their $x$")

fname = "A_tau-A_tsf_Mt.png"
dp.plot.scatter(x = "A_tsf",
                y = "A_tau",
                c = "logMt", grid = True)
plt.xscale('log')
plt.yscale('log')
plt.savefig(fname)
plt.close()
caption(fname, "Comparison of the 2 A_{del}s according to their total masses")
