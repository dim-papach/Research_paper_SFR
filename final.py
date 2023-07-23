import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tabulate import tabulate
import uncertainties as unc
from astropy import units as u



pd.set_option('display.float_format', lambda x: '%.2E' % x)
######Print images with caption
def caption(fname,caption,name = None):
    if name == None:
        name = caption
    return "#+name: fig:{} \n#+label: fig:{} \n#+caption:{} \n#+ATTR_LaTeX: :placement [!htpb]\n[[./{}.png]]".format(caption,name,name,fname)

def simple_regplot(data ,
    X, Y, cap = None, name = None, n_std=2, n_pts=100, ax=None, scatter_kws=None, line_kws=None, ci_kws=None):
    df_data=data
    x=df_data[X]
    y=df_data[Y]
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
    ############################
    # if log == 1:             #
    #     ax.set_yscale("log") #
    #     ax.set_xscale("log") #
    ############################

    ax.legend(loc = "best")
    if scatter_kws == {}:
        lab = X+"-"+Y
        fname = "figs/"+lab
        plt.savefig(fname)
        plt.close()
    else:
        col = scatter_kws["c"].name
        PCM=ax.get_children()[2] #get the mappable, the 1st and the 2nd are the x and y axes
        plt.colorbar(PCM, ax=ax).set_label(col)
        lab = X+"-"+Y+"-color_"+col
        fname = "figs/"+lab
        plt.savefig(fname)
        plt.close()

    if name == None:
        name = cap

    Cap = "#+name: fig:{} \n#+caption:{}\n#+ATTR_LaTeX: :placement [!htpb] \n[[./{}.png]]".format(name,cap,fname)
    return fit_results, Cap , lab

# Calculate the bin width using the Freedman-Diaconis rule
def fd_bins(x):
    iqr = x.quantile(0.75) - x.quantile(0.25)
    bin_width = 2 * iqr / x.count()**(1/3)

    # Calculate the number of bins using the bin width
    return int((df[x].max() - df[x].min()) / bin_width)
#
## Format the table as a string using the tabulate library
def no_col_str(df,string):
    return df.drop(df.filter(regex=string).columns, axis=1)
#######


#### PRINTING THE OLS PARAMETERS AS LATEX
def inline_ols(y,x,ols):
        return "${} = ({:.ueSL})\cdot {} + ({:.ueSL})$".format(y, unc.ufloat(ols[0].params[1], ols[0].bse[1]), x, unc.ufloat(ols[0].params[0], ols[0].bse[0]))

def inline_r2(ols):
        return r"$R^2 = {}\%$ \noindent".format(round(ols[0].rsquared*100))

def eq_ols(y,x,ols):
    return (r"\begin{"+"equation"+"}\label{eq:"+"{}".format(ols[2])+"}\n"+
            r"\begin{"+"align"+"}\n"+
            "& {} = ({:.ueSL})\cdot {} + ({:.ueSL}) \\\ \n".format(y, unc.ufloat(ols[0].params[1], ols[0].bse[1]), x, unc.ufloat(ols[0].params[0], ols[0].bse[0]))+
            r"& \textrm"+"{"+"with correlation "+"}"+" R^2={}\%\n".format(round(ols[0].rsquared*100))+
            r"\end{"+"align"+"}\n"+
            "\end{"+"equation"+"}\n"+r"\noindent")

flag=pd.read_csv("Karachentsev_list_flags.csv")


data = pd.read_csv("Karachentsev_list.csv")
data["TType"] = data["TType"].astype('category')
data["Tdw1"] = data["Tdw1"].astype('category')
data["Tdw2"] = data["Tdw2"].astype('category')

df = data.copy()
df['SFR_Ha']=10**df['log_SFR_Ha']

df['SFR_FUV']=10**df['log_SFR_FUV']

df['K']=10**df['logKLum']

df['MHI']=10**df['logMHI']

df["color"] = -df["Bmag"] + df['FUVmag']

temp_table = no_col_str(df,'log')
temp_table = temp_table.count()
temp_table.columns = ["sepal_length", "sepal_width"]
temp_table.to_markdown(headers = ["Measurment","Number of Galaxies"] ,tablefmt= "orgtbl")

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

typ=pd.read_csv("Karachentsev_list_flags.csv")
typ["TType"]=typ["TType"].astype('category')
typ["Tdw1"]=typ["Tdw1"].astype('category')
typ["Tdw2"]=typ["Tdw2"].astype('category')
print(typ.count())
typ['TType'].value_counts(sort=False).plot(kind='bar',logy=True,grid = 'True')
plt.xlabel("Morphology")
plt.ylabel("Number of Galaxies")
plt.savefig("figs/hist-Type")
plt.close()

typ['Tdw1'].value_counts(sort=False).plot(kind='bar', logy=True,grid = 'True')
plt.xlabel("Dwarf galaxy morphology")
plt.ylabel("Number of Galaxies")
plt.savefig("figs/hist-Tdw1")
plt.close()

typ['Tdw2'].value_counts(sort=False).plot(kind='bar', logy=True,grid = 'True')
plt.xlabel("Dwarf galaxy surface brightness morphology")
plt.ylabel("Number of Galaxies")
plt.savefig("figs/hist-Tdw2")
plt.close()

###Constant tsf
dts=df.copy()
tsf=12.5*10**9
zeta=1.3

dts['av_SFR']=dts['StellarMass']*1.3/(12.5*10**9)
dts['log_av_SFR']=np.log10(dts['av_SFR'])

dts['ratio']=dts['av_SFR']/dts['SFR_0']
dts['log_ratio']=np.log10(dts['ratio'])

def sfrx(z):
    x =max(0,z)

    ratio=dts.loc[i]['ratio']

    #f=ratio-(np.exp(x)-np.abs(x)-1)/x**2
    f=ratio-(np.exp(x)-x-1)/x**2
    return f


for i in dts.index:
    #for i in dts.index:
    z = fsolve(sfrx,3.0)
    dts.at[i,'x_tsf']=(z)

dts["log_x_tsf"]=np.log10(dts["x_tsf"])

dts['tau']=tsf/dts['x_tsf']
dts["log_tau"]=np.log10(dts["tau"])

dts["A_tsf"]=dts["av_SFR"]*tsf/(1-(1+dts["x_tsf"])*np.exp(-dts['x_tsf']))
dts["log_A_tsf"] = np.log10(dts["A_tsf"])
dts[["A_tsf","tau","x_tsf"]].describe(include='all').round(3).to_markdown(tablefmt= "orgtbl")

fname = "figs/x-A_tsf"
dts.plot(kind='scatter', x='x_tsf', y='A_tsf',c= "logMt")
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.savefig(fname)
plt.close()
caption(fname,"$A_{del} = f(x)$ for constant t_{sf}")

fname = "figs/T-A_tsf"
dts.plot(kind='scatter', x='tau', y="A_tsf", c= "logMt")
plt.xscale('log')
plt.yscale('log')
plt.savefig(fname)
plt.close()
caption(fname,r"$A_{del} = f(\tau)$ for constant t_{sf}")

A_tsf_Mt = simple_regplot(dts,"logMt","log_A_tsf",scatter_kws={"c": dts["x_tsf"]}, cap = r"Total Mass $M_t$ - $A_{del}|_{t_{sf}}$" ,name = "A_tsf_Mt")
A_tsf_Mt[1]

eq_ols("$log(A_{del}|_t_{sf})","$log(M_t)$",  A_tsf_Mt)

###Constant tau
dtau=df.copy()
tau=3.5*10**9
zeta=1.3

dtau["z"]=zeta*dtau["StellarMass"]/tau

def sfrx(var):
    x = max(13e-9,var)

    cons=dtau.loc[i]["z"]
    SFR=dtau.loc[i]['SFR_0']


    #f=ratio-(np.exp(x)-np.abs(x)-1)/x**2
    f=cons/SFR-(np.exp(x)-x-1)/x
    return f

for i in dtau.index:

    #for i in df.index:
    solution = fsolve(sfrx,3.0)
    dtau.at[i,'x_tau']=solution
dtau["log_x_tau"]=np.log10(dtau.x_tau)

dtau["tsf"]=dtau['x_tau']*tau
dtau["log_tsf"]=np.log10(dtau.tsf)
dtau["av_SFR"]=dtau.z/dtau.x_tau
dtau['ratio']=dtau.av_SFR/dtau.SFR_0
dtau['A_tau']=tau*dtau['SFR_0']*np.exp(dtau.x_tau)/dtau.x_tau
dtau["log_A_tau"]=np.log10(dtau.A_tau)

dtau[["A_tau","x_tau","tsf"]].describe(include='all').to_markdown(tablefmt = "orgtbl")

fname = "figs/x-A_tau"
dtau.plot(kind='scatter', x='x_tau', y='A_tau',c= "logMt")
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.savefig(fname)
plt.close()

caption(fname,r"$A_{del} = f(x)$ for constant $\tau$")

fname = "figs/T-A_tau"
dtau.plot(kind='scatter', x='tsf', y='A_tau',c= "logMt")
plt.xscale('log')
plt.yscale('log')
plt.savefig(fname)
plt.close()

caption(fname,r"$A_{del} = f(t_{sf})$ for constant $\tau$")

A_tau_Mt = simple_regplot(dtau,"logStellarMass","log_A_tau",scatter_kws={"c": dtau["x_tau"]}, cap = r"Total Mass $M_t$ - $A_{del}|_{\tau}$", name = "A_tau_Mt")
A_tau_Mt[1]

eq_ols(r"$log(A_{del}|_\tau)", "$log(M_t)$", A_tau_Mt)

dp=pd.merge(dtau[["Name","A_tau","log_A_tau", "x_tau", "tsf"]], dts, on = 'Name')
dp["log_x_tau"]=np.log10(dp["x_tau"])
dp["log_x_tsf"]=np.log10(dp["x_tsf"])
dp["log_tau"]=np.log10(dp["tau"])
dp["log_tsf"]=np.log10(dp["tsf"])

dp[["x_tau","x_tsf"]].describe(include = 'all').to_latex(position = "hc")

fname="figs/Comparing_the_x_Mt"

plt.scatter(data = dtau, y = "x_tau", x = "Mt", label=r"$\tau$=3.5 Gyr")
plt.scatter(data = dts, y = "x_tsf", x = "Mt",alpha=0.5,label="$t_{sf}$=12.5 Gyr")

plt.xscale('log')
plt.yscale('log')
plt.ylabel('x')
plt.xlabel('Mt')
plt.legend(loc='upper right')
plt.grid()
plt.savefig(fname)
plt.close()
caption(fname,"Comparing the two x's, According to their total masses")

fname="figs/x_tau-Mt-color"

dtau.plot.scatter(x = "Mt",y = "x_tau", c = "color")
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.savefig(fname)
plt.close()
caption(fname,r"$x|_\tau=f(M_t)$, with their color index")

x_comp=simple_regplot(dp,'x_tsf','x_tau',cap = "Comparing the two x")
x_comp_Mt=simple_regplot(dp,'x_tsf','x_tau',scatter_kws={"c":dp["logMt"]},cap = "Comparing the two x, according to their total mass")
x_comp_tt=simple_regplot(dp,'x_tsf','x_tau',scatter_kws={"c":dp["TType"]},cap = "Comparing the two x, according to their type")
x_comp_col=simple_regplot(dp,'x_tsf','x_tau',scatter_kws={"c":dp["color"]},cap = "Comparing the two x, according to their color index")

x_comp_Mt[1]

x_comp_tt[1]

x_comp_col[1]

eq_ols(r"x|_\tau", "x|_{tsf}" , x_comp)

#Comparing the 2 results
fname="figs/Comparing_the_A_x"
plt.scatter(data = dtau, x = "x_tau", y = "A_tau", label=r"$\tau$=3.5 Gyr")
plt.scatter(data = dts, x = "x_tsf", y = "A_tsf",alpha=0.5,label="$t_{sf}$=12.5 Gyr")
plt.xscale('log')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('A_del')
plt.legend(loc='upper right')
plt.grid()
plt.savefig(fname)
plt.close()
caption(fname,"Comparing the two A_{del}")

fname = "figs/A_tau-A_tsf_colo_X"
dp.plot.scatter(x = "A_tsf",
                y = "A_tau",
                c = "x_tsf", grid = True)
plt.xscale('log')
plt.yscale('log')
plt.savefig(fname)
plt.close()
caption(fname, "Comparison of the 2 A_{del}s according to their $x$")

fname = "figs/A_tau-A_tsf_Mt"
dp.plot.scatter(x = "A_tsf",
                y = "A_tau",
                c = "logMt", grid = True)
plt.xscale('log')
plt.yscale('log')
plt.savefig(fname)
plt.close()
caption(fname, "Comparison of the 2 A_{del}s according to their total masses")

dcl = dp.copy()

fname = "figs/A_tau-A_tsf_Mt"
dp.plot.scatter(x = "A_tsf",
                y = "A_tau",
                c = "logMt", grid = True)
plt.xscale('log')
plt.yscale('log')
plt.savefig(fname)
plt.close()
caption(fname, "Comparison of the 2 A_{del}s according to their total masses")

dcl['const_tsf']=np.exp(dcl.x_tsf)/(np.exp(dcl.x_tsf)-dcl.x_tsf-1)
dcl['const_tau']=np.exp(dcl.x_tau)/(np.exp(dcl.x_tau)-dcl.x_tsf-1)
dcl['A_theor_tsf']=dcl['StellarMass']*dcl.const_tsf
dcl['A_theor_tau']=dcl['StellarMass']*dcl.const_tau
dcl["log_A_theor_tsf"]=np.log10(dcl["A_theor_tsf"])
dcl["log_A_theor_tau"]=np.log10(dcl["A_theor_tau"])

fname = "figs/A_theor_tau-M*"
dcl.plot.scatter(x = "A_theor_tsf",
                y = "A_theor_tau",
                c = "x_tau",
                grid = True)
plt.xscale('log')
plt.yscale('log')
plt.savefig(fname)
plt.close()
caption(fname, "Comparison of the 2 A_{del}s according to their total masses")

A_theor_tau_StellarMass_log=simple_regplot(dcl,"log_A_theor_tau","logStellarMass",scatter_kws={"c":dcl["x_tsf"]}, cap = r"Comparison of the A_del according to their Stellar Mass")
A_theor_tau_StellarMass_log[1]

A_theor_A_exp_tau=simple_regplot(dcl,"log_A_theor_tau","log_A_tau",scatter_kws={"c":dcl["x_tsf"]}, cap = r"Comparison of the 2 $A_{del}|_{\tau=const.}$s (theoretical and experimental)", name = "A_theor_A_exp_tau")
A_theor_A_exp_tau[1]

A_theor_A_exp_tsf=simple_regplot(dcl,"log_A_theor_tsf","log_A_tsf",scatter_kws={"c":dcl["x_tsf"]}, cap = r"Comparison of the 2 $A_{del}|_{tsf=const.}$s (theoretical and experimental)", name = "A_theor_A_exp_tsf")
A_theor_A_exp_tsf[1]

t = np.linspace(0.1, 5, 900)
fname = "figs/SFR_avSFR"

plt.plot(t, t/(np.exp(t)-t-1), label = "$SFR_{0,del}$")
plt.plot(t, 1/t, label = "$\overline{SFR_{del}}$")
plt.legend(loc ="upper right")
plt.grid()
plt.xlabel("$t_{sf}$")
plt.ylabel("$SFR$")
plt.savefig(fname)
plt.close()

caption(fname, r"The $SFR_{0,del}$ and $\overline{SFR_{del}}$ for constant $\tau=1$ and $\zeta M_*=1$")

dsol = df.copy()
dsol.SFR_0.describe()

def positive_constraint(vars):
    x = max(1e-11, vars[0])  # Ensures x is positive or zero
    y = max(1e-11, vars[1])  # Ensures y is positive or zero
    return x, y
def solved(var):

    cons = positive_constraint(var)
    tsf = cons[0]
    tau = cons[1]
    x = tsf/tau

    SFR=dsol.loc[i]["SFR_0"]
    Mstellar=dsol.loc[i]["StellarMass"]

    A = zeta*Mstellar*np.exp(x)/(np.exp(x)-(x)-1)

    f = np.zeros(2)

    #####################################################
    # f[0] = tau*SFR-A*x*np.exp(-x)                     #
    # f[1] = tsf*(np.exp(x)-x-1)*SFR-zeta*Mstellar*x**2 #
    #####################################################

    f[0] = SFR*tau*np.exp(x)-A*x
    f[1] = zeta*Mstellar*x**2/tsf/SFR-(np.exp(x)-x-1)
    return f

for i in dsol.index:

    #for i in dts.index:
    z = fsolve(solved,[9.6*10**9,3.5*10**9])
    dsol.at[i,'tsf']=z[0]
    dsol.at[i,'tau']=z[1]

dsol["x"]=dsol.tsf/dsol.tau

def solved2(var):

    cons = positive_constraint(var)
    tsf = cons[0]
    tau = cons[1]
    x = tsf/tau

    SFR=dsol.loc[i]["SFR_0"]
    Mstellar=dsol.loc[i]["StellarMass"]

    A = zeta*Mstellar*np.exp(x)/(np.exp(x)-(x)-1)

    f = np.zeros(2)

    f[0] = tau*SFR-A*x*np.exp(-x)
    f[1] = tsf*(np.exp(x)-x-1)*SFR-zeta*Mstellar*x**2

    #####################################################
    # f[0] = SFR*tau*np.exp(x)-A*x                      #
    # f[1] = zeta*Mstellar*x**2/tsf/SFR-(np.exp(x)-x-1) #
    #####################################################
    return f

for i in dsol.index:

    #for i in dts.index:
    z = fsolve(solved2,[9.6*10**9,3.5*10**9])
    dsol.at[i,'tsf2']=z[0]
    dsol.at[i,'tau2']=z[1]

dsol["x2"]=dsol.tsf2/dsol.tau2

dsol['tsfG'] = dsol.tsf/10**9
dsol['tauG'] = dsol.tau/10**9

fname = "figs/tsf-hist"
plt.hist(dsol["tsfG"],range=[0,20], bins =20)
plt.grid(True)
plt.xlabel("$t_{sf}$")
plt.ylabel("Number of Galaxies")
plt.savefig(fname)
plt.close()

caption(fname, "Histogram of t_{sf} from 0 to 20 Gyr")

dsol[["tsfG","tauG","x"]].describe(include='all').round(3).to_markdown(tablefmt= "orgtbl")

dsol['tsfG2'] = dsol.tsf2/10**9
dsol['tauG2'] = dsol.tau2/10**9
dsol[["tsfG2","tauG2", "x2"]].describe(include='all').round(3).to_markdown(tablefmt= "orgtbl")

fname = "figs/tsf2-hist"
plt.hist(dsol["tsfG2"],range=[0,20], bins =20)
plt.grid(True)
plt.xlabel("$t_{sf}$")
plt.ylabel("Number of Galaxies")
plt.savefig(fname)
plt.close()

caption(fname, "Histogram of t_{sf} from 0 to 20 Gyr")

fname = "figs/tsf_tsf2"
m = 1
for i in ["tsfG","tsfG2"]:
    plt.scatter(dsol[i], dsol["color"], label = "Method {}".format(m) )
    m=m+1
plt.xscale("log")
plt.xlabel(r"$t_{sf}$ Gyr")
plt.ylabel("Color index")
plt.legend(loc = "upper right")
plt.grid()
plt.savefig(fname)
plt.close()

caption(fname, r"Comparing the two $t_{sf}$")

fname = "figs/tsf_tsf2"
m = 1
for i in ["tsfG","tau"]:
    plt.scatter(dsol[i], dsol["color"],label = "Method {}".format(m) )
    m=m+1
plt.xscale("log")
plt.xlabel(r"$t_{sf}$ Gyr")
plt.ylabel("Color index")
plt.legend(loc = "upper right")
plt.grid()
plt.savefig(fname)
plt.close()

caption(fname, r"Comparing the two $t_{sf}$")

color_mass=simple_regplot(dsol,"logMt","color",scatter_kws={"c":dcl["tsf"]}, cap = r"Comparison of the 2 $A_{del}|_{tsf=const.}$s (theoretical and experimental)", name = "A_theor_A_exp_tsf")
color_mass[1]

cols_to_use = dp.columns.difference(df.columns)
dtg = pd.merge(df, dp[cols_to_use], left_index=True, right_index=True, how='outer')

dtg["tau_g"]=df["Mg"]/df["SFR_0"]
dtg["log_tau_g"]=np.log10(dtg["tau_g"])

fname = "figs/tau_g-Mg-color_SFR"
dtg.plot(kind="scatter",x="Mg",y="tau_g", c = 'log_SFR_0')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title(r"$\tau_g=f(M_g$), with color= SFR")
plt.savefig(fname)
plt.close()
taug_cap = "[[./{}.png]]".format(fname)


taug_SFR_Mg=simple_regplot(dtg,"log_SFR_0","log_tau_g",scatter_kws={"c":dtg["logMg"]}, cap = r"Correlation of the $\tau_g$ with the SFR and the gas mass")
taug_cap + "\n" + taug_SFR_Mg[1]

taug_StellarMass=simple_regplot(dtg,"logStellarMass","log_tau_g",scatter_kws={"c":dtg["log_tau"]}, cap = r"Correlation of the $\tau_g$ with the SFR and the Stellar mass")
taug_StellarMass[1]

taug_Mt=simple_regplot(dtg,"logMt","log_tau_g",scatter_kws={"c":dtg["log_tsf"]}, cap = r"Correlation of the $\tau_g$ with the total mass and the mass of the gas")
taug_Mt[1]

taug_color=simple_regplot(dtg,"color","log_tau_g",scatter_kws={"c":dtg["log_Mass_ratio"]}, cap = r"Correlation of the $\tau_g$ with the color index")
taug_color[1]

taug_tsf=simple_regplot(dtg,"log_tsf","log_tau_g",scatter_kws={"c":dtg["log_tau"]}, cap = r"Correlation of the $\tau_g$ with the color index")
taug_tsf[1]

cols_to_use = dtg.columns.difference(df.columns)
dm = pd.merge(df, dtg[cols_to_use], left_index=True, right_index=True, how='outer')

nam = "mg_SMass"
cap = "Gas Mass-Stellar Mass plot"
mg_SMass = simple_regplot(dm,"logMg","logStellarMass",cap=cap, name = nam)
mg_SMass_tg = simple_regplot(dm,"logMg","logStellarMass",scatter_kws={"c": dm["log_tau_g"]},cap=cap, name = nam)
mg_SMass_color = simple_regplot(dm,"logMg","logStellarMass",scatter_kws={"c": dm["color"]},cap=cap, name = nam)
mg_SMass_color[1]

eq_ols("$M_g$","$M_*$", mg_SMass)

nam = "SMass_m26"
cap = "Mass inside the Holmberg radius-Stellar Mass plot"
SMass_m26 = simple_regplot(dm,"logStellarMass","logM26",cap=cap, name = nam)
SMass_m26_tg = simple_regplot(dm,"logStellarMass","logM26",scatter_kws={"c": dm["log_tau_g"]},cap=cap, name = nam)
SMass_m26_tg[1]

eq_ols("M26", "M*",SMass_m26)

nam = "mg_m26"
cap = "Mass inside the Holmberg radius-Gas Mass plot"
mg_m26 = simple_regplot(dm,"logMg","logM26",cap = cap, name = nam)
mg_m26[1]

eq_ols("M26", "Mg",mg_m26)

cap = "Stellar Mass-Total Mass plot"
nam = "SMass_mt"
SMass_mt = simple_regplot(dm,"logStellarMass","logMt",cap = cap, name = nam)
SMass_mt_tg = simple_regplot(dm,"logStellarMass","logMt",scatter_kws = {"c": dm["log_tau_g"]},cap = cap, name = nam)
SMass_mt_SFR = simple_regplot(dm,"logStellarMass","logMt",scatter_kws = {"c": dm["log_SFR_0"]},cap = cap, name = nam)
SMass_mt_mg = simple_regplot(dm,"logStellarMass","logMt",scatter_kws = {"c": dm["logMg"]},cap = cap, name = nam)
SMass_mt_ratio = simple_regplot(dm,"logStellarMass","logMt",scatter_kws = {"c": dm["log_Mass_ratio"]},cap = cap, name = nam)
SMass_mt_color = simple_regplot(dm,"logStellarMass","logMt",scatter_kws = {"c": dm["color"]},cap = cap, name = nam)
SMass_mt_SFR[1]

eq_ols('$M_t$',"$M_*$", SMass_mt )

cap = "Total Mass - Gas Mass plot"
nam = "mg_mt"
mg_mt = simple_regplot(dm,"logMg","logMt",scatter_kws = {"c":dm['log_SFR_0']},cap = cap, name = nam)
mg_mt_SFR = simple_regplot(dm,"logMg","logMt",scatter_kws = {"c":dm['log_SFR_0']},cap = cap, name = nam)
mg_mt_tg = simple_regplot(dm,"logMg","logMt",scatter_kws = {"c":dm['log_tau_g']},cap = cap, name = nam)
mg_mt_SMass = simple_regplot(dm,"logMg","logMt",scatter_kws = {"c":dm['logStellarMass']},cap = cap, name = nam)
mg_mt_SFR[1]

eq_ols('$M_t$',"$M_g$", mg_mt )

cap = "Mass inside the Holmberg radius-Total Mass plot"
nam = "m26_mt"
m26_mt = simple_regplot(dm,"logM26","logMt",cap = cap, name = nam)

m26_mt[1]

eq_ols("M26", "$M_t$", m26_mt)

cap = r"$\t_{sf}$-Mass ratio $\left(\frac{M_*}{M_g}\right)$ plot"
nam = "tsf_mr"
tsf_mr = simple_regplot(dm,"log_tsf","log_Mass_ratio",scatter_kws={"c": dm["color"]},cap = cap, name = nam)
tsf_mr[1]

col_Mr = simple_regplot(dm,"color","log_Mass_ratio", scatter_kws={"c":dm["logMt"]}, cap = r"Mass ratio $\frac{M_*}{M_g}$-Color index plot", name = "col_Mr")
col_Mr[1]

######### SFR ##########

SFR_SMass_tg = simple_regplot(dm, "log_SFR_0", "logStellarMass", scatter_kws = {"c":dm["log_tau_g"]})

SFR_tg_SMass = simple_regplot(dm, "log_SFR_0", "log_tau_g", scatter_kws = {"c":dm["logStellarMass"]})

SFR_Mg_tg = simple_regplot(dm, "log_SFR_0", "logMg", scatter_kws = {"c":dm["log_tau_g"]})

SFR_Mt_tg = simple_regplot(dm, "logMt", "log_SFR_0", scatter_kws = {"c":dm["log_tau_g"]})

SFR_col = simple_regplot(dm, "log_SFR_0", "color")
SFR_col[1]

SFR_SMass_tg[1]

SFR_tg_SMass[1]

SFR_Mt_tg[1]

simple_regplot(dm, "logMt", "log_tau", scatter_kws = {"c":dm["log_tau_g"]})

simple_regplot(dm, "logMt", "log_tsf", scatter_kws = {"c":dm["log_tau_g"]})
