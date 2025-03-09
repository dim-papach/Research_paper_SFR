topcat -stilts plot2plane \
   xpix=1818 ypix=543 \
   xlog=true ylog=true xlabel='sSFR\ [1/yr]' ylabel='A\ [M_\odot]' grid=true texttype=latex \
   xmin=4.0E-15 xmax=8.287E-8 ymin=603 ymax=1.622E27 \
   legend=true legpos=0.0112,0.939 \
   ifmt=CSV x=sSFR size=2 \
   layer_1=Mark \
      in_1=/home/dp/Documents/Research_paper_SFR/r_mcmc/joined_data.csv \
      y_1=A \
      shading_1=auto color_1=blue \
      leglabel_1=MCMC \
   layer_2=Mark \
      in_2=/home/dp/Documents/Research_paper_SFR/NR/filled_with_NR.csv \
      y_2=A_n \
      shading_2=translucent translevel_2=0.2 \
      leglabel_2=N-R \
   out=/home/dp/Documents/Research_paper_SFR/method_comparison/sSFR_A.png

topcat -stilts plot2plane \
   xpix=1242 ypix=525 \
   xlabel='\tau\ [Gyr]' ylabel="Number of galaxies" grid=true texttype=latex \
   xmin=0 xmax=20 ymin=-2.0E-5 ymax=216 \
   legend=true legpos=0.9911,0.976 \
   ifmt=CSV x=tau \
   layer_1=Histogram \
      in_1=/home/dp/Documents/Research_paper_SFR/r_mcmc/joined_data.csv \
      leglabel_1=MCMC color_1=blue\
   layer_2=Histogram \
      in_2=/home/dp/Documents/Research_paper_SFR/NR/filled_with_NR.csv \
      color_2=red \
      leglabel_2=N-R \
   out=/home/dp/Documents/Research_paper_SFR/method_comparison/tau_hist.png

topcat -stilts plot2plane \
   xpix=1242 ypix=525 \
   ylog=true xlabel='log(M_*/M_\odot)' ylabel='A_{del}\ [M_\odot]' grid=true texttype=latex \
   xmin=2.94 xmax=11.4 ymin=1199 ymax=9.713E11 \
   auxmap=viridis auxfunc=histogram auxmin=0.07 auxmax=13.71 \
   auxvisible=true auxlabel='x=t_{sf}/\tau' \
   legend=true legpos=0.0103,0.944 \
   in=/home/dp/Documents/Research_paper_SFR/r_mcmc/joined_data.csv ifmt=CSV x=logM_total y=A \
    leglabel='log(A_{del}) = 0.86\cdot log(M_*) + 1.5 \\ Correlation=94\%' antialias=true \
   layer_1=Mark \
      aux_1=x \
      shading_1=aux size_1=2 \
   layer_2=LinearFit \
   layer_3=LinearFit \
      weight_3=x \
      color_3=black dash_3=3,3 \
   legseq=_1,_2 \
   out=/home/dp/Documents/Research_paper_SFR/method_comparison/A_M_MCMC.png
