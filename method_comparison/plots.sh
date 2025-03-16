topcat -stilts plot2plane \
   xpix=1818 ypix=543 \
   xlog=true ylog=true xlabel='sSFR\ [1/yr]' ylabel='A\ [M_\odot]' grid=true texttype=latex \
   xmin=4.0E-15 xmax=8.287E-8 ymin=603 ymax=1.622E27 \
   legend=true legpos=0.0112,0.939 \
   ifmt=CSV x=sSFR size=2 \
   layer_1=Mark \
      in_1=/home/dp/Documents/Research_paper_SFR/r_mcmc_uni/joined_data.csv \
      y_1=A_up \
      shading_1=auto color_1=blue \
      leglabel_1="MCMC\ with\ \tau \ \sim uniform(1,20)" \
   layer_2=Mark \
      in_2=/home/dp/Documents/Research_paper_SFR/NR/filled_with_NR.csv \
      y_2=A_n \
      shading_2=translucent translevel_2=0.2 shape_2=filled_diamond\
      leglabel_2=N-R \
   layer_3=Mark \
      in_3=/home/dp/Documents/Research_paper_SFR/r_mcmc_normal/joined_data.csv \
      y_3=A_np \
      shading_3=auto color_3=green shape_3=cross\
      leglabel_3="MCMC with\ \tau \sim normal(4,1)" \
   out=/home/dp/Documents/Research_paper_SFR/method_comparison/sSFR_A.png

topcat -stilts plot2plane \
   xpix=1242 ypix=525 \
   xlabel='\tau\ [Gyr]' ylabel="Number of galaxies" grid=true texttype=latex \
   xmin=0 xmax=20 ymin=-2.0E-5 ymax=216 \
   legend=true legpos=0.9911,0.976 \
   ifmt=CSV \
   layer_1=Histogram \
      in_1=/home/dp/Documents/Research_paper_SFR/r_mcmc_uni/joined_data.csv \
      x_1=tau_up \
      leglabel_1="MCMC\ with\ \tau \ \sim uniform(1,20)" color_1=blue\
   layer_2=Histogram \
      in_2=/home/dp/Documents/Research_paper_SFR/NR/filled_with_NR.csv \
      x_2=tau_n \
      color_2=red \
      leglabel_2=N-R \
   layer_3=Histogram \
      in_3=/home/dp/Documents/Research_paper_SFR/r_mcmc_normal/joined_data.csv \
      x_3=tau_np \
      leglabel_3="MCMC with\ \tau \sim normal(4,1)" color_3=green\
   out=/home/dp/Documents/Research_paper_SFR/method_comparison/tau_hist.png
   
topcat -stilts plot2plane \
   xpix=1242 ypix=525 \
   xlabel='t_{sf}\ [Gyr]' ylabel="Number of galaxies" grid=true texttype=latex \
   legend=true legpos=0.9911,0.976 \
   ifmt=CSV\
   layer_1=Histogram \
      in_1=/home/dp/Documents/Research_paper_SFR/r_mcmc_uni/joined_data.csv \
      x_1=t_sf_up\
      leglabel_1="MCMC\ with\ t_{sf}\ \sim uniform(1,20)" color_1=blue\
   layer_3=Histogram \
      in_3=/home/dp/Documents/Research_paper_SFR/r_mcmc_normal/joined_data.csv \
      x_3=t_sf_np\
      leglabel_3="MCMC\ with\ t_{sf}\ \sim normal(4,1)" color_3=green\
   out=/home/dp/Documents/Research_paper_SFR/method_comparison/tsf_hist.png

topcat -stilts plot2plane \
   xpix=1242 ypix=525 \
   ylog=true xlabel='log(M_*/M_\odot)' ylabel='A_{del}\ [M_\odot]' grid=true texttype=latex \
   xmin=2.94 xmax=11.4 ymin=1199 ymax=9.713E11 \
   auxmap=viridis auxfunc=histogram auxmin=0.07 auxmax=13.71 \
   auxvisible=true auxlabel='x=t_{sf}/\tau' \
   legend=true legpos=0.0103,0.944 \
   in=/home/dp/Documents/Research_paper_SFR/r_mcmc_uni/joined_data.csv ifmt=CSV x=logM_total y=A_up \
    leglabel='log(A_{del}) = 0.86\cdot log(M_*) + 1.5 \\ Correlation=94\%' antialias=true \
   layer_1=Mark \
      aux_1=x_up \
      shading_1=aux size_1=2 \
   layer_2=LinearFit \
   layer_3=LinearFit \
      weight_3=x_up \
      color_3=black dash_3=3,3 \
   legseq=_1,_2 \
   out=/home/dp/Documents/Research_paper_SFR/method_comparison/A_M_MCMC_uni.png
   
#weighted for x Corr = 98%
   
topcat -stilts plot2plane \
   xpix=1242 ypix=525 \
   ylog=true xlabel='log(M_*/M_\odot)' ylabel='A_{del}\ [M_\odot]' grid=true texttype=latex \
   xmin=2.94 xmax=11.4 ymin=1199 ymax=9.713E11 \
   auxmap=viridis auxfunc=histogram auxmin=0.07 auxmax=13.71 \
   auxvisible=true auxlabel='x=t_{sf}/\tau' \
   legend=true legpos=0.0103,0.944 \
   in=/home/dp/Documents/Research_paper_SFR/r_mcmc_normal/joined_data.csv ifmt=CSV x=logM_total y=A_np \
    leglabel='log(A_{del}) = 0.94\cdot log(M_*) + 0.75 \\ Correlation=99\%' antialias=true \
   layer_1=Mark \
      aux_1=x_np \
      shading_1=aux size_1=2 \
   layer_2=LinearFit \
   layer_3=LinearFit \
      weight_3=x_np \
      color_3=black dash_3=3,3 \
   legseq=_1,_2 \
   out=/home/dp/Documents/Research_paper_SFR/method_comparison/A_M_MCMC_norm.png
   
#weighted for x Corr = 99.5%
