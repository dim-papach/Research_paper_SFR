topcat -stilts plot2plane \
   xpix=1242 ypix=525 \
   xlog=true ylog=true xlabel='sSFR [1/yr]' ylabel='A_{del} [M_\odot]' grid=true texttype=latex \
   xmin=4.0E-15 xmax=8.287E-8 ymin=458 ymax=4.545E33 \
   legend=true legpos=0.9957,0.968 \
   in=/home/dp/Documents/Research_paper_SFR/NR/filled_with_NR.csv ifmt=CSV x=sSFR y=A_n shading=auto size=2 forcebitmap = true\
   layer_1=Mark \
      icmd_1='select A_n>1e17' \
      color_1=blue \
      leglabel_1='A_{del}>10^{15}' \
   layer_2=Mark \
      icmd_2='select !(A_n>1e17)' \
      leglabel_2='A_{del}<10^{15}'\
   out=/home/dp/Documents/Research_paper_SFR/NR/A-sSFR.png 
   
   
topcat -stilts plot2plane \
   xpix=1242 ypix=525 \
   ylog=true xlabel='\log(M_*/M_\odot)' ylabel='A_{del}\, [M_\odot]' grid=true texttype=latex \
   xmin=2.94 xmax=11.4 ymin=458 ymax=4.545E33 \
   legend=true legpos=0.0091,0.984 \
   in=/home/dp/Documents/Research_paper_SFR/NR/filled_with_NR.csv ifmt=CSV x=logM_total y=A_n shading=auto size=2 antialias=true forcebitmap = true\
   layer_1=Mark \
      icmd_1='select A_n>1e17' \
      color_1=blue \
      leglabel_1='A_{del}>10^{15}\\ \log(A_{del})=0.58\cdot\log(M_*)+18.5\\Correlation = 55\% \\ \hline' \
   layer_2=LinearFit \
      icmd_2='select A_n>1e17' \
      color_2=blue \
      leglabel_2='A_{del}>10^{15}\\ \log(A_{del})=0.58\cdot\log(M_*)+18.5\\Correlation = 55\% \\ \hline' \
   layer_3=Mark \
      icmd_3='select !(A_n>1e17)' \
      color_3=green \
      leglabel_3='A_{del}<10^{15}\\ \log(A_{del})=0.92\cdot\log(M_*)+1.02\\Correlation = 91\%' \
   layer_4=LinearFit \
      icmd_4='select !(A_n>1e17)' \
      color_4=green \
      leglabel_4='A_{del}<10^{15}\\ \log(A_{del})=0.92\cdot\log(M_*)+1.02\\Correlation = 91\%' \
   out=/home/dp/Documents/Research_paper_SFR/NR/A-M_*.png 
   
topcat -stilts plot2plane \
   xpix=1242 ypix=525 \
   ylog=true xlabel='\log(M_*/M_\odot)' ylabel='A_{del}\, [M_\odot]' grid=true texttype=latex \
   xmin=2.94 xmax=11.4 ymin=458 ymax=4.545E33 \
   auxclip=0,1 auxfunc=histolog auxmin=3 auxmax=5.123E13 \
   auxvisible=true auxlabel='\tau\, [Gyr]' \
   legend=true legpos=0.0091,0.984 \
   in=/home/dp/Documents/Research_paper_SFR/NR/filled_with_NR.csv ifmt=CSV x=logM_total y=A_n aux=tau shading=aux size=2 auxnullcolor= antialias=true forcebitmap = true\
   layer_1=Mark \
      icmd_1='select A_n>1e17' \
      leglabel_1='A_{del}>10^{15}\\ \log(A_{del})=0.58\cdot\log(M_*)+18.5\\Correlation = 55\% \\ \hline' \
   layer_2=LinearFit \
      icmd_2='select A_n>1e17' \
      color_2=blue \
      leglabel_2='A_{del}>10^{15}\\ \log(A_{del})=0.58\cdot\log(M_*)+18.5\\Correlation = 55\% \\ \hline' \
   layer_3=Mark \
      icmd_3='select !(A_n>1e17)' \
      leglabel_3='A_{del}<10^{15}\\ \log(A_{del})=0.92\cdot\log(M_*)+1.02\\Correlation = 91\%' \
   layer_4=LinearFit \
      icmd_4='select !(A_n>1e17)' \
      color_4=green \
      leglabel_4='A_{del}<10^{15}\\ \log(A_{del})=0.92\cdot\log(M_*)+1.02\\Correlation = 91\%' \
   out=/home/dp/Documents/Research_paper_SFR/NR/A-M_*-c_tau.png 
   
   
topcat -stilts plot2plane \
   xpix=1242 ypix=525 \
   ylog=true xlabel='\log(M_*/M_\odot)' ylabel='A_{del}\, [M_\odot]' grid=true texttype=latex \
   xmin=2.94 xmax=11.4 ymin=458 ymax=4.545E33 \
   auxmap=viridis auxfunc=sqrt auxmin=-1.0E-7 auxmax=5.21 \
   auxvisible=true auxlabel='x=t_{sf}/\tau' \
   legend=true legpos=0.0091,0.984 \
   in=/home/dp/Documents/Research_paper_SFR/NR/filled_with_NR.csv ifmt=CSV x=logM_total y=A_n aux=x_n shading=aux size=2 auxnullcolor= antialias=true forcebitmap = true\
   layer_1=Mark \
      icmd_1='select A_n>1e17' \
      leglabel_1='A_{del}>10^{15}\\ \log(A_{del})=0.58\cdot\log(M_*)+18.5\\Correlation = 55\% \\ \hline' \
   layer_2=LinearFit \
      icmd_2='select A_n>1e17' \
      color_2=blue \
      leglabel_2='A_{del}>10^{15}\\ \log(A_{del})=0.58\cdot\log(M_*)+18.5\\Correlation = 55\% \\ \hline' \
   layer_3=Mark \
      icmd_3='select !(A_n>1e17)' \
      leglabel_3='A_{del}<10^{15}\\ \log(A_{del})=0.92\cdot\log(M_*)+1.02\\Correlation = 91\%' \
   layer_4=LinearFit \
      icmd_4='select !(A_n>1e17)' \
      color_4=green \
      leglabel_4='A_{del}<10^{15}\\ \log(A_{del})=0.92\cdot\log(M_*)+1.02\\Correlation = 91\%'\
      out=/home/dp/Documents/Research_paper_SFR/NR/A-M_*-c_x.png 
