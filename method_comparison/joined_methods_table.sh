stilts tmatchn nin=3 \
  in1=/home/dp/Documents/Research_paper_SFR/r_mcmc_uni/joined_data.csv ifmt1=csv \
  in2=/home/dp/Documents/Research_paper_SFR/r_mcmc_normal/joined_data.csv ifmt2=csv \
  in3=/home/dp/Documents/Research_paper_SFR/NR/filled_with_NR.csv ifmt3=csv \
  matcher=exact \
  values1='ID' values2='ID' values3='ID' \
  out=joined_output.csv ofmt=csv

stilts tmatch2 \
  in1=/home/dp/Documents/Research_paper_SFR/method_comparison/joined_output.csv ifmt1=csv \
  in2=/home/dp/Documents/Research_paper_SFR/tables/filled.csv ifmt2=csv \
  matcher=exact \
  values1='ID_1' values2='ID' \
  out=joined_output_HECATE.csv ofmt=csv
