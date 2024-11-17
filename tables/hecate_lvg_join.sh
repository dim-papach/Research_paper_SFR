#!/bin/bash

#Keep only the LCV from Hecate

#stilts tpipe in=HECATE_v1.1.fits  cmd="select D<11" out=HECATE_LCV.fits

#Outter join for each row of hecate
stilts tmatch2 in1=final_table.ecsv in2=HECATE_LCV.fits out=outer_join.ecsv\
    join=1or2 matcher=sky\
    values1="Ra Dec" values2="RA DEC" params=2 

stilts tpipe in=outer_join.ecsv out=outer_join.csv 

echo "Done Outter"    
#Inner join
stilts tmatch2 in1=final_table.ecsv in2=HECATE_LCV.fits out=inner_join.ecsv\
    join=1and2 matcher=sky\
    values1="Ra Dec" values2="RA DEC " params=2 
stilts tpipe in=inner_join.ecsv out=inner_join.csv 
echo "Done Inner"    
#Hecate not LCV
stilts tmatch2 in1=final_table.ecsv in2=HECATE_LCV.fits out=HEC_not_LVG_join.ecsv\
    join=2not1 matcher=sky\
    values1="Ra Dec" values2="RA DEC" params=2
stilts tpipe in=HEC_not_LVG_join.ecsv out=HEC_not_LVG_join.csv
echo "Done Hecate not LCV"    
#LCV not Hecate  
stilts tmatch2 in2=final_table.ecsv in1=HECATE_LCV.fits out=LVG_not_HEC_join.ecsv\
    join=2not1 matcher=sky\
    values1="Ra Dec" values2="RA DEC" params=2
stilts tpipe in=LVG_not_HEC_join.ecsv out=LVG_not_HEC_join.csv
echo "Done LCV not Hecate"    
