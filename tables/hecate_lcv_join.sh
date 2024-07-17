#!/bin/bash

#Keep only the LCV from Hecate

stilts tpipe in=HECATE_v1.1.fits  cmd="select D<11" out=HECATE_LCV.fits

#Outter join for each row of hecate
stilts tmatch2 in1=final_table.ecsv in2=HECATE_LCV.fits out=inner_join.fits\
    join=all2 matcher=sky3d\
    values1="Ra Dec Dis" values2="RA DEC D" params=0.2 
#Inner join
stilts tmatch2 in1=final_table.ecsv in2=HECATE_LCV.fits out=inner_join.fits\
    join=1and2 matcher=sky3d\
    values1="Ra Dec Dis" values2="RA DEC D" params=0.2 
