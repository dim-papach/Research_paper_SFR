#!/bin/bash

stilts tmatch2 in1=final_table.fits in2=HECATE_LCV.fits out=inner_join.fits\
    join=1and2 matcher=sky3d\
    values1="Coordinates.ra Coordinates.dec Dis" values2="RA DEC D" params=0.5 
