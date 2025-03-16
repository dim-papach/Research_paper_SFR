#!/bin/bash

stilts tpipe \
  in=outer_join.ecsv \
  omode=out out=filled.ecsv \
  cmd='addcol -units "Mpc" -desc "Merge Distance" merge_D "NULL_D ? Dis : D"' \
  cmd='addcol -units "km/h" -desc "Merge radial velocity" merge_V "NULL_V ? RVel : V"' \
  cmd='addcol -desc "Merge numerical Hubble Type" merge_T "NULL_T ? TType : T"' \
  cmd='addcol -units "log(Lsol)" -desc "Merge logK" merge_logKLum "NULL_logL_K ? logKLum : logL_K"' \
  cmd='addcol -units "mag" -desc "Merge K mag" merge_Kmag "NULL_K ? Kmag : K"' \
  cmd='addcol -units "mag" -desc "Merge B mag" merge_Bmag "NULL_BT ? mag_B : BT"' \
  cmd='addcol -units "log(Msol/yr)" -desc "Merge of log SFR" logSFR_total "NULL_logSFR_HEC ? 0.7*logSFR_UNGC-0.6 : logSFR_HEC"'\
  cmd='addcol -units "Msol/yr" -desc "Total SFR in linear scale" SFR_total "exp10(logSFR_total)"'\
  cmd='addcol -units "log(Msol)" -desc "" logM_total "NULL_logM_HEC ? log10(0.82)+merge_logKLum: logM_HEC"'\
  cmd='addcol -units "Msol" -desc "Total M_* in linear scale" M_total "exp10(logM_total)"'\
  cmd='addcol -units "1/yr" -desc "Specific SFR" sSFR "SFR_total/M_total"'\
  cmd='addcol -units "log10(1/yr)" -desc "Specific SFR in logarithmic scale" logsSFR "log10(sSFR)"'\
  cmd='addcol -desc "ID column" ID "Index"'\
  cmd='addcol -desc "g-r color" gr "g-r"'\
  cmd='addcol -desc "b-v color" bv "bt-vt"'\
  cmd='addcol -desc "b-u color" bu "bt-ut"'\
  cmd='addcol -desc "absolute mag of B-Band" MB "merge_Bmag - 5*log10(merge_D)-25"'\
  cmd='addcol -desc "absolute mag of r-Band" MR "r - 5*log10(r)-25"'\


stilts tpipe \
  in=filled.ecsv \
  omode=out out=filled.csv\
