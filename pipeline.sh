#!/bin/bash

cowsay "Download UNGC"
./downloads.sh
cd tables
cowsay "Delete Problem Galaxies"
./delete_lines.sh lvg_*
cd ../notes
cowsay "compare UNCG"
./list.py
cd ../tables
cowsay "Join UNGC-HEC based on distance"
./hecate_lcv_join.sh
cd ../compare
cowsay "Compare lists"
quarto render compare.qmd
cd ../tables
cowsay "Mass Flags"
./mass_type.sh
cd ../compare.sh
cowsay "quick comparison"
./quick_diff.py
./quick_comp.py
cd ../tables
cowsay "Join the common columns"
./column_join.sh
