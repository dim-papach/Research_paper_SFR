#!/bin/bash

./downloads.sh
cd tables
./delete_lines.sh lvg_*
cd ../notes
/bin/python3 list.py
cd ../tables
./hecate_lcv_join.sh
cd ../compare
quarto render compare.qmd
cd ../tables
./mass_type.sh
cd ../compare.sh
./quick_diff.py
./quick_comp.py
