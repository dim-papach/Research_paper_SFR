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
