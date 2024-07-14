#!/bin/bash
cd tables
./delete_lines.sh
cd ../notes
/bin/python3 list.py
cd ../tables
./hecate_lcv_join.sh
