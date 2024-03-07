#!/usr/bin/env sh

wget --no-check-certificate -r -nH --cut-dirs=2 --accept="*.dat" --reject="*8.dat" -e robots=off -U mozilla https://www.sao.ru/lv/lvgdb/tables/
