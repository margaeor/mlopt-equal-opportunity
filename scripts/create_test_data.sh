#!/bin/bash
head -n $1 ../data/dataset/psam_husa.csv > ../data/dataset/test_ha.csv
head -n $1 ../data/dataset/psam_husb.csv > ../data/dataset/test_hb.csv
head -n $1 ../data/dataset/psam_pusa.csv > ../data/dataset/test_pa.csv
head -n $1 ../data/dataset/psam_pusb.csv > ../data/dataset/test_pb.csv
