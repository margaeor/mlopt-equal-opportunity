#!/bin/bash
perl -ne 'print if (rand() < .001)' ../data/dataset/psam_husa.csv > ../data/dataset/test_ha.csv
perl -ne 'print if (rand() < .001)' ../data/dataset/psam_husb.csv > ../data/dataset/test_hb.csv
perl -ne 'print if (rand() < .001)' ../data/dataset/psam_pusa.csv > ../data/dataset/test_pa.csv
perl -ne 'print if (rand() < .001)' ../data/dataset/psam_pusb.csv > ../data/dataset/test_pb.csv