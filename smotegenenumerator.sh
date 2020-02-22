#! /bin/bash
loc_to_load=$1
loc_to_save=$2
num_to_generate=6000
neighbor_constant=10
for current_number in {0..9}
do
  random_constant=$(date +%N)
  python3 smoteimagedataset.py $loc_to_load$current_number/ 28 28 $num_to_generate $loc_to_save$current_number/ $neighbor_constant $random_constant
done
