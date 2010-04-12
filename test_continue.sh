#!/bin/bash

path="$(mktemp -d)"

./bistable.py --seed=1234 --force=0.5:0.7:50 --gamma_=0.3 --psd=1.0 --spp=200 --rng=nr32 --paths=16384 --simperiods=160 --samples=2000 --output="${path}/golden" --output_mode=path --d0=0.030 
./bistable.py --seed=1234 --force=0.5:0.7:50 --gamma_=0.3 --psd=1.0 --spp=200 --rng=nr32 --paths=16384 --simperiods=70 --samples=2000 --output="${path}/part_t1" --output_mode=path --d0=0.030  --dump_state="${path}/part1.dump"
./bistable.py --seed=1234 --force=0.5:0.7:50 --gamma_=0.3 --psd=1.0 --spp=200 --rng=nr32 --paths=16384 --simperiods=95 --samples=2000 --output="${path}/part_t2" --output_mode=path --d0=0.030  --restore_state="${path}/part1.dump" --dump_state="${path}/part2.dump" --continue
./bistable.py --seed=1234 --force=0.5:0.7:50 --gamma_=0.3 --psd=1.0 --spp=200 --rng=nr32 --paths=16384 --simperiods=160 --samples=2000 --output="${path}/part_t3" --output_mode=path --d0=0.030  --restore_state="${path}/part2.dump" --continue

tail -n 100 "${path}/part_t3" > "${path}/b"
tail -n 100 "${path}/golden" > "${path}/a"

if diff "${path}/a" "${path}/b"; then
	echo "Test passed."
	rm -rf "${path}"
else
	echo "Test failed."
	echo "Data dir: ${path}"
fi



