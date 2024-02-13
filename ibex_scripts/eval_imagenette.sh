
for CLASS in "cassette player" "chain saw" "church" "gas pump" "tench" "garbage truck" "english springer" "golf ball" "parachute" "french horn"
do
	echo $CLASS	
	sbatch ibex_scripts/eval_metrics.sh "$CLASS"
done
