
for CLASS in "cassette player" "chain saw" "church" "gas pump" "tench" "garbage truck" "English springer" "golf ball" "parachute" "French horn"
do
	echo $CLASS	
	sbatch ibex_scripts/train_with_chatgpt_negatives.sh "$CLASS"
done
