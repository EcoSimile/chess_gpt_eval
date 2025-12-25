run_batch() {
  local lvl=$1
  for i in 1 2 3; do
    STOCKFISH_SKILL="$lvl" \
    RECORDING_FILE="logs/CGPT_run${i}_vs_stockfish_level_${lvl}_2s.csv" \
    PLAYER_ONE_RECORDING_NAME="CGPT_run${i}" \
    PLAYER_TWO_RECORDING_NAME="stockfish_level_${lvl}_2s" \
    python -u main.py &
  done
  wait
}

# three at level 6, then three at level 7
run_batch 6
run_batch 7
