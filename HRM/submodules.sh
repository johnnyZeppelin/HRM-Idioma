git clone --recursive https://github.com/fchollet/ARC-AGI.git
git clone --recursive https://github.com/arcprize/ARC-AGI-2.git
git clone --recursive https://github.com/victorvikram/ConceptARC.git

mv ./ARC-AGI ./dataset/raw-data/
mv ./ARC-AGI-2 ./dataset/raw-data/
mv ./ConceptARC ./dataset/raw-data/

# ARC-1
# python dataset/build_arc_dataset.py  # ARC offical + ConceptARC, 960 examples
# ARC-2
# python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000  # ARC-2 official, 1120 examples

# Sudoku-Extreme
# python dataset/build_sudoku_dataset.py  # Full version
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples

# Maze
# python dataset/build_maze_dataset.py  # 1000 examples
