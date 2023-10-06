# !/bin/sh

# run from MedAI folder
# Get raw data

# Run your script from the MedAI folder
cd $HOME/MedAI

# Create folders
mkdir -p raw_data/train/images
mkdir -p raw_data/train/labels

# Train part 1: 300 chest CT (images 36.6GB)
wget -O raw_data/train/ribfrac-train-images-1.zip https://zenodo.org/record/3893508/files/ribfrac-train-images-1.zip
wget -O raw_data/train/ribfrac-train-labels-1.zip https://zenodo.org/record/3893508/files/ribfrac-train-labels-1.zip 
wget -O raw_data/train/ribfrac-train-info-1.csv https://zenodo.org/record/3893508/files/ribfrac-train-info-1.csv

# Unzip the zip files, move and remove old folders
unzip raw_data/train/ribfrac-train-images-1.zip -d raw_data/train/ && mv raw_data/train/Part1/* raw_data/train/images
rm -r raw_data/train/Part1

unzip raw_data/train/ribfrac-train-labels-1.zip -d raw_data/train/ && mv raw_data/train/Part1/* raw_data/train/labels
rm -r raw_data/train/Part1

# # Clean up zip files
rm raw_data/train/ribfrac-train-images-1.zip
rm raw_data/train/ribfrac-train-labels-1.zip

# Train part 2: 120 chest CT (images 14.6GB)
wget -O raw_data/train/ribfrac-train-images-2.zip https://zenodo.org/record/3893498/files/ribfrac-train-images-2.zip
wget -O raw_data/train/ribfrac-train-labels-2.zip https://zenodo.org/record/3893498/files/ribfrac-train-labels-2.zip 
wget -O raw_data/train/ribfrac-train-info-2.csv https://zenodo.org/record/3893498/files/ribfrac-train-info-2.csv

# Unzip the zip files, move and remove old folders
unzip raw_data/train/ribfrac-train-images-2.zip -d raw_data/train/ && mv raw_data/train/Part2/* raw_data/train/images
rm -r raw_data/train/Part2

unzip raw_data/train/ribfrac-train-labels-2.zip -d raw_data/train/ && mv raw_data/train/Part2/* raw_data/train/labels
rm -r raw_data/train/Part2

# Clean up zip files
rm raw_data/train/ribfrac-train-images-2.zip
rm raw_data/train/ribfrac-train-labels-2.zip 

# Combine the two train CSV files
tail -n +2 raw_data/train/ribfrac-train-info-2.csv >> raw_data/train/ribfrac-train-info-1.csv
mv raw_data/train/ribfrac-train-info-1.csv raw_data/train/ribfrac-train-info.csv

# Remove old CSV file
rm raw_data/train/ribfrac-train-info-2.csv

echo "Training files downloaded and extracted successfully."