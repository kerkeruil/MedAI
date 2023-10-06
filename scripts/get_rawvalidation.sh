# !/bin/sh

# run from MedAI folder
# Get raw data

# Run your script from the MedAI folder
cd $HOME/MedAI

# Create folders
mkdir -p raw_data/validation/images
mkdir -p raw_data/validation/labels

# Validation: 80 CT
wget -O raw_data/validation/ribfrac-val-images.zip https://zenodo.org/record/3893496/files/ribfrac-val-images.zip
wget -O raw_data/validation/ribfrac-val-labels.zip https://zenodo.org/record/3893496/files/ribfrac-val-labels.zip
wget -O raw_data/validation/ribfrac-val-info.csv https://zenodo.org/record/3893496/files/ribfrac-val-info.csv

# Unzip the zip files and rename folders
unzip raw_data/validation/ribfrac-val-images.zip -d raw_data/validation/ && mv raw_data/validation/ribfrac-val-images/* raw_data/validation/images
unzip raw_data/validation/ribfrac-val-labels.zip -d raw_data/validation/ && mv raw_data/validation/ribfrac-val-labels/* raw_data/validation/labels

# Clean up folders
rm -r raw_data/validation/ribfrac-val-images
rm -r raw_data/validation/ribfrac-val-labels

# Clean up zip files
rm raw_data/validation/ribfrac-val-images.zip
rm raw_data/validation/ribfrac-val-labels.zip

echo "Validation files downloaded and extracted successfully."