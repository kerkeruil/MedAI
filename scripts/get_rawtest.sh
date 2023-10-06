# !/bin/sh

# run from MedAI folder
# Get raw data

# Run your script from the MedAI folder
cd $HOME/MedAI

# Create folders
mkdir -p raw_data/test/images

# Test images: 160 CT
wget -O raw_data/test/ribfrac-test-images.zip https://zenodo.org/record/3993380/files/ribfrac-test-images.zip

# Unzip the zip files and rename folders
unzip raw_data/test/ribfrac-test-images.zip -d raw_data/test/ && mv raw_data/test/ribfrac-test-images/* raw_data/test/images

# Clean up folders
rm -r raw_data/test/ribfrac-test-images

# Clean up zip files
rm raw_data/test/ribfrac-test-images.zip

echo "Test files downloaded and extracted successfully."