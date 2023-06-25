[ ! -d ~/wordshk ] && mkdir ~/wordshk
cd ~/wordshk

dict_dev_kit_path=/Applications/Utilities/Dictionary Development Kit

if [ -d $dict_dev_kit_path ]
then
    echo "Found Dictionary Development Kit."
else
    echo "Downloading Dictionary Development Kit..."
    curl -L https://sourceforge.net/projects/wordshk-apple/files/dict_dev_kit.tar.gz/download > dict_dev_kit.tar.gz
    tar -zxf dict_dev_kit.tar.gz "Dictionary Development Kit"
    echo "Installing Dictionary Development Kit..."
    mv "Dictionary Development Kit" /Applications/Utilities
fi

# Download latest wordshk_tools
echo "Downloading latest wordshk_tools executable..."
curl -L https://sourceforge.net/projects/wordshk-apple/files/wordshk.tar.gz/download > wordshk.tar.gz
tar -zxf wordshk.tar.gz
mv wordshk/* .
rmdir wordshk
rm wordshk.tar.gz

# Download latest wordshk CSV data
echo "Downloading latest wordshk CSV data..."
curl https://words.hk/static/all.csv.gz -o wordshk.csv.gz
gunzip wordshk.csv.gz
# Remove first two info lines in CSV
sed -i.bak '1,2d' wordshk.csv
rm wordshk.csv.bak

echo "Making the latest version of the wordshk dictionary..."
sudo chmod +x ./wordshk_tools
./wordshk_tools < wordshk.csv > wordshk.xml
echo "Installing wordshk to Apple Dictionary..."
make && make install
rm wordshk_tools
