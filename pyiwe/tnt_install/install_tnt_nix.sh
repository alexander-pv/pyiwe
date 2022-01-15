echo -e "\nTNT installation for Linux\n"

if [ -z "$1" ]; then
  TNT_PATH=$HOME/TNT
  echo -e "\nNo PATH specified for TNT. Selecting default PATH: $TNT_PATH\n"

else
  TNT_PATH=$1
  echo -e "\nSpecified TNT PATH: $1\n"

fi

echo -e "\nPreparing installation"
echo -e "\nInstalling Parallel Virtual Machine (PVM)"
sudo apt-get update
sudo apt-get install -y pvm

echo -e "\nCreating installation folder"
mkdir $TNT_PATH && cd $TNT_PATH

echo -e "\nDownloading TNT"
wget http://www.lillo.org.ar/phylogeny/tnt/tnt64.zip

echo -e "\nUnpacking..."
unzip tnt64.zip -d $TNT_PATH
rm tnt64.zip

echo -e "\nAdding TNT to PATH"
echo 'export PATH=$PATH:'"${TNT_PATH}/" >>~/.bashrc
echo -e "\nInstallation was completed. Write 'tnt' command in new terminal to start the program."
echo -e "\nYou may need to restart your terminal."
source ~/.bashrc
