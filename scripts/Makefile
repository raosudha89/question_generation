cd /fs/clip-ml/raosudha/stackexchange
#download 36GB datadump from stackoverflow
wget https://archive.org/compress/stackexchange/formats=7Z&file=/stackexchange.zip
unzip formats\=7Z
#extract each 7z file
#login to cherokee since 7za is only there
for f in *.7z; do /opt/local/stow/p7zip-9.38.1/bin/7za e $f -o${f%.7z}; done
sh scripts/run_cluster_questions_v2.sh
