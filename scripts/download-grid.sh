cd "data/"
mkdir "grid" && cd "grid"
mkdir "raw" "audio" "video"
cd "raw" && mkdir "audio" "video"

for i in `seq 1 3`
do
    printf "Downloading $i th speaker\n"

    cd "audio" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/audio/s$i.tar" > "s$i.tar" && cd ..
    cd "video" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/video/s$i.mpg_vcd.zip" > "s$i.zip" && cd ..

    unzip -q "video/s$i.zip" -d "../video"
    tar -xf "audio/s$i.tar" -C "../audio"
done
