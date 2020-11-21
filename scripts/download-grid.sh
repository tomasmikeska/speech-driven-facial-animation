cd "data/"
mkdir "grid" && cd "grid"
mkdir "raw" "audio" "video"
cd "raw" && mkdir "audio" "video"

for i in `seq 1 34`
do
    printf "Downloading $i th speaker\n"

    # Download and unzip GRID videos
    cd "video" && curl "http://spandh.dcs.shef.ac.uk/gridcorpus/s$i/video/s$i.mpg_vcd.zip" > "s$i.zip" && cd ..
    unzip -q "video/s$i.zip" -d "../video"

    mkdir "audio/s$i"

    # Extract audio from downloaded videos using ffmpeg
    for video_path in video/s$i/*.mpg
    do
        file_name="$(basename "$video_path" .mpg)"
        ffmpeg -i "$video_path" -vn -ar 16000 -ac 1 "audio/s$i/$file_name.wav" &>> /dev/null
    done
done
