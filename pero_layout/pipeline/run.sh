python3 ./pero-ocr/user_scripts/parse_folder.py  \
         -i ../../img/validation \
         --output-alto-path ./output/alto \
         -c ./pero-ocr/engine/config.ini \
         --device cpu

python3 ./processing/ocr-to-json.py ./output/alto/ ../../img/validation

# rm -rf ./output