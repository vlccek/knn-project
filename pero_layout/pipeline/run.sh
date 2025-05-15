python3 ./pero-ocr/user_scripts/parse_folder.py  \
         -i ./images \
         --output-render-path ./output/renders/ \
         --output-logit-path ./output/logit/ \
         --output-alto-path ./output/alto \
         --output-xml-path ./output/page \
         -c ./pero-ocr/engine/config.ini \
         --device cpu

python3 ./processing/ocr-to-json.py ./pero-ocr/output/alto/ ./images

rm -rf ./output