bash ./clean.sh
bash ./comp.sh

./build/darwin/antarys \
  --port=8080 \
  --data-dir=./data \
  --enable-gpu=false \
  --shards=16 \
  --query-threads=8 \
  --commit-interval=60 \
  --cache-size=10000 \
  --enable-hnsw=true \
  --optimization=3 \
  --meta-dir=./metadata \
  --enable-pq=true
