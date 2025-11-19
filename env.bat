pip3 install transformers==4.57.1
nvcc -V
echo "torch的版本要根据自己机器的CUDA版本调整，上面已经显示"
pip3 install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
pip3 install accelerate==1.11.0
pip3 install datasets==4.3.0
pip3 install peft==0.17.1
