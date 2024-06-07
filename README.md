# INSTALL
```bash
git clone https://github.com/ManTD2003/YOLOv8_custom.git
cd YOLOv8_custom/ultralytics
pip install -v -e .
```

# Config train.txt for train positive and negative images
```bash
train: 
- /home/s/man/train.txt
train_negative:
- /home/s/man/train_negative.txt
positive_ratio: 0.9 # postive_ratio = num_positive / (num_positive + num_negative) in one batch
val: 
- /home/s/man/val.txt
test: 
- /home/s/man/test.txt
# Classes
names:
  {0: polyp}
```
