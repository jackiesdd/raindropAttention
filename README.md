# Deep Learning for Seeing Through Window With Raindrops
by Yuhui Quan, Shijie Deng, Yixin Chen, Hui Ji

Shijie Deng is in charge of this code repo. If you have any questions, please contact shijie.deng.cs@foxmail.com

See more details on http://csyhquan.github.io/

## Environment
- Python3.5
- Tensorflow 1.9 with NVIDIA GPU


## Testing
The testing checkpoints can be downloaded at ：https://pan.baidu.com/s/1Ocp-xM83s2Irts1ssd5UVQ 
access code：y70s 

We use the dataset with rainy and clean images created by https://github.com/rui1996/DeRaindrop.  Edges maps are added for attention, all pictures are under the directory /testing_real/ and the outputs are under /testing_result/.
Run the code below to get the result of our model.
```bash
python run_model.py --inputdata_path ./testing_real/ --output_path ./testing_result --phase test
```
Quantitative results of PSNR and SSIM will be printed, you can check /testing_result/ for a qualitative evaluation.

## Training

Put your training rainy images at "train_img/data/" and corresponding gt clean images at "train_img/gt/"(images should be *.jpg or *.png format).

Then run the train_list.sh to generate datalist to train.
