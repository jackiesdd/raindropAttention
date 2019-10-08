# Deep Learning for Seeing Through Window With Raindrops
by Yuhui Quan, Shijie Deng, Yixin Chen, Hui Ji

## Environment
- Python3.5
- Tensorflow 1.9 with NVIDIA GPU

## Testing
We use the dataset with rainy and clean images created by https://github.com/rui1996/DeRaindrop.  Edges maps are added for attention, all pictures are under the directory /testing_real/ and the outputs are under /testing_result/.
Run the code below to get the result of our model.
```bash
python run_model.py --inputdata_path ./testing_real/ --output_path ./testing_result
```
Quantitative results of PSNR and SSIM will be printed, you can check /testing_result/ for a qualitative evaluation.
