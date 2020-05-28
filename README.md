# Hierarchical Online Instance Matching for Person Search

This repository hosts our code for our paper [Hierarchical Online Instance Matching for Person Seach](https://aaai.org/Papers/AAAI/2020GB/AAAI-ChenD.1557.pdf). 



## Preparation

1. Clone this repo

   ```bash
   git clone https://github.com/DeanChan/HOIM-PyTorch.git && cd HOIM-PyTorch
   ```

2. Build environment with [conda](https://docs.anaconda.com/anaconda/install/linux/)

   ```bash
   conda create --name HOIM --file requirements.txt
   ```



## Experiments

1.  Download [CUHK-SYSU](https://github.com/ShuangLI59/person_search) and [PRW](http://www.liangzheng.com.cn/Project/project_prw.html) to `data/`

2. Download the trained model
   ```bash
   mkdir logs && wget https://github.com/DeanChan/HOIM-PyTorch/releases/download/v0.0.0/logs.tar.gz -P logs/ 
   cd logs && tar xvzf logs.tar.gz
   ```

3. Test
   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/test_hoim.py -p logs/cuhk_sysu/
   ```
   
4. Train
   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/train_hoim.py --debug --lr_warm_up -p ./logs/<your_logging_path>/ --batch_size 5 --nw 5 --w_RCNN_loss_bbox 10.0 --epochs 22 --lr 0.003
   ```



## Citation

```latex
@inproceedings{chen2020hoim,
      title={Hierarchical Online Instance Matching for Person Search},
      author={Chen, Di and Zhang, Shanshan and Ouyang, Wanli and Yang, Jian and Schiele, Bernt},
      booktitle={AAAI},
      year={2020}
    }
```
