# Semantic-TemporalNet: Refined Detection of Changes in Urban Blocks

**Note**: This project is under active development and will be continuously updated upon publication.

![Method Overview](img/method.jpg)

## 🔍 Project Overview

Semantic-TemporalNet is a deep learning framework designed to detect refined temporal changes in urban blocks using multi-temporal remote sensing imagery.

This project code is partially inspired by the architecture in [pytorch-playground.](https://github.com/aaron-xichen/pytorch-playground/)

## 🧪 Example Inference

To evaluate semantic consistency scores of urban blocks, run the following command using the default parser settings:

```bash
python test.py --threshold 0.8
```


This will generate semantic coherence score visualizations for:

### 🏙️ Block 46 (Wuhan)

- Demo Output  
  ![](img/46.png)

- Visualization in Paper  
  ![](img/46_paper.png)

### 🏙️ Block 1932 (Wuhan)

- Demo Output  
  ![](img/1932.png)

- Visualization in Paper  
  ![](img/1932_paper.png)

These images illustrate the model's ability to detect and evaluate fine-grained temporal consistency across urban regions.

If this is helpful for you, please cite our paper:
```bash
@ARTICLE{11172373,
  author={Sun, Lingjun and Jin, Ming and Yan, Jining and He, Haixu and Cao, Li},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Semantic-TemporalNet: A Novel Urban Block Change Detection Method Based on Semantic Coherence Analysis}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Semantics;Noise;Remote sensing;Time series analysis;Coherence;Urban areas;Residual neural networks;Land surface;Convolution;Urban Renewal;Change Detection;Time-Series Semantic Coherence;Remote Sensing;Sentinel-2},
  doi={10.1109/TGRS.2025.3611378}}
```

