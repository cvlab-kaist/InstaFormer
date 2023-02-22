# Instance-aware Image-to-Image Translation with Transformer (CVPR 2022)
This is the implementation of the paper <a href="https://arxiv.org/abs/2203.16248">Instance-Aware Image-to-Image Translation with Transformer</a> by Kim et al.

You can check out project at [[Project Page](https://KU-CVLAB.github.io/InstaFormer/)] and the paper on [[arXiv](https://arxiv.org/abs/2203.16248)].

### Environment
* Python 3.8, PyTorch 1.11.0


### Datasets
* INIT [[dataset]](https://zhiqiangshen.com/projects/INIT/index.html)

### Training & Test Script

- train

```python
train.py
```

- test

```python
test.py
```




### Config Path


config/base_train.yaml

config/base_test.yaml



### BibTeX
If you find this research useful, please consider citing:
````BibTeX
@inproceedings{kim2022instaformer,
  title={InstaFormer: Instance-Aware Image-to-Image Translation with Transformer},
  author={Kim, Soohyun and Baek, Jongbeom and Park, Jihye and Kim, Gyeongnyeon and Kim, Seungryong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18321--18331},
  year={2022}
}



