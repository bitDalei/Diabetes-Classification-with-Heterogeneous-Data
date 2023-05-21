# Diabetes-Classification-with-Heterogeneous-Data

Official implementation for [A dual-attention based coupling network for diabetes classification with heterogeneous data(BI 2023)](https://www.sciencedirect.com/science/article/pii/S1532046423000217?casa_token=ihSFgzMaz7UAAAAA:bN4cmQRF6GsbM-MKWi7drP7omjG-m7AktF70BPvvF8HEfOceIW6Dm6DL0gTtPhkvbvavT7ifbUI)

<p align="center">
  <img src="https://github.com/bitDalei/Diabetes-Classification-with-Heterogeneous-Data/blob/main/others/graphical%20structure.png" width="650" alt="network structure">
</p>

### What is the purpose of the study?
Classification of two types of diabetes: T1DM and T2DM.

### What are the results?
Accuracy: **95.835%(SOTA)**

MCC: **91.333%**

F1-score: **94.939%**

G-mean: **94.937%**

### How does it works?
The networks takes into two distint type of data, FGM(Flash Glucose Monitoring) data(continuous) and Biomarkers data(discrete), for FGM data, LSTM with CBAM attention is carried out, for Biomarkers obtained from EMR(Electric Medical Record), CNN is construted. Feature maps from the two networks are integrated with self-attention and do further classification

### What's the data used?
Unfortunately, data cannot be made public due to confidential agreement. However, in order to make this implementation helpful for other researchers, a simulated data is provided. Note that all of the data is generated randomly but are in the same form as we actually used.
[training/testing data](https://huggingface.co/datasets/seidouz/Diabetes)
You can use 
```python
from datasets import load_dataset
dataset = load_dataset("seidouz/Diabetes")
```
to import them into your codes.

**Explanation**
- column 0: label
- column 1-576: FGM data
- column 576-587: Biomarkers data
  
  
You might notice some of consecutive rows have same biomarkers, this means that these few rows are contributed by the same patient. There are also some missing value in biomarkers, presented as '0'.

## Citing
Please cite us if our work has been involed.
```
@article{WANG2023104300,
title = {A dual-attention based coupling network for diabetes classification with heterogeneous data},
journal = {Journal of Biomedical Informatics},
volume = {139},
pages = {104300},
year = {2023},
issn = {1532-0464},
doi = {https://doi.org/10.1016/j.jbi.2023.104300},
url = {https://www.sciencedirect.com/science/article/pii/S1532046423000217},
author = {Lei Wang and Zhenglin Pan and Wei Liu and Junzheng Wang and Linong Ji and Dawei Shi},
keywords = {Diabetes types classification, Dual-attention, Coupling network, Heterogeneous data},
}
```
