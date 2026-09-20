# LG-PF: Lightweight Confidence-Guided Polarization Image Fusion

<p align="center">
  <b>Zhuangfan Huang</b>,
  Zhenyu Kuang,
  Huafeng Li,
  Gao Wang,
  Yang Liu,
  Haishu Tan,
  Xiaosong Li
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.12787">
    <img src="https://img.shields.io/badge/arXiv-2609.12787-b31b1b.svg">
  </a>
  <a href="https://pan.baidu.com/s/1FAuq250fF-A7OH4s8SJHoA?pwd=sw5c">
    <img src="https://img.shields.io/badge/Dataset-MSP-blue.svg">
  </a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.12787">arXiv</a> |
  <a href="#overview">Overview</a> |
  <a href="#network-architecture">Network</a> |
  <a href="https://pan.baidu.com/s/1FAuq250fF-A7OH4s8SJHoA?pwd=sw5c">Dataset</a> |
  <a href="#experimental-results">Results</a> |
  <a href="#installation">Installation</a> |
  <a href="#training">Training</a> |
  <a href="#testing">Testing</a>
</p>

---

# Overview

Polarization image fusion combines the stable luminance and structural information of the total-intensity image **S0** with the material-sensitive details contained in the degree of linear polarization (**DoLP**) image.

However, the informativeness of DoLP is spatially nonuniform. Indiscriminate polarization transfer may amplify unstable responses or disturb the structural appearance anchored by S0.

We propose **LG-PF**, a lightweight confidence-guided polarization image fusion framework that formulates polarization fusion as a **selective residual transfer process**.

LG-PF contains three key components:

- **Polarization Confidence Prior (PCP)**  
  Estimates spatially informative polarization responses.

- **Mask-guided Multi-scale Fusion (MMF)**  
  Regulates confidence-guided polarization transfer across three feature scales.

- **Lightweight Context-aware Bounded Correction Head (LCH)**  
  Performs lightweight bounded correction of local photometric and structural transitions.

The same confidence guidance is further incorporated into the optimization objectives to preserve informative polarization-sensitive details while suppressing unsupported responses.

We also construct **MSP**, a multi-scene polarization fusion dataset containing **1000 pixel-aligned S0–DoLP image pairs from 17 indoor and outdoor scene categories**.

---

# Network Architecture

<p align="center">
  <img src="./fig/fig1.png" width="95%">
</p>

<p align="center">
  <b>Fig. 1. Overall architecture of the proposed LG-PF.</b>
</p>

LG-PF follows a lightweight dual-stream fusion pipeline:

1. **Dual-stream lightweight encoding** of S0 and DoLP.
2. **PCP-based spatial confidence estimation**.
3. **MMF-based confidence-guided multi-scale polarization transfer**.
4. **Residual reconstruction and polarization detail refinement**.
5. **LCH-based bounded local correction**.

The total-intensity image S0 serves as the photometric and structural anchor, while complementary DoLP information is selectively introduced through confidence-guided residual transfer.

---

# Core Modules

## Polarization Confidence Prior (PCP)

PCP estimates a spatial polarization confidence mask by jointly considering:

- Local polarization-response prominence
- Fine-scale texture evidence
- Cross-modal gradient advantage of DoLP over S0

The three confidence cues are combined to generate a shared polarization confidence mask, which is subsequently used to regulate polarization information transfer throughout the network and guide optimization.

---

## Mask-guided Multi-scale Fusion (MMF)

MMF extracts three-scale features from the S0 and DoLP branches.

At each feature scale, the polarization feature is modulated by the resized confidence mask through a soft confidence-gating mechanism.

The minimum polarization retention coefficient is:

```text
eta = 0.35
```

This design preserves high-confidence polarization information while attenuating, rather than completely discarding, uncertain polarization responses.

The confidence-modulated DoLP features are subsequently combined with the corresponding S0 features through lightweight channel mixing. The fused multi-scale representations are progressively decoded to obtain a full-resolution representation for residual reconstruction and polarization detail refinement.

---

## Lightweight Context-aware Bounded Correction Head (LCH)

LCH performs lightweight local correction instead of reconstructing the complete fused image.

Its inputs include:

- S0
- Gradient magnitude of DoLP
- DoLP
- Refined fusion result F1
- Polarization confidence mask M

These five maps are processed through compact pointwise and depthwise convolutions to predict a signed correction map.

The correction magnitude is explicitly bounded using:

```text
lambda_c = 0.05
```

This design stabilizes local photometric and structural transitions while introducing only minor computational overhead.

---

# MSP Dataset

<p align="center">
  <a href="https://pan.baidu.com/s/1FAuq250fF-A7OH4s8SJHoA?pwd=sw5c">
    <img src="https://img.shields.io/badge/Download-MSP--dataset.zip-blue.svg">
  </a>
</p>

<p align="center">
  <b>MSP Dataset:</b>
  <a href="https://pan.baidu.com/s/1FAuq250fF-A7OH4s8SJHoA?pwd=sw5c">Baidu Netdisk</a>
  &nbsp;|&nbsp;
  <b>Password:</b> <code>sw5c</code>
</p>

MSP contains **1000 pixel-aligned S0–DoLP image pairs** from **17 indoor and outdoor scene categories**, including **872 outdoor pairs** and **128 indoor pairs**, with an image resolution of **1125 × 938**. The dataset is divided into **900 training**, **50 validation**, and **50 testing** pairs.

<p align="center">
  <img src="./fig/fig2.png" width="92%">
</p>

<p align="center">
  <b>Fig. 2.</b> Acquisition systems and composition of the MSP dataset, including the DoFP imaging process, four-direction polarization observations, derived S0 and DoLP images, and representative indoor and outdoor scenes.
</p>

## Table I. Comparison with Existing Polarization Fusion Datasets

<div align="center">

<table>
<tr>
<th>Dataset</th>
<th>Acquisition Pattern</th>
<th>Total Pairs</th>
<th>Indoor Pairs</th>
<th>Outdoor Pairs</th>
<th>Scene Categories</th>
<th>Image Size</th>
</tr>

<tr>
<td>PIF</td>
<td>DoFP</td>
<td>74</td>
<td>5</td>
<td>69</td>
<td>N/R</td>
<td>1024 × 1224</td>
</tr>

<tr>
<td>GAND</td>
<td>DoFP</td>
<td>415</td>
<td>331</td>
<td>84</td>
<td>N/R</td>
<td>768 × 576</td>
</tr>

<tr>
<td><b>MSP (Proposed)</b></td>
<td><b>DoFP</b></td>
<td><b>1000</b></td>
<td><b>128</b></td>
<td><b>872</b></td>
<td><b>17</b></td>
<td><b>1125 × 938</b></td>
</tr>

</table>

</div>

---

# Installation

The source code and detailed environment configuration will be released in this repository.

After the code is released, install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

# Training

The MSP dataset split used in the paper is:

```text
Training   : 900 image pairs
Validation : 50 image pairs
Testing    : 50 image pairs
```

Before training, please modify the dataset path according to your local environment.

```bash
# The exact training command will be updated together with the released code.
python train.py
```

The model is trained from scratch on the MSP training set.

The main training configuration reported in the paper includes:

```text
Optimizer             : AdamW
Batch size            : 16
Initial learning rate : 5e-5
Weight decay          : 5e-5
Training epochs       : 175
Selected checkpoint   : Epoch 130
Base channel width    : 16
GPU                   : NVIDIA GeForce RTX 3090
```

---

# Testing

LG-PF is evaluated on:

- MSP test set
- PIF fixed subset
- GAND fixed subset

For the PIF and GAND evaluations, LG-PF is directly evaluated **without fine-tuning or domain adaptation**.

```bash
# The exact testing command will be updated together with the released code.
python test.py
```

---

# Experimental Results

<p align="center">
  🔴 <b>Best</b>
  &nbsp;&nbsp;&nbsp;&nbsp;
  🔵 <b>Second Best</b>
</p>

---

## Table II. Quantitative Comparison on the MSP Test Set

<div align="center">

<table>
<tr>
<th>Method</th>
<th>EN ↑</th>
<th>SF ↑</th>
<th>SD ↑</th>
<th>SCD ↑</th>
<th>MS-SSIM ↑</th>
<th>Q<sub>CB</sub> ↑</th>
</tr>

<tr>
<td><b>LG-PF</b></td>
<td>🔴 <b>6.5485</b></td>
<td>🔴 <b>15.6979</b></td>
<td>🔴 <b>46.0210</b></td>
<td>🔴 <b>1.6632</b></td>
<td>🔴 <b>0.9439</b></td>
<td>🔴 <b>0.3099</b></td>
</tr>

<tr>
<td>CPIFuse</td>
<td>6.3884</td>
<td>8.9657</td>
<td>40.8085</td>
<td>1.4867</td>
<td>0.9296</td>
<td>0.3020</td>
</tr>

<tr>
<td>DT-F</td>
<td>🔵 <b>6.5269</b></td>
<td>🔵 <b>15.5525</b></td>
<td>40.4660</td>
<td>🔵 <b>1.6563</b></td>
<td>0.7926</td>
<td>0.2621</td>
</tr>

<tr>
<td>LFDT</td>
<td>6.5263</td>
<td>12.1124</td>
<td>🔵 <b>45.5976</b></td>
<td>1.6251</td>
<td>🔵 <b>0.9387</b></td>
<td>🔵 <b>0.3074</b></td>
</tr>

<tr>
<td>PAPIF</td>
<td>6.4486</td>
<td>10.8050</td>
<td>42.7253</td>
<td>1.5935</td>
<td>0.9057</td>
<td>0.2892</td>
</tr>

<tr>
<td>PIPFNet</td>
<td>6.3281</td>
<td>8.0722</td>
<td>39.0078</td>
<td>1.1774</td>
<td>0.7650</td>
<td>0.2727</td>
</tr>

<tr>
<td>TIPFNet</td>
<td>6.3078</td>
<td>10.8383</td>
<td>38.0155</td>
<td>1.5136</td>
<td>0.8252</td>
<td>0.2996</td>
</tr>

</table>

</div>

LG-PF achieves the best performance across all six evaluated metrics on the MSP test set.

---

## Qualitative Comparison on MSP

<p align="center">
  <img src="./fig/fig3.png" width="100%">
</p>

<p align="center">
  <b>Fig. 3.</b> Qualitative comparison on representative samples from the MSP test set.
</p>

LG-PF selectively preserves polarization-sensitive details while maintaining the photometric and structural information anchored by S0.

---

# Cross-Dataset Evaluation

To investigate cross-dataset transferability, the trained LG-PF model is directly evaluated on fixed subsets of **PIF** and **GAND** without fine-tuning or domain adaptation.

---

## Table III. Quantitative Comparison on the PIF Dataset

<div align="center">

<table>
<tr>
<th>Method</th>
<th>EN ↑</th>
<th>SF ↑</th>
<th>SD ↑</th>
<th>SCD ↑</th>
<th>MS-SSIM ↑</th>
<th>Q<sub>CB</sub> ↑</th>
</tr>

<tr>
<td><b>LG-PF</b></td>
<td>🔴 <b>7.4989</b></td>
<td>🔴 <b>23.2688</b></td>
<td>🔴 <b>63.8530</b></td>
<td>1.5917</td>
<td>🔴 <b>0.9214</b></td>
<td>🔵 <b>0.4562</b></td>
</tr>

<tr>
<td>CPIFuse</td>
<td>7.3479</td>
<td>13.8353</td>
<td>55.0505</td>
<td>1.5699</td>
<td>0.9155</td>
<td>0.4165</td>
</tr>

<tr>
<td>DT-F</td>
<td>7.3064</td>
<td>🔵 <b>21.7685</b></td>
<td>52.4917</td>
<td>🔴 <b>1.7609</b></td>
<td>0.8166</td>
<td>0.3603</td>
</tr>

<tr>
<td>LFDT</td>
<td>🔵 <b>7.4509</b></td>
<td>17.8186</td>
<td>🔵 <b>62.7805</b></td>
<td>1.5718</td>
<td>🔵 <b>0.9175</b></td>
<td>🔴 <b>0.4728</b></td>
</tr>

<tr>
<td>PAPIF</td>
<td>7.4261</td>
<td>17.0269</td>
<td>60.0223</td>
<td>🔵 <b>1.6770</b></td>
<td>0.9140</td>
<td>0.4455</td>
</tr>

<tr>
<td>PIPFNet</td>
<td>7.2163</td>
<td>13.1146</td>
<td>55.7901</td>
<td>1.4345</td>
<td>0.8443</td>
<td>0.3887</td>
</tr>

<tr>
<td>TIPFNet</td>
<td>7.2585</td>
<td>15.7598</td>
<td>52.8156</td>
<td>1.5841</td>
<td>0.8583</td>
<td>0.4494</td>
</tr>

</table>

</div>

### Qualitative Comparison on PIF

<p align="center">
  <img src="./fig/fig4.png" width="100%">
</p>

<p align="center">
  <b>Fig. 4.</b> Qualitative comparison on randomly selected PIF samples without fine-tuning.
</p>

LG-PF preserves material-sensitive textures and object boundaries while maintaining the overall luminance distribution of S0.

---

## Table IV. Quantitative Comparison on the GAND Dataset

<div align="center">

<table>
<tr>
<th>Method</th>
<th>EN ↑</th>
<th>SF ↑</th>
<th>SD ↑</th>
<th>SCD ↑</th>
<th>MS-SSIM ↑</th>
<th>Q<sub>CB</sub> ↑</th>
</tr>

<tr>
<td><b>LG-PF</b></td>
<td>🔴 <b>6.9700</b></td>
<td>🔴 <b>17.6705</b></td>
<td>🔴 <b>39.1293</b></td>
<td>1.7269</td>
<td>0.7335</td>
<td>🔵 <b>0.5290</b></td>
</tr>

<tr>
<td>CPIFuse</td>
<td>6.8694</td>
<td>11.3982</td>
<td>36.9481</td>
<td>1.7230</td>
<td>🔴 <b>0.9430</b></td>
<td>🔴 <b>0.5883</b></td>
</tr>

<tr>
<td>DT-F</td>
<td>6.8655</td>
<td>🔵 <b>16.6782</b></td>
<td>29.9436</td>
<td>🔴 <b>1.8802</b></td>
<td>0.6180</td>
<td>0.3696</td>
</tr>

<tr>
<td>LFDT</td>
<td>6.8695</td>
<td>14.7905</td>
<td>38.2868</td>
<td>1.7196</td>
<td>0.5697</td>
<td>0.4923</td>
</tr>

<tr>
<td>PAPIF</td>
<td>🔵 <b>6.9613</b></td>
<td>14.5522</td>
<td>🔵 <b>38.8881</b></td>
<td>1.7233</td>
<td>0.6147</td>
<td>0.4716</td>
</tr>

<tr>
<td>PIPFNet</td>
<td>6.8056</td>
<td>13.4443</td>
<td>36.9769</td>
<td>1.6107</td>
<td>🔵 <b>0.8007</b></td>
<td>0.5078</td>
</tr>

<tr>
<td>TIPFNet</td>
<td>6.6308</td>
<td>12.8721</td>
<td>31.6894</td>
<td>🔵 <b>1.8365</b></td>
<td>0.7013</td>
<td>0.5261</td>
</tr>

</table>

</div>

### Qualitative Comparison on GAND

<p align="center">
  <img src="./fig/fig5.png" width="100%">
</p>

<p align="center">
  <b>Fig. 5.</b> Qualitative comparison on a randomly selected GAND sample without fine-tuning.
</p>

The evaluations on PIF and GAND provide promising evidence of cross-dataset transferability without fine-tuning or domain adaptation.

---

# Ablation Study

## Table V. Ablation Study of Key Modules

<div align="center">

<table>
<tr>
<th>Method</th>
<th>EN ↑</th>
<th>SF ↑</th>
<th>SD ↑</th>
<th>SCD ↑</th>
<th>MS-SSIM ↑</th>
<th>Q<sub>CB</sub> ↑</th>
</tr>

<tr>
<td>Full Model</td>
<td>🔴 <b>6.9094</b></td>
<td>15.5907</td>
<td>🔵 <b>56.3164</b></td>
<td>🔴 <b>1.7211</b></td>
<td>🔵 <b>0.9466</b></td>
<td>0.4165</td>
</tr>

<tr>
<td>w/o PCP</td>
<td>6.8752</td>
<td>🔵 <b>16.5595</b></td>
<td>🔴 <b>56.3806</b></td>
<td>1.7019</td>
<td>0.9251</td>
<td>0.4119</td>
</tr>

<tr>
<td>w/o MMF</td>
<td>6.8931</td>
<td>15.3270</td>
<td>55.8327</td>
<td>🔵 <b>1.7063</b></td>
<td>🔴 <b>0.9475</b></td>
<td>🔴 <b>0.4224</b></td>
</tr>

<tr>
<td>w/o LCH</td>
<td>🔵 <b>6.8962</b></td>
<td>15.1074</td>
<td>56.2214</td>
<td>1.7054</td>
<td>0.9403</td>
<td>🔵 <b>0.4190</b></td>
</tr>

<tr>
<td>Baseline</td>
<td>6.8653</td>
<td>🔴 <b>16.8300</b></td>
<td>56.1553</td>
<td>1.7012</td>
<td>0.9233</td>
<td>0.4129</td>
</tr>

</table>

</div>

Removing individual components results in different trade-offs among information content, structural fidelity, and perceptual quality. The complete LG-PF model provides the most consistent overall performance across the evaluated metrics.

---

## Table VI. Ablation Study of Loss Configurations

<div align="center">

<table>
<tr>
<th>Setting</th>
<th>EN ↑</th>
<th>SF ↑</th>
<th>SD ↑</th>
<th>SCD ↑</th>
<th>MS-SSIM ↑</th>
<th>Q<sub>CB</sub> ↑</th>
</tr>

<tr>
<td>Full Loss</td>
<td>🔴 <b>6.9094</b></td>
<td>15.5907</td>
<td>56.3164</td>
<td>🔴 <b>1.7211</b></td>
<td>🔵 <b>0.9466</b></td>
<td>🔴 <b>0.4165</b></td>
</tr>

<tr>
<td>w/o LREG</td>
<td>6.8767</td>
<td>🔵 <b>15.6931</b></td>
<td>56.1723</td>
<td>1.6984</td>
<td>🔴 <b>0.9535</b></td>
<td>0.4119</td>
</tr>

<tr>
<td>w/o LSF</td>
<td>🔵 <b>6.8942</b></td>
<td>🔴 <b>15.7950</b></td>
<td>🔵 <b>56.7184</b></td>
<td>🔵 <b>1.7093</b></td>
<td>0.9230</td>
<td>0.4147</td>
</tr>

<tr>
<td>w/o LTC</td>
<td>6.7902</td>
<td>15.3528</td>
<td>🔴 <b>56.7781</b></td>
<td>1.3409</td>
<td>0.8917</td>
<td>0.4107</td>
</tr>

<tr>
<td>LTC with PAPIF only</td>
<td>6.8869</td>
<td>15.5807</td>
<td>55.4955</td>
<td>1.6954</td>
<td>0.9316</td>
<td>0.4005</td>
</tr>

<tr>
<td>LTC with LFDT only</td>
<td>6.7906</td>
<td>14.9664</td>
<td>55.9281</td>
<td>1.6028</td>
<td>0.9415</td>
<td>🔵 <b>0.4160</b></td>
</tr>

</table>

</div>

The loss ablation results demonstrate the complementary roles of structure-frequency constraints, regularization, and teacher-consistency guidance.

---

# Hyperparameter Sensitivity

<p align="center">
  <img src="./fig/fig6.png" width="95%">
</p>

<p align="center">
  <b>Fig. 6.</b> Sensitivity analysis of the three group-level loss weights.
</p>

The final loss-weight configuration adopted in LG-PF is:

```text
lambda_sf  = 0.42
lambda_reg = 0.66
lambda_tc  = 1.00
```

The evaluated metrics remain relatively stable under moderate variations of these loss weights.

---

# Computational Efficiency

## Table VII. Comparison of Computational Efficiency

<div align="center">

<table>
<tr>
<th>Method</th>
<th>FLOPs (G) ↓</th>
<th>Parameters (M) ↓</th>
<th>Time (ms) ↓</th>
</tr>

<tr>
<td><b>LG-PF</b></td>
<td>🔴 <b>51.9883</b></td>
<td>0.2936</td>
<td>🔴 <b>21.7120</b></td>
</tr>

<tr>
<td>CPIFuse</td>
<td>173.8998</td>
<td>0.8740</td>
<td>🔵 <b>110.8920</b></td>
</tr>

<tr>
<td>LFDT</td>
<td>1936.6296</td>
<td>17.8980</td>
<td>692.8200</td>
</tr>

<tr>
<td>PAPIF</td>
<td>287.4564</td>
<td>🔵 <b>0.2609</b></td>
<td>323.3230</td>
</tr>

<tr>
<td>PIPFNet</td>
<td>🔵 <b>76.3328</b></td>
<td>🔴 <b>0.0343</b></td>
<td>288.6660</td>
</tr>

<tr>
<td>TIPFNet</td>
<td>979.7782</td>
<td>4.1509</td>
<td>485.6140</td>
</tr>

<tr>
<td>DT-F</td>
<td>737.8300</td>
<td>0.8344</td>
<td>4951.9010</td>
</tr>

</table>

</div>

Under an input resolution of **1125 × 938** and the reported RTX 3090 setting, LG-PF requires only **0.2936 M parameters** and **51.9883 G FLOPs**, with an inference time of **21.712 ms per image**.

LG-PF achieves the lowest computational complexity and fastest inference time among the evaluated methods while maintaining a compact model size.

---

# Citation

If you find **LG-PF** or the **MSP dataset** useful in your research, please cite our work:

**Paper:** LG-PF: Lightweight Confidence-Guided Polarization Image Fusion  
**arXiv:** [https://arxiv.org/abs/2609.12787](https://arxiv.org/abs/2609.12787)

```bibtex
@article{huang2026lgpf,
  title   = {LG-PF: Lightweight Confidence-Guided Polarization Image Fusion},
  author  = {Huang, Zhuangfan and Kuang, Zhenyu and Li, Huafeng and Wang, Gao and Liu, Yang and Tan, Haishu and Li, Xiaosong},
  journal = {arXiv preprint arXiv:2609.12787},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.12787}
}
```

---

# Acknowledgements

We sincerely thank the authors of the public datasets and comparison methods used in this work, including **PIF, GAND, CPIFuse, DT-F, LFDT-Fusion, PAPIF, PIPFNet, and TIPFNet**.

---

# Contact

For questions regarding the paper, source code, pretrained models, or the MSP dataset, please open an issue in this repository.
