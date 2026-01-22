MGSD_LLap_DAU
====



Official Pytorch implementation of the paper "Algorithm Unrolling-based Denoising of Multimodal Graph Signals" (submmited to IEEE TSIPN).

![Proposed Method](doc/img/proposed_method.jpg)

[![paper info](https://img.shields.io/badge/IEEE_TSIPN-Submitted-gray?labelColor=00629B)]()
[![arXiv](https://img.shields.io/badge/arXiv-2505.22175v2-gray?labelColor=b31b1b)](https://https://arxiv.org/abs/2505.22175v2)
[![Python](https://custom-icon-badges.herokuapp.com/badge/Python-3572A5?logo=Python&logoColor=white)]()
[![paper info](https://img.shields.io/badge/Our_Homepage-green)](https://www.sip.comm.eng.osaka-u.ac.jp/)

## Abstract
> We propose a denoising method for multimodal graph signals by an alternating minimization scheme that sequentially solves signal restoration and graph learning problems. Many complex-structured data, i.e., those on sensor networks, can capture multiple modalities at each measurement point, referred to as *modalities*. They are also assumed to have an underlying structure or correlations in modality as well as space. Such multimodal data are regarded as graph signals on a *twofold graph* and they are often corrupted by noise. Furthermore, their spatial/modality relationships are not always given a priori: We need to estimate twofold graphs during a denoising algorithm. In this paper, we consider a signal denoising method on twofold graphs, where graphs are learned simultaneously. Specifically, the graph learning subproblems are solved using the primal-dual splitting (PDS) algorithm, while the signal update has a closed-form solution. Parameters in this iterative algorithm are learned from training data by unrolling the iteration with deep algorithm unrolling. Experimental results on synthetic and real-world data demonstrate that the proposed method outperforms existing model- and deep learning-based graph signal denoising methods.

## Install
We use [uv](https://docs.astral.sh/uv/) to manage Python environment. 
Dataset and pretrained model using experiments can be downloaded [here](https://drive.google.com/drive/folders/1BrJmYvldVDsAms-1lzGqUOjOU1rwkQvb?usp=sharing). 
Please place `./data`!

```bash
git clone https://github.com/kojima-msp/MGSD_LLAP_DAU.git
cd MGSD_LLAP_DAU
uv sync
source .venv/bin/activate
```

## Usage

Yout can execute train & test models with `Synthetic`/`Weather` argument. 

### Train Models
```shell
#　Train existing and proposed method. 
python3 src/train.py Weather ae
python3 src/train.py Weather gcn
python3 src/train.py Weather TGSR_DAU
python3 src/train.py Weather MGSD_LLap_DAU
# Train for ablation study.
python3 src/train_ablation.py gt
python3 src/train_ablation.py rbf
python3 src/train_ablation.py ls
python3 src/train_ablation.py lm
```

### Test Models
```shell
# Test existing and proposed method.
python3 src/test.py Weather ae
python3 src/test.py Weather gcn
python3 src/test.py Weather TGSR_DAU
python3 src/test.py Weather MGSD_LLap_DAU
python3 src/exe_existing.py Weather glpf
python3 src/exe_existing.py Weather svds
python3 src/exe_existing.py Weather hd
```

### Print/Plot Results
```shell
# print Table. 2 and Table 4 in [1]
python3 results/print_rmse.py Weather
# plot Fig. 4 and Fig. 9 in [1]
python3 results/plot_signal.py Weather
# print Table. 3 in [1]
python3 src/print_ablation.py
# plot Fig. 5, Fig. 6, Fig. 10 and Fig. 11 in [1]
python3 results/plot_laplacian.py Weather
# plot Fig. 7 in [1]
python3 results/plot_parameters.py Weather
# plot Fig. 8 in [1]
python3 results/plot_layer_vs_rmse.py
```

## Citation
> H. Kojima, K. Takanami, Junya Hara, Yukihiro Bandoh, Seishi Takamura, Hirhoshi Higashi, and Yuichi Tanaka, "Algorithm Unrolling-based Denoising of Multimodal Graph Signals," xxx, xxx. 