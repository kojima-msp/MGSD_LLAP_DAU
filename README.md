MGSD_LLap_DAU
====

## Abstract
> We propose a denoising method of multimodal graph signals by iteratively solving signal restoration and  graph learning problems. Many complex-structured data, i.e., those on sensor networks, can capture multiple modalities at each measurement point, referred to as *modalities*. They are also assumed to have an underlying structure or correlations in modality as well as space. Such multimodal data are regarded as graph signals on a *twofold graph* and they are often corrupted by noise. Furthermore, their spatial/modality relationships are not always given a priori: We need to estimate twofold graphs during a denoising algorithm. In this paper, we consider a signal denoising method on twofold graphs, where graphs are learned simultaneously. We formulate an optimization problem for that and parameters in an iterative algorithm are learned from training data by *unrolling* the iteration with deep algorithm unrolling. Experimental results on synthetic and real-world data demonstrate that the proposed method outperforms existing model- and deep learning-based graph signal denoising methods.

## Requirement
```bash
numpy==1.26.4
matplotlib==3.9.2
seaborn==0.13.2
scipy==1.13.1
sklearn==1.5.1
torch==2.5.1
pandas==2.2.2
scienceplots==2.1.1
skimage==0.24.0
```

## Install
```bash
git clone xxx
```

## Usage

### Execute to train & test models with `Synthetic`/`Weather` Dataset

```shell
python3 src/train.py Weather ae
python3 src/train.py Weather gcn
python3 src/train.py Weather TGSR_DAU
python3 src/train.py Weather MGSD_LLap_DAU

python3 src/test.py Weather ae
python3 src/test.py Weather gcn
python3 src/test.py Weather TGSR_DAU
python3 src/test.py Weather MGSD_LLap_DAU

python3 src/exe_existing.py Weather glpf
python3 src/exe_existing.py Weather svds
python3 src/exe_existing.py Weather hd
```

### Plot results
```shell
python3 results/plot_laplacian.py Synthetic 0 # idx is a target number of the model
python3 results/plot_parameters.py Synthetic
python3 results/plot_rmse_psnr.py Synthetic
python3 results/plot_signal.py Synthetic 0 # idx is a target number of the dataset

python3 results/plot_laplacian.py Weather 0
python3 results/plot_parameters.py Weather
python3 results/plot_rmse_psnr.py Weather
python3 results/plot_signal.py Weather 0
```

### Note
`ae` pretrained model size is over 100MB, so they can be downloaded [here]()

## Reference
[1] H. Kojima, K. Takanami, Junya Hara, Yukihiro Bandoh, Seishi Takamura, Hirhoshi Higashi, and Yuichi Tanaka, "Multimodal Graph Signal Denoising with Simultaneous Graph Learning," xxx, xxx.