# adi_action_segmentation_tcn
> This project was produced during an internship at [Analog Devices Inc](https://www.analog.com/en/index.html).

A low-cost Temporal Conv. Network (TCN) solution targeting action segmentation of videos from Kinetics dataset. This code can train and evaluate several models, reports logs and TensorBoard outputs. Most of the candidate models have seperate CNN layers (processing frames) and TCN structure on top (processing in time).

The challenges are:
- Target hardware is an edge-AI board, meaning it is low in processing power and capacity. Owing to target limitation, we implemented custom efficient PyTorch layers by [MaximIntegratedAI/ai8x-training](https://www.github.com/MaximIntegratedAI/ai8x-training).
- Kinetics dataset is challeging due to non-consistent photometric features and occasionally non-informative content.
- Video processing is always expensive, as it deals with 4-dimensions (RGB frames w/ time axis).

![image](https://user-images.githubusercontent.com/97564250/232345756-6ea6e340-8f65-4127-a4ff-afbc9b6d7b79.png)
_Fig. 1: Basic TCN structure [[ref]](/doc/paper/wavenet.pdf)_
## :books: Main Dependencies
- [ai8x-training](https://www.github.com/MaximIntegratedAI/ai8x-training)
- PyTorch v1.8.1
- torchvision v0.9.1
- NumPy, Pickle, PyYAML, [distiller](https://github.com/MaximIntegratedAI/distiller).

_Plus, all the dependencies indicated in ai8x repo._

## :wrench: Hardware
Trained models eventually targets Analog Devices [MAX78000FTHR](https://www.analog.com/en/design-center/evaluation-hardware-and-software/evaluation-boards-kits/max78000fthr.html). To make it compatible with the hardware, use layers presented in ai8x-training repo, enable QAT (Quantization Aware Training), then sythesize machine code following [this repo](https://github.com/MaximIntegratedAI/ai8x-synthesis) before deployment. 
