# MAPO: Advancing Multilingual Reasoning through Multilingual Alignment-as-Preference Optimization

*The code and models for this project will be released later this week.*

![](/fig/MAPO2.png)

# :trophy: Benchmarks
|        System          | [mSVAMP](https://huggingface.co/datasets/Mathoctopus/MSVAMP)| [mGSM](https://huggingface.co/datasets/juletxara/mgsm) |  [mNumGLUESub](https://huggingface.co/datasets/Mathoctopus/MSVAMP) | Download |
|:--------------------------:|:----:|:----:|:------:|:--------:|
| **MAPO-DPO(ours)** |        **58.4**    |      **41.1**                 | **49.8** |   [link](https://huggingface.co/kevinpro/MAPO-DPO-7B)   |
| MathOctopus |        40.7    |      38.1                 | 36.0 |   [link](https://huggingface.co/Mathoctopus/Parallel_7B)   |
| MultiLingual-RFT |        48.6    |      33.4                 | 44.8 |   [link](https://huggingface.co/kevinpro/MAPO-MultiLingual-RFT-Baseline)   |
| ChatGPT Zero-shot |        45.4    |      40.8                 | 40.7 |   -   |



## Overall Result on Out-Domain Benchmark: [mSVAMP](https://huggingface.co/datasets/Mathoctopus/MSVAMP) 

|  Model                        | Bn      | Th      | Sw      | Ja      | Zh      | Ru      | De      | Es      | Fr      | Avg.      | En |
|:--------------------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| **MAPO-DPO(ours)**         |  **48.8**  |   **55.2**  |   **56.0**  |   **60.3**  |   **58.8**  |   **58.3**  |  **58.1**  |   **59.7**  |   **60.8**  |   **57.3**  |  **58.4** | 
| MathOctopus|  27.7  | 35.9  | 39.4  | 41.6  | 42.7  | 44.2  | 44.0  | 45.1  | 45.3  | 40.7  | 46.4   |
| MultiLingual-RFT    | 37.9  | 46.4  | 46.4  | 49.6  | 50.8  | 50.4  | 50.7  | 51.6  | 53.4  | 48.6  | 49.4    |
|  ChatGPT Zero-shot         | 29.9  | 40.8  | 44.3  | 44.0  | 47.9  | 48.4  | 51.2  | 52.4  | 50.1  | 45.4  | 53.8   |

## Overall Result on In-Domain Benchmark: [mMGSM](https://huggingface.co/datasets/Mathoctopus/MSVAMP) 

|  Model                        | Bn      | Th      | Sw      | Ja      | Zh      | Ru      | De      | Es      | Fr      | Avg.      | En |
|:--------------------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| **MAPO-DPO(ours)**         |  **30.8**  | **38.0**  | **37.6**  | **45.2**  | **47.2**  | **42.0**  | **45.2**  | **43.2**  | **40.8**  | **41.1**  | 45.6 | 
| MathOctopus|  29.2 | 33.6 | 36.4 | 35.2 | 39.2 | 38.8 | 44.8 | 42.4 | 43. | 38.1 | **52.0**   |
| MultiLingual-RFT    | 25.6 | 31.2 | 28.8 | 34.0 | 39.2 | 36.0 | 34.8 | 34.4 | 36.4 | 33.4 | 43.2    |
|  ChatGPT Zero-shot         | 31.2  | 38.0  | 40.0  | 36.0  | 44.0  | 43.2  | 46.0  | 47.2  | 41.6  | 40.8  | 54.4

## Overall Result on In-Domain Benchmark: [mNumGLUESub](https://huggingface.co/datasets/Mathoctopus/MSVAMP) 

|  Model                        | Bn      | Th      | Sw      | Ja      | Zh      | Ru      | De      | Es      | Fr      | Avg.      | En |
|:--------------------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| **MAPO-DPO(ours)**         | **41.8** | **45.8** | **46.9** | **52.9** | **54.4** | **49.9** | **50.7** | **54.0** | **51.4** | **49.8** | **55.9** | 
| MathOctopus|   26.6 | 30.9 | 34.3 | 40.9 | 44.4 | 36.0 | 32.6 | 42.0 | 36.2 | 36.0 | 46.9 |
| MultiLingual-RFT    | 38.0 | 42.9 | 41.8 | 48.2 | 51.6 | 45.2 | 42.9 | 49.3 | 42.9 | 44.8 | 51.2    |
|  ChatGPT Zero-shot         | 36.2 | 42.6 | 47.2 | 58.1 | 60.6 | 42.6 | 41.5 | 54.9 | 39.4 | 47.0 | 70.6 |





### :trophy: Alignment Performance

<p float="left">
  <img src="/fig/Alignment.png" alt="Alt text for image 1" width="350" />
  <img src="/fig/ARC.png" alt="Alt text for image 2" width="350" />
</p>





### :hammer_and_wrench: Training & Evaluation

The code is on its way.

### Citation
If you find this repository helpful, feel free to cite our paper:
```
@misc{she2024mapo,
      title={MAPO: Advancing Multilingual Reasoning through Multilingual Alignment-as-Preference Optimization}, 
      author={Shuaijie She and Shujian Huang and Wei Zou and Wenhao Zhu and Xiang Liu and Xiang Geng and Jiajun Chen},
      year={2024},
      eprint={2401.06838},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```