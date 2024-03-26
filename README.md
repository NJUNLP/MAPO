# MAPO: Advancing Multilingual Reasoning through Multilingual Alignment-as-Preference Optimization
<p align="center">
  <a href="https://arxiv.org/abs/2401.06838"> 📃 Paper</a> | 
  <a href="https://huggingface.co/kevinpro"> 🤗 Huggingface</a> | 
  <a href="https://ricardokevins.github.io/"> 📭 Contact</a> 
</p>

## Overview

Welcome to the repository of MAPO, our cutting-edge framework designed to revolutionize multilingual reasoning capabilities in large language models (LLMs). 

* 🚀 We propose a framework that enhances the reasoning multilingual reasoning capabilities by aligning reasoning processes of other languages with those of English. We use off-the-shelf translation models to estimate the alignment of reasoning processes in other languages, and then optimize this alignment as a preference using popular preference optimization methods such as DPO or PPO.

* 📈 By utilizing our framework, you can effectively improve the consistency of multilingual reasoning, thereby enhancing the multilingual reasoning capabilities of large models in a more generalizable manner. Our approach has achieved impressive performance improvements, surpassing all baselines, including ChatGPT, and has reached state-of-the-art (SOTA) results.

* 🌐 Overall, our method demonstrates a novel way of improving the multilingual reasoning abilities of models without the need for extensive annotation of reasoning processes in other languages,  enabling a more generalizable enhancement of multilingual reasoning capabilities.














![](/fig/Alignv2.png)






## :trophy: Benchmarks

Below is the average accuracy across ten languages on three multilingual mathematical reasoning datasets . Our method significantly improves the multilingual reasoning capabilities of LLMs by a large margin, achieving the SOTA performance. We also hope that in the future, more multilingual reasoning LLMs can collaborate with our work to further enhance multilingual reasoning capabilities.

<table>
    <thead>
        <tr>
            <th>System</th>
            <th><a href="https://huggingface.co/datasets/Mathoctopus/MSVAMP">MSVAMP</a></th>
            <th><a href="https://huggingface.co/datasets/juletxara/mgsm">MGSM</a></th>
            <th><a href="https://huggingface.co/datasets/Mathoctopus/MSVAMP">MNumGLUESub</a></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>GPT-3.5-Turbo</td>
            <td style="text-align: center;">46.6</td>
            <td style="text-align: center;">42.2</td>
            <td style="text-align: center;">49.4</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/TIGER-Lab/MAmmoTH-7B">MAmmoTH 7B</a></td>
            <td style="text-align: center;">26.3</td>
            <td style="text-align: center;">21.3</td>
            <td style="text-align: center;">24.2</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/WizardLM/WizardMath-7B-V1.1">WizardMath 7B</a></td>
            <td style="text-align: center;">32.5</td>
            <td style="text-align: center;">23.0</td>
            <td style="text-align: center;">28.7</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/meta-math/MetaMath-7B-V1.0">MetaMath 7B</a></td>
            <td style="text-align: center;">46.2</td>
            <td style="text-align: center;">37.0</td>
            <td style="text-align: center;">43.2</td>
        </tr>
        <!-- <tr>
           <td colspan="5" style="text-align: center;"> MathOctopus 7B</td>
        </tr> -->
        <tr>
            <td><a href="https://huggingface.co/Wenhao97/QAlign-MetaMathQA-7B">QAlign 7B</a></td>
            <td style="text-align: center;">57.2</td>
            <td style="text-align: center;">49.6</td>
            <td style="text-align: center;">-</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/Mathoctopus/Parallel_7B">MathOctopus 7B</a></td>
            <td style="text-align: center;">41.2</td>
            <td style="text-align: center;">39.5</td>
            <td style="text-align: center;">37.1</td>
        </tr>
        <!-- <tr> (ours)
            <td>+ m-RFT</td>
            <td style="text-align: center;">48.7</td>
            <td style="text-align: center;">34.4</td>
            <td style="text-align: center;">45.4</td>
        </tr> -->
        <tr>
            <td><strong><a href="https://huggingface.co/kevinpro/MathOctopus-MAPO-DPO-7B">+ MAPO-DPO(ours)</a>🔥</strong></td>
            <td style="text-align: center;"><strong>57.4</strong></td>
            <td style="text-align: center;"><strong>41.6</strong></td>
            <td style="text-align: center;"><strong>50.4</strong></td>
        </tr> 
        <!-- <tr>
           <td colspan="5" style="text-align: center;">MetaMathOctopus 7B</td>
        </tr> -->
        <tr>
            <td><a href="https://huggingface.co/kevinpro/MetaMathOctopus-7B">MetaMathOctopus 7B</a></td>
            <td style="text-align: center;">53.0</td>
            <td style="text-align: center;">45.5</td>
            <td style="text-align: center;">39.2</td>
        </tr>
        <!-- <tr>
            <td>+ m-RFT</td>
            <td style="text-align: center;">56.7</td>
            <td style="text-align: center;">41.4</td>
            <td style="text-align: center;">51.7</td>
        </tr> -->
        <tr>
           <td><strong><a href="https://huggingface.co/kevinpro/MetaMathOctopus-MAPO-DPO-7B">+ MAPO-DPO(ours)</a> 👑</strong></td>
            <td style="text-align: center;"><strong>64.7</strong></td>
            <td style="text-align: center;"><strong>51.6</strong></td>
            <td style="text-align: center;"><strong>52.9</strong></td>
        </tr>
                <tr>
            <td><a href="https://huggingface.co/kevinpro/MistralMathOctopus-7B">MistralMathOctopus 7B</a></td>
            <td style="text-align: center;">59.0</td>
            <td style="text-align: center;">58.0</td>
            <td style="text-align: center;">56.8</td>
        </tr>
        <!-- <tr>
            <td>+ m-RFT</td>
            <td style="text-align: center;">56.7</td>
            <td style="text-align: center;">41.4</td>
            <td style="text-align: center;">51.7</td>
        </tr> -->
        <tr>
           <td><strong><a href="https://huggingface.co/kevinpro/MistralMathOctopus-MAPO-DPO-7B">+ MAPO-DPO(ours)</a> 👑</strong></td>
            <td style="text-align: center;"><strong>74.6</strong></td>
            <td style="text-align: center;"><strong>67.3</strong></td>
            <td style="text-align: center;"><strong>70.0</strong></td>
        </tr>
    </tbody>
</table>


<table>
    <thead>
        <tr>
            <th>System</th>
            <th><a href="https://huggingface.co/datasets/Mathoctopus/MSVAMP">MSVAMP</a></th>
            <th><a href="https://huggingface.co/datasets/juletxara/mgsm">MGSM</a></th>
            <th><a href="https://huggingface.co/datasets/Mathoctopus/MSVAMP">MNumGLUESub</a></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>GPT-3.5-Turbo</td>
            <td style="text-align: center;">46.6</td>
            <td style="text-align: center;">42.2</td>
            <td style="text-align: center;">49.4</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/TIGER-Lab/MAmmoTH-13B">MAmmoTH 13B</a></td>
            <td style="text-align: center;">38.6</td>
            <td style="text-align: center;">28.9</td>
            <td style="text-align: center;">29.5</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/WizardLM/WizardMath-13B-V1.1">WizardMath 13B</a></td>
            <td style="text-align: center;">35.7</td>
            <td style="text-align: center;">28.3</td>
            <td style="text-align: center;">29.0</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/meta-math/MetaMath-13B-V1.0">MetaMath 13B</a></td>
            <td style="text-align: center;">46.2</td>
            <td style="text-align: center;">43.9</td>
            <td style="text-align: center;">43.3</td>
        </tr>
        <!-- <tr>
           <td colspan="5" style="text-align: center;"> MathOctopus 7B</td>
        </tr> -->
                <tr>
            <td><a href="https://huggingface.co/Wenhao97/QAlign-MetaMathQA-13B">QAlign 13B</a></td>
            <td style="text-align: center;">62.6</td>
            <td style="text-align: center;">57.1</td>
            <td style="text-align: center;">-</td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/Mathoctopus/Parallel_13B">MathOctopus 13B</a></td>
            <td style="text-align: center;">51.8</td>
            <td style="text-align: center;">46.0</td>
            <td style="text-align: center;">40.3</td>
        </tr>
        <!-- <tr>
            <td>+ m-RFT</td>
            <td style="text-align: center;">48.7</td>
            <td style="text-align: center;">34.4</td>
            <td style="text-align: center;">45.4</td>
        </tr> -->
        <tr>
            <td><strong><a href="https://huggingface.co/kevinpro/MathOctopus-MAPO-DPO-13B">+ MAPO-DPO(ours)</a>🔥</strong></td>
            <td style="text-align: center;"><strong>60.1</strong></td>
            <td style="text-align: center;"><strong>48.5</strong></td>
            <td style="text-align: center;"><strong>53.8</strong></td>
        </tr> 
        <!-- <tr>
           <td colspan="5" style="text-align: center;">MetaMathOctopus 7B</td>
        </tr> -->
        <tr>
            <td><a href="https://huggingface.co/kevinpro/MetaMathOctopus-13B">MetaMathOctopus 13B</a></td>
            <td style="text-align: center;">56.3</td>
            <td style="text-align: center;">51.4</td>
            <td style="text-align: center;">49.5</td>
        </tr>
        <!-- <tr>
            <td>+ m-RFT</td>
            <td style="text-align: center;">56.7</td>
            <td style="text-align: center;">41.4</td>
            <td style="text-align: center;">51.7</td>
        </tr> -->
        <tr>
           <td><strong><a href="https://huggingface.co/kevinpro/MetaMathOctopus-MAPO-DPO-13B">+ MAPO-DPO(ours)</a> 👑</strong></td>
            <td style="text-align: center;"><strong>67.0</strong></td>
            <td style="text-align: center;"><strong>58.0</strong></td>
            <td style="text-align: center;"><strong>59.8</strong></td>
        </tr>
    </tbody>
</table>


<!-- |        System          | [mSVAMP](https://huggingface.co/datasets/Mathoctopus/MSVAMP)| [mGSM](https://huggingface.co/datasets/juletxara/mgsm) |  [mNumGLUESub](https://huggingface.co/datasets/Mathoctopus/MSVAMP) | Download |
|--------------------------|:----:|:----:|:------:|:--------:|
| ChatGPT Zero-shot |        46.6    |      42.2                 | 49.4 |   -   |
| MathOctopus 7B |        41.2    |      39.5                 | 37.1 |   [link](https://huggingface.co/Mathoctopus/Parallel_7B)   |
| + MultiLingual-RFT |        48.7    |      34.4                 | 45.4 |   [link](https://huggingface.co/kevinpro/MAPO-MultiLingual-RFT-Baseline)   |
| **+ MAPO-DPO(ours)** |        **57.4**    |      **41.6**                 | **50.4** |   [link](https://huggingface.co/kevinpro/MAPO-DPO-7B)   |
|-------|-------|-------|
| MetaMathOctopus 7B |         53.0    |      45.5                 | 39.2 |   [link](https://huggingface.co/Mathoctopus/Parallel_7B)   |
| + MultiLingual-RFT |        56.7    |      41.4                 | 51.7 |   [link](https://huggingface.co/kevinpro/MAPO-MultiLingual-RFT-Baseline)   |
| **+ MAPO-DPO(ours)** |        **64.7**    |      **51.6**                 | **52.9** |   [link](https://huggingface.co/kevinpro/MAPO-DPO-7B)   | -->





<!-- ## Overall Result on Out-Domain Benchmark: [mSVAMP](https://huggingface.co/datasets/Mathoctopus/MSVAMP)  -->

<!-- |  Model                        | Bn      | Th      | Sw      | Ja      | Zh      | Ru      | De      | Es      | Fr      | Avg.      | En |
|:--------------------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| **MAPO-DPO(ours)**         |  **48.8**  |   **55.2**  |   **56.0**  |   **60.3**  |   **58.8**  |   **58.3**  |  **58.1**  |   **59.7**  |   **60.8**  |   **57.3**  |  **58.4** | 
| MathOctopus|  27.7  | 35.9  | 39.4  | 41.6  | 42.7  | 44.2  | 44.0  | 45.1  | 45.3  | 40.7  | 46.4   |
| MultiLingual-RFT    | 37.9  | 46.4  | 46.4  | 49.6  | 50.8  | 50.4  | 50.7  | 51.6  | 53.4  | 48.6  | 49.4    |
|  ChatGPT Zero-shot         | 29.9  | 40.8  | 44.3  | 44.0  | 47.9  | 48.4  | 51.2  | 52.4  | 50.1  | 45.4  | 53.8   | -->

<!-- ## Overall Result on In-Domain Benchmark: [mMGSM](https://huggingface.co/datasets/Mathoctopus/MSVAMP)  -->

<!-- |  Model                        | Bn      | Th      | Sw      | Ja      | Zh      | Ru      | De      | Es      | Fr      | Avg.      | En |
|:--------------------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| **MAPO-DPO(ours)**         |  **30.8**  | **38.0**  | **37.6**  | **45.2**  | **47.2**  | **42.0**  | **45.2**  | **43.2**  | **40.8**  | **41.1**  | 45.6 | 
| MathOctopus|  29.2 | 33.6 | 36.4 | 35.2 | 39.2 | 38.8 | 44.8 | 42.4 | 43. | 38.1 | **52.0**   |
| MultiLingual-RFT    | 25.6 | 31.2 | 28.8 | 34.0 | 39.2 | 36.0 | 34.8 | 34.4 | 36.4 | 33.4 | 43.2    |
|  ChatGPT Zero-shot         | 31.2  | 38.0  | 40.0  | 36.0  | 44.0  | 43.2  | 46.0  | 47.2  | 41.6  | 40.8  | 54.4 -->

<!-- ## Overall Result on In-Domain Benchmark: [mNumGLUESub](https://huggingface.co/datasets/Mathoctopus/MSVAMP)  -->

<!-- |  Model                        | Bn      | Th      | Sw      | Ja      | Zh      | Ru      | De      | Es      | Fr      | Avg.      | En |
|:--------------------------------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|:--------|
| **MAPO-DPO(ours)**         | **41.8** | **45.8** | **46.9** | **52.9** | **54.4** | **49.9** | **50.7** | **54.0** | **51.4** | **49.8** | **55.9** | 
| MathOctopus|   26.6 | 30.9 | 34.3 | 40.9 | 44.4 | 36.0 | 32.6 | 42.0 | 36.2 | 36.0 | 46.9 |
| MultiLingual-RFT    | 38.0 | 42.9 | 41.8 | 48.2 | 51.6 | 45.2 | 42.9 | 49.3 | 42.9 | 44.8 | 51.2    |
|  ChatGPT Zero-shot         | 36.2 | 42.6 | 47.2 | 58.1 | 60.6 | 42.6 | 41.5 | 54.9 | 39.4 | 47.0 | 70.6 | -->





# :trophy: Alignment Performance

<p float="left">
  <img src="/fig/Alignment.png" alt="Alt text for image 1" width="350" />
  <img src="/fig/ARC.png" alt="Alt text for image 2" width="350" />
</p>

We report PPL-based alignment score (left) and ACR (right), respectively assessing the consistency of the reasoning process and the reasoning answer. MAPO achieves significant improvements in the consistency of both the reasoning processes and the reasoning answers of LLM across various languages.





## :hammer_and_wrench: Training & Evaluation
- Preference optimization data preparation
  - Generation: bash sampling.sh
  - Preference estimation: bash PreferenceEstimate.sh
  - Format paired data: python3 extract_dpo_data.py

- Training: 
  - DPO: bash dpo.sh/dpo13b.sh yourconfig.json
  - PPO: bash ppo_lora.sh yourconfig.json

- Evaluation: bash run.sh

For more details about training/evaluating, please navigate to the Alignment/Evaluation directory.

# Citation
If you find this repository helpful, feel free to cite our paper:
```
@misc{she2024mapo,
      title={MAPO: Advancing Multilingual Reasoning through Multilingual Alignment-as-Preference Optimization}, 
      author={Shuaijie She and Wei Zou and Shujian Huang and Wenhao Zhu and Xiang Liu and Xiang Geng and Jiajun Chen},
      year={2024},
      eprint={2401.06838},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


