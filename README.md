## MAPO: Advancing Multilingual Reasoning through Multilingual Alignment-as-Preference Optimization

*The code and models for this project will be released later this week.*

![](/fig/MAPO.png)

### :trophy: Benchmarks

|        System (7B)         | [mGSM](https://huggingface.co/datasets/juletxara/mgsm) | [mSVAMP](https://huggingface.co/datasets/Mathoctopus/MSVAMP) | Download |
|:--------------------------:|:----:|:------:|:--------:|
| **MAPO-DPO** (ours) |       41.1 |  57.3 |   [link](https://huggingface.co/kevinpro/MAPO-DPO-7B)   |
|         MathOctopus        |           38.1            |       40.7 |
|         MultiLingual-RFT        |          33.4            |       48.6 |
|         MathOctopus        |           38.1            |       40.7 |
|         MultiLingual-RFT        |          33.4            |       48.6 |
|          ChatGPT Zero-shot |       46.2 |      40.8    | -  |


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