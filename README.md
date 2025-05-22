---
# 详细文档见 https://modelscope.cn/docs/%E5%88%9B%E7%A9%BA%E9%97%B4%E5%8D%A1%E7%89%87
domain:
    - nlp
tags: #自定义标签
    -
datasets: # 关联数据集
    evaluation:
    #- iic/ICDAR13_HCTR_Dataset
    test:
    #- iic/MTWI
    train:
    #- iic/SIBR
models: # 关联模型
    - BAAI/JudgeLM-7B-v1.0
    - JHL2004/JudgeLM-7B-Debiased

## 启动文件
deployspec:
    entry_file: webui/app.py
license: Apache License 2.0
---

#### Clone with HTTP

```bash
 git clone https://www.modelscope.cn/studios/JHL2004/LLMEvalWeb.git
```

#### Citation

```bibtex
@inproceedings{zhou2024mitigating,
    title={Mitigating the Bias of Large Language Model Evaluation},
    author={Zhou, Hongli and Huang, Hui and Long, Yunfei and Xu, Bing and Zhu, Conghui and Cao, Hailong and Yang, Muyun and Zhao, Tiejun},
    booktitle={Proceedings of the 23rd Chinese National Conference on Computational Linguistics (Volume 1: Main Conference)},
    year={2024},
    address={Taiyuan, China},
    publisher={Chinese Information Processing Society of China},
    url={https://aclanthology.org/2024.ccl-1.101/},
    pages={1310--1319}
}
```

```bibtex
@misc{huang2024empiricalstudyllmasajudgellm,
      title={An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Model is not a General Substitute for GPT-4},
      author={Hui Huang and Yingqi Qu and Xingyuan Bu and Hongli Zhou and Jing Liu and Muyun Yang and Bing Xu and Tiejun Zhao},
      year={2024},
      eprint={2403.02839},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2403.02839},
}
```
