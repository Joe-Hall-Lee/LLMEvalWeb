
---
inference: false
language:
- en
tags:
- instruction-finetuning
pretty_name: JudgeLM-100K
task_categories:
- text-generation
---


<br>

# JudgeLM Model Card

## Model Details

JudgeLM is a judge model trained by fine-tuning Vicuna on JudgeLM-100K dataset.

- **Developed by:** [HUST](https://english.hust.edu.cn/), [BAAI](https://www.baai.ac.cn/english.html)
- **Model type:** An auto-regressive language model based on the transformer architecture.
- **License:** Non-commercial license
- **Finetuned from model:** [Vicuna](https://vicuna.lmsys.org).

### Model Sources

- **Repository:** https://github.com/baaivision/JudgeLM
- **Paper:** https://arxiv.org/abs/2310.17631
- **Demo:** http://218.91.113.230:9004/

## Uses

The primary use of JudgeLM is research on evaluating the performance of large language models and chatbots.
The primary intended users of the model are researchers and hobbyists in natural language processing, machine learning, and artificial intelligence.

## How to Get Started with the Model

- Judge large language models with this model: https://github.com/baaivision/JudgeLM/tree/main/judgelm/llm_judge.  
- Serve this model with the gradio: https://github.com/baaivision/JudgeLM/tree/main/judgelm/serve.  

## Training Details

JudgeLM v1.0 is fine-tuned from Vicuna-v1.3 with supervised instruction fine-tuning.
The training data is around 200K judge samples from [JudgeLM-100K dataset](https://huggingface.co/datasets/BAAI/JudgeLM-100K).
See more details in the "Fine-tuning Settings" section in the appendix of this [paper](https://arxiv.org/abs/2310.17631).

## Evaluation

JudgeLM is evaluated on JudgeLM val set, with judgements produced by GPT-4 teacher. See more details in this [paper](https://arxiv.org/abs/2310.17631) and try it with [code](https://github.com/baaivision/JudgeLM/tree/main/judgelm/llm_judge).

## Additional Information

### Citation Information

```
@article{zhu2023judgelm,  
	title={JudgeLM: Fine-tuned Large Language Models are Scalable Judges},  
	author={Lianghui Zhu and Xinggang Wang and Xinlong Wang},  
	year={2023},  
	eprint={2310.17631},  
	archivePrefix={arXiv},  
	primaryClass={cs.CL}  
}
```


