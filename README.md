# Associative Recurrent Memory Transformer implementation compatible with Hugging Face models


ARMT is a memory-augmented segment-level recurrent Transformer. It scales up to 50M tokens being trained only on 16k. 

>[paper](https://arxiv.org/abs/2407.04841) [code](https://github.com/RodkinIvan/associative-recurrent-memory-transformer/tree/llama_armt) Associative Recurrent Memory Transformer

>[paper](https://arxiv.org/abs/2304.11062) Scaling Transformer to 1M tokens and beyond with RMT

>[paper](https://arxiv.org/abs/2207.06881) Recurrent Memory Transformer


We implement our memory mechanism with no changes to Transformer model by adding special memory tokens and linear-attention style associative memory. The model is trained to control both memory operations and sequence representations processing.


## Installation
```bash
pip install -e .
```
This command will install `lm_experiments_tools` with only required packages for Trainer and tools.

`lm_experiments_tools` Trainer supports gradient accumulation, logging to tensorboard, saving the best models
based on metrics, custom metrics and data transformations support.

### Install requirements for all experiments
Full requirements for all experiments are specified in requirements.txt. Install requirements after cloning the repo:
```bash
pip install -r requirements.txt
```

To run langudge modelling with ARMT with sliding window:

```
cd scripts/pg19
bash finetune_armt_llama3.2_pg19_sliding.sh
```


## Citation
If you find our work useful, please cite the RMT papers:

```
@misc{rodkin2025associativerecurrentmemorytransformer,
      title={Associative Recurrent Memory Transformer}, 
      author={Ivan Rodkin and Yuri Kuratov and Aydar Bulatov and Mikhail Burtsev},
      year={2025},
      eprint={2407.04841},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.04841}, 
}
```

```
@inproceedings{
        bulatov2022recurrent,
        title={Recurrent Memory Transformer},
        author={Aydar Bulatov and Yuri Kuratov and Mikhail Burtsev},
        booktitle={Advances in Neural Information Processing Systems},
        editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
        year={2022},
        url={https://openreview.net/forum?id=Uynr3iPhksa}
}
```
```
@misc{bulatov2023scaling,
      title={Scaling Transformer to 1M tokens and beyond with RMT}, 
      author={Aydar Bulatov and Yuri Kuratov and Mikhail S. Burtsev},
      year={2023},
      eprint={2304.11062},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
```
@misc{kuratov2024search,
      title={In Search of Needles in a 11M Haystack: Recurrent Memory Finds What LLMs Miss}, 
      author={Yuri Kuratov and Aydar Bulatov and Petr Anokhin and Dmitry Sorokin and Artyom Sorokin and Mikhail Burtsev},
      year={2024},
      eprint={2402.10790},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
