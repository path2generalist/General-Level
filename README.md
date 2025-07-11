
<div align="center">
<img src='https://cdn-uploads.huggingface.co/production/uploads/647773a1168cb428e00e9a8f/N8lP93rB6lL3iqzML4SKZ.png'  width=100px>

<h1 align="center"><b>On Path to Multimodal Generalist: General-Level and General-Bench</b></h1>
<p align="center">
<a href="https://generalist.top/">[📖 Project]</a>
<a href="https://generalist.top/leaderboard">[🏆 Leaderboard]</a>
<a href="https://arxiv.org/abs/2505.04620">[📄 Paper]</a>
<a href="https://huggingface.co/papers/2505.04620">[🤗 Paper-HF]</a>
<a href="https://huggingface.co/General-Level/General-Bench-Closeset">[🤗 Dataset-HF (Close-Set)]</a>
<a href="https://huggingface.co/General-Level/General-Bench-Openset">[🤗 Dataset-HF (Open-Set)]</a>
</p>




---

<h1 align="center" style="color: red"><b>General-Level Evaluation Suite</b></h1>

---
</div>



<h1 align="center" style="color:#F27E7E"><em>
Does higher performance across tasks indicate a stronger capability of MLLM, and closer to AGI?
<br>
NO! But <b style="color:red">synergy</b> does.
</em></h1>


Most current MLLMs predominantly build on the language intelligence of LLMs to simulate the indirect intelligence of multimodality, which is merely extending language intelligence to aid multimodal understanding. While LLMs (e.g., ChatGPT) have already demonstrated such synergy in NLP, reflecting language intelligence, unfortunately, the vast majority of MLLMs do not really achieve it across modalities and tasks.

We argue that the key to advancing towards AGI lies in the synergy effect—a capability that enables knowledge learned in one modality or task to generalize and enhance mastery in other modalities or tasks, fostering mutual improvement across different modalities and tasks through interconnected learning.


<div align="center">
<img src='https://cdn-uploads.huggingface.co/production/uploads/647773a1168cb428e00e9a8f/-Asn68kJGjgqbGqZMrk4E.png'  width=950px>
</div>


---

# 🏆 Overall Leaderboard<a name="leaderboard" />

<div align="center">
<img src='https://cdn-uploads.huggingface.co/production/uploads/647773a1168cb428e00e9a8f/s1Q7t6Nmtnmv3bSvkquT0.png'  width=1200px>
</div>


------

# 🚀 General-Level <a name="level" />
  
**A 5-scale level evaluation system with a new norm for assessing the multimodal generalists (multimodal LLMs/agents).  
The core is the use of <b style="color:red">synergy</b> as the evaluative criterion, categorizing capabilities based on whether MLLMs preserve synergy across comprehension and generation, as well as across multimodal interactions.**

General-Level evaluates generalists based on the levels and strengths of the synergy they preserve. Specifically, we define three scopes of synergy, ranked from low to high: no synergy, task-level synergy (‘task-task’), paradigm-level synergy (‘comprehension-generation’), and cross-modal total synergy (‘modality-modality’), as illustrated here:


<div align="center">
<img src='https://cdn-uploads.huggingface.co/production/uploads/647773a1168cb428e00e9a8f/lnvh5Qri9O23uk3BYiedX.jpeg'  width=1000px>
</div>


Achieving these levels of synergy becomes progressively more challenging, corresponding to higher degrees of general intelligence. Assume we have a benchmark of various modalities and tasks, where we can categorize tasks under these modalities into the Comprehension group and the Generation group, as well as the language (i.e., NLP) group, as illustrated here:


<div align="center">
<img src='https://github.com/user-attachments/assets/e9e9a53a-49e7-422f-b2ff-c3cc67f8e16f'  width=900px>
</div>



Let’s denote the number of datasets or tasks within the Comprehension task group by M; the number within the Generation task group by N; and the number of NLP tasks by T.

Now, we demonstrate the specific definition and calculation of each level:
<div align="center">
<img src='https://cdn-uploads.huggingface.co/production/uploads/647773a1168cb428e00e9a8f/BPqs-3UODQWvjFzvZYkI4.png'  width=1000px>
</div>



------

<h1 style="font-weight: bold; text-decoration: none;"> ⚠️ Scoring Relaxation <a name="relaxation" /> </a> </h1>

A central aspect of our General-Level framework lies in **how synergy effects are computed**. According to the standard understanding of the synergy concept, 
e.g., the performance of a generalist model on joint modeling of tasks A and B (e.g., Pθ(y|A,B)) should exceed its performance when modeling task A alone (e.g., Pθ(y|A)) 
or task B alone (e.g., Pθ(y|B)). However, adopting this approach poses a significant challenge that hinders the measurement of synergy: there is no feasible way to 
establish two independent distributions, Pθ(y|A) and Pθ(y|B), and a joint distribution Pθ(y|A,B). 
This limitation arises because a given generalist model has already undergone extensive pre-training and fine-tuning, where tasks A and B have likely been jointly modeled.
It is impractical to retrain such a generalist to isolate the learning and modeling of tasks A or B independently in order to derive these distributions. 
Otherwise, such an approach would result in excessive redundant computation and inference on the benchmark data.

To simplify and relax the evaluation of synergy, we introduce a key assumption in the scoring algorithm:


> Theoretically, we posit that the stronger a model's synergy capability, the more likely it is to surpass the task performance of SoTA specialists when synergy is effectively employed.
> Then, we can simplify the synergy measurement as: if a generalist outperforms a SoTA specialist in a specific task, we consider it as evidence of a synergy effect, i.e.,
> leveraging the knowledge learned from other tasks or modalities to enhance its performance in the targeted task.

By making this assumption, we avoid the need for direct pairwise measurements between `task-task', `comprehension-generation', or `modality-modality', which would otherwise require complex and computationally intensive algorithms.


------


<h1 style="font-weight: bold; text-decoration: none;"> 📌 Citation <a name="cite" /> </a> </h1>

If you find our benchmark useful in your research, please kindly consider citing us:

```bibtex
@articles{fei2025pathmultimodalgeneralistgenerallevel,
  title={On Path to Multimodal Generalist: General-Level and General-Bench},
  author={Hao Fei and Yuan Zhou and Juncheng Li and Xiangtai Li and Qingshan Xu and Bobo Li and Shengqiong Wu and Yaoting Wang and Junbao Zhou and Jiahao Meng and Qingyu Shi and Zhiyuan Zhou and Liangtao Shi and Minghe Gao and Daoan Zhang and Zhiqi Ge and Weiming Wu and Siliang Tang and Kaihang Pan and Yaobo Ye and Haobo Yuan and Tao Zhang and Tianjie Ju and Zixiang Meng and Shilin Xu and Liyu Jia and Wentao Hu and Meng Luo and Jiebo Luo and Tat-Seng Chua and Shuicheng Yan and Hanwang Zhang},
  eprint={2505.04620},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
  url={https://arxiv.org/abs/2505.04620},
}
```


