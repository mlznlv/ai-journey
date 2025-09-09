# AI Journey Roadmap (44 Weeks)

## 🎯 Main Goals
- **Foundation:** strengthen math and ML knowledge
- **Practice:** master deep learning, LLM fine-tuning, and deployment

---

## 📚 Core References

### Foundation
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Deisenroth
- [Pattern Recognition and Machine Learning (PRML)](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) — C. M. Bishop
- [Hands-On Machine Learning, 3rd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron · [GitHub repo](https://github.com/ageron/handson-ml3)

### Deep Learning & LLMs
- [Deep Learning Book](https://www.deeplearningbook.org/) — Goodfellow, Bengio, Courville
- [Dive into Deep Learning (D2L)](https://d2l.ai/)
- [Transformers for NLP](https://www.packtpub.com/product/transformers-for-natural-language-processing-second-edition/9781803247337) — Denis Rothman

### Research & Open Source
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Sutton & Barto
- [Probabilistic Machine Learning](https://probml.github.io/pml-book/) — Kevin Murphy
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al.

### Visibility & Product
- [The Hundred-Page ML Book](http://themlbook.com/wiki/doku.php) — Andriy Burkov
- [Building Machine Learning Powered Applications](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/) — Emmanuel Ameisen
- [You Look Like a Thing and I Love You](https://www.janelleshane.com/book) — Janelle Shane

---

## 📅 Weekly Plan (44 Weeks)

### Weeks 1–4: Setup + ML Start
**Week 1 — Titanic EDA**
- Python Data Science Handbook
- [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [pandas Getting Started](https://pandas.pydata.org/docs/getting_started/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- **Project:** Titanic dataset exploratory analysis

**Week 2 — Regression**
- Hands-On ML, Chapter 4
- [scikit-learn regression tutorial](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html)
- [scikit-learn examples](https://github.com/scikit-learn/scikit-learn/tree/main/examples)
- **Project:** Boston/California Housing regression

**Week 3 — Probability Foundations**
- Mathematics for ML (Probability chapter)
- [Khan Academy Probability](https://www.khanacademy.org/math/statistics-probability/probability-library)
- [Harvard Stat110](https://projects.iq.harvard.edu/stat110/home)
- **Project:** Dice roll simulation + Bayesian updating

**Week 4 — Regularization**
- Hands-On ML, Chapter 5
- [Ridge & Lasso documentation](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)
- [ML-From-Scratch repo](https://github.com/eriklindernoren/ML-From-Scratch)
- **Project:** Ridge vs Lasso comparison

---

### Weeks 5–8: ML Expansion + Math
**Week 5 — Ensembles**
- [XGBoost Docs](https://xgboost.readthedocs.io/)
- Hands-On ML, Chapters 6–7
- **Project:** Titanic predictor with RF/GBM

**Week 6 — Imbalanced Data & Metrics**
- [imbalanced-learn](https://imbalanced-learn.org/stable/)
- PRML (evaluation metrics)
- **Project:** Credit Card Fraud dataset

**Week 7 — ML Capstone**
- [UCI SMS Spam dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [Kaggle Spam Detection tutorial](https://www.kaggle.com/code/uciml/sms-spam-collection-dataset)
- **Deliverable:** Spam classifier + blog post

**Week 8 — Linear Algebra**
- Mathematics for ML (Linear Algebra chapters)
- MIT 18.06 (Strang lectures)
- [3Blue1Brown Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
- **Project:** Eigenvalues + SVD demo

---

### Weeks 9–12: Probability + Optimization + PyTorch
**Week 9 — Bayes Rule**
- [Harvard Stat110 lectures](https://projects.iq.harvard.edu/stat110/youtube)
- **Project:** Bayes rule notebook

**Week 10 — Optimization**
- [Convex Optimization (Boyd)](https://web.stanford.edu/~boyd/cvxbook/)
- [CS229 optimization notes](http://cs229.stanford.edu/notes2020fall/cs229-notes1.pdf)
- **Project:** Gradient descent from scratch

**Week 11 — PyTorch Basics**
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch examples](https://github.com/pytorch/examples)
- **Project:** MLP on tabular dataset

**Week 12 — CNNs**
- Goodfellow DL, Chapter 9
- [CS231n CNN Notes](http://cs231n.github.io/convolutional-networks/)
- [TorchVision CIFAR-10](https://pytorch.org/vision/stable/datasets.html#cifar)
- **Project:** CIFAR-10 classifier

---

### Weeks 13–18: Deep Learning Foundations
**Week 13 — Optimizers**
- [Dive into Deep Learning (Ch.6–7)](https://d2l.ai/)
- **Project:** Adam vs SGD on MNIST

**Week 14 — fast.ai**
- [fast.ai course](https://course.fast.ai/)
- **Project:** Data augmentation + LR schedule

**Week 15 — Sequence Models**
- [PyTorch Seq2Seq repo](https://github.com/bentrevett/pytorch-seq2seq)
- [IMDB sentiment dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Project:** IMDB sentiment classifier

**Week 16 — Multi-GPU Training**
- [PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [HF Accelerate](https://huggingface.co/docs/accelerate/index)
- **Project:** Train ResNet on multi-GPU

**Weeks 17–18 — DL Capstone**
- Option A: [CIFAR-10 Image Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- Option B: [IMDB Sentiment RNN](https://github.com/bentrevett/pytorch-sentiment-analysis)
- **Deliverable:** Notebook + training curves + metrics report

---

### Weeks 19–30: Transformers & LLMs
- Week 19: [Attention Is All You Need](https://arxiv.org/abs/1706.03762), [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- Week 20: [HF Transformers Course](https://huggingface.co/course/chapter1), [HF examples](https://github.com/huggingface/transformers/tree/main/examples)
- Week 21: [HF Tokenizers Docs](https://huggingface.co/docs/tokenizers/index) — Project: custom tokenizer
- Week 22: [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
- Week 23: [HF text classification notebook](https://github.com/huggingface/notebooks/tree/main/examples/text_classification) — Project: fine-tune DistilBERT on IMDB
- Weeks 24–25: [HF PEFT Docs](https://huggingface.co/docs/peft/index) — Project: LoRA/QLoRA experiment
- Week 26: [HF RAG demo](https://github.com/huggingface/notebooks/blob/main/examples/rag/rag.ipynb) — Project: RAG Q&A bot
- Weeks 27–28: [HF Generation Docs](https://huggingface.co/docs/transformers/generation) — Project: decoding strategies comparison
- Week 29: [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — Project: quantized model deployment
- Week 30: LLM Capstone — build a domain chatbot and deploy to [HF Spaces](https://huggingface.co/spaces)

---

### Weeks 31–40: Specializations + MLOps
- Weeks 31–32: NLP — [Stanford CS224N](http://web.stanford.edu/class/cs224n/) · [Assignments](https://github.com/stanfordnlp/cs224n-winter17-notes)
- Weeks 33–34: CV — [Stanford CS231n](http://cs231n.stanford.edu/) · [Assignments](https://github.com/cs231n/cs231n.github.io)
- Weeks 35–36: Diffusion — [HF Diffusers Course](https://huggingface.co/learn/diffusion-course) · [Repo](https://github.com/huggingface/diffusers)
- Week 37: Advanced Diffusion — [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Project: ControlNet demo
- Weeks 38–40: MLOps — [Full Stack Deep Learning](https://fullstackdeeplearning.com/), [Made With ML](https://madewithml.com/), tools: MLflow, W&B, Docker, Kubernetes — Project: CI/CD pipeline + monitoring

---

### Weeks 41–44: Reinforcement Learning
- Week 41: [Spinning Up in Deep RL](https://spinningup.openai.com/) — Project: CartPole Q-learning
- Week 42: Sutton & Barto (Ch.13) — Project: Actor-Critic notebook
- Week 43: [HF RLHF blog](https://huggingface.co/blog/rlhf) — Project: PPO experiment
- Week 44: RL Capstone — PPO agent on CartPole
