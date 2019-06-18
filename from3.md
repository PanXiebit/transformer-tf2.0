---
title: 从0开始GAN-3-文本生成planning
date: 2019-06-16 09:48:54
mathjax: true
tags: GAN
categories: GAN
---

写这篇博客源于在知乎上看到大佬Towser在 “BERT模型在NLP中目前取得如此好的效果，那下一步NLP该何去何从？” 这个问题下的回答，对于文本生成的总结觉得太赞了。所以基于大佬的回答，画了一个脑图(http://www.xmind.net/m/AcA3bE)，接下来一两个月的时间也决定按照这个路线进行学习。

![](从0开始GAN-3-文本生成planning/text_generation.png)

# RL+GAN in Text Generation
## Generating Sentences from a Continuous Space Samuel
这是非常早期的一篇基于变分自编码做文本生成的论文，我们都知道VAE和GAN是非常类似的。所以在看 GAN text generation相关的paper之前先学习下如何用VAE做文本生成。

关于 VAE 有两篇非常不错的blog:  

- [苏剑林变分自编码器（一）：原来是这么一回事](https://kexue.fm/archives/5253)   
- [Variational Autoencoders Explained](http://anotherdatum.com/vae.html)
### 何为 VAE  

### Motivation
传统 RNNLM 在做text生成的时候，其结构是把一个序列breaking成一个个next word的prediction. 这使得模型并没有显示的获取文本的全局信息，也没有获取类似于topic和句法相关的高级特征。

于是乎，作者提出了一种基于vatiational encoder的方法，能有效获取global feature，并且能避免 MLM 带来的几乎不可能完成的计算。同时，作者认为基于传统的language model的验证方法并不能有效展示出global feature的存在，于是提出了一种新的 evaluation strategy.

> For this task, we introduce a novel evaluation strategy using an adversarial classifier, sidestepping the issue of intractable likelihood computations by drawing inspiration from work on non-parametric two-sample tests and adversarial training.
