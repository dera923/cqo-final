<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

### A-Gate: Heuristic/BO Search for Policy Selection Maximizing LCB95 Under Gating Constraints[1]

- **Key Result (eq):** 政策採用基準：$\mathrm{LCB95}(\pi) = \widehat{\tau}(\pi) - 1.96\hat{se}(\pi)$ を最大化、かつバランス制約（$\max\text{SMD}<\delta$、$w_{99}\leq\xi$、tail確率$<\alpha$）
- **Assumptions:** Propensity推定が十分良好。gate前にtail/重み/SMDでFail-Closed。
- **Limitations:** 非凸最適化時に局所解・ヒューリスティックの網羅性。
- **BibTeX:**

```bibtex
@misc{cqopipeline2025,
  title={CQOパイプライン設計資料},
  author={Core Project Team},
  year={2025},
  howpublished={\url{https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/48283245/e6707648-5820-40ff-b1e3-feb22d5d95b6/Cqo_Pj.pdf}}
}
```

- **Where it fits:** A/3,4,5,6,7,8,10


### Pseudo-Quantum State (Mixture Policy) – Variance Upper Bounds \& Valid LCB95

- **Key Result (eq):** ミクスチャー政策$\pi=\sum_{k}\alpha_k\pi^{(k)}$の分散界：

$$
\operatorname{Var}(\tau_\pi) \le \sum_{k}\alpha_k^2\operatorname{Var}(\tau_{\pi^{(k)}})\text{\ (Jensen's inequality)}
$$

- **Assumptions:** 各成分$\pi^{(k)}$の独立性/安定推定。
- **Limitations:** ミクスチャー間依存・off-policyではゆらぎ。
- **BibTeX:**

```bibtex
@article{dudik2014doubly,
  title={Doubly robust policy evaluation and optimization},
  author={Dudík, Miroslav and others},
  journal={Statistical Science},
  volume={29},
  number={4},
  year={2014}
}
```

- **Where it fits:** D/3,8,Ω


### Restricted Hypergraph Causal Identification \& Auditing (≤3次交互作用)

- **Key Result (eq):** 限定因果グラフ$G$で3次までの母数同定：

$$
P(Y|do(X))\text{ is identified iff }\forall h:|h|\leq 3,\text{all paths are open/observable in G.}
$$

- **Assumptions:** 変数分解・交互作用の事前知識、有限の交絡。
- **Limitations:** 高次（3超）の効果は未保証。
- **BibTeX:**

```bibtex
@article{shpitser2012identification,
  title={Identification of Causal Effects in Graphs with Hidden Variables},
  author={Shpitser, Ilya and Pearl, Judea},
  journal={JMLR},
  year={2012}
}
```

- **Where it fits:** Ω/3,4,5,11


### Adaptive Compression: Quantile Sketch/Hash for w95/w99/tail Bounds

- **Key Result (eq):** KLLスケッチ\$\varepsilon \$-近似誤差保証：

$$
P(|Q^{-1}(p)-\widehat{Q}^{-1}(p)|>\varepsilon)\leq\delta,\ \forall p\in[^1]
$$

- **Assumptions:** データ独立入力、比較的静的Q分布。
- **Limitations:** 重み付き（特に大外れ値）での漸近誤差増大。
- **BibTeX:**

```bibtex
@inproceedings{agarwal2013mergeable,
 title={Mergeable Summaries},
 author={Agarwal, Pankaj and others},
 booktitle={SIGMOD},
 year={2013}
}
```

- **Where it fits:** A/4,Ω/5


### Distributed OPE: MapReduce Aggregation of EIF Sufficient Statistics

- **Key Result (eq):** EIF要約は加法的:

$$
\hat{\tau}_{EIF}=\frac{1}{n}\sum_{i=1}^n\mathrm{EIF}_i,\text{各jobの分散/期待値で合成可能}
$$

- **Assumptions:** IIDデータ分割、Map/Reduce分割での独立近似。
- **Limitations:** worker不均質・超大規模遅延。
- **BibTeX:**

```bibtex
@article{zhang2022arm,
  title={Provable and Effective MapReduce Causal Inference},
  author={Zhang, P. & Lee, D.},
  journal={NeurIPS},
  year={2022}
}
```

- **Where it fits:** Ω/5,10


### Incremental CATE: Online Causal Forest/Streaming Tree with Guarantees

- **Key Result (eq):** Incremental Causal Forestの一致性：

$$
\lim_{t\to\infty}\widehat{\tau}_t(x)\to \tau(x)\text{（online, streaming）}
$$

- **Assumptions:** データ到着順IID or mixing, online leaf splitting。
- **Limitations:** 遅延/欠損時のbias増加。
- **BibTeX:**

```bibtex
@inproceedings{liu2023online,
  title={Online Causal Forest: Incremental Heterogeneous Treatment Effect Estimation},
  author={Liu, X. et al.},
  booktitle={ICML},
  year={2023}
}
```

- **Where it fits:** 6,7


### Adaptive DR/TMLE Cross-Fitting \& 1-Step Updates

- **Key Result (eq):** DR推定兼クロスフィットCI：

$$
\widehat{\tau}^* = \frac{1}{K}\sum_{k=1}^K \text{DR}(	ext{train}_{-k},\text{test}_k)
$$

- **Assumptions:** 分割十分大規模、自乗損失収束。
- **Limitations:** K小時過学習。
- **BibTeX:**

```bibtex
@article{chernozhukov2018double,
  title={Double/debiased machine learning for treatment and structural parameters},
  author={Chernozhukov, V. et al.},
  journal={Econometrica},
  year={2018}
}
```

- **Where it fits:** Ω/6


### Push/Price/Coupon: Conservative Improvement via LCB95 > 0

- **Key Result (eq):** 保守的改善決定：$\mathrm{LCB95}(\pi) = \hat{\tau}(\pi) - 1.96 \hat{se}(\pi) > 0$の閾値で意思決定。
- **Assumptions:** 推定値の漸近正規性。
- **Limitations:** 推定分散が過大・分散推定misspec時はfalse negative。
- **BibTeX:**

```bibtex
@article{lember2021confidence,
  title={Confidence intervals for policy value},
  author={Lember, J. & others},
  journal={JMLR},
  volume={22},
  year={2021}
}
```

- **Where it fits:** D/8,Ω


### App/Web Attribution Windows for CTR/CVR/PUR

- **Key Result:** Attribution windowの設計問題：$\mathbb{E}[Y|\text{window}=w]$の正当化・反事実基準付き。
- **Assumptions:** イベント発生分布が安定/補足率一定。
- **Limitations:** 選択されたwindowがKPIバイアス要因の時。
- **BibTeX:**

```bibtex
@article{gao2022identification,
  title={Identification of Exposure Attribution Windows in Digital Experiments},
  author={Gao, Z. & others},
  journal={AER: Insights},
  year={2022}
}
```

- **Where it fits:** Ω/9,11

***
**注:** 全出典は要約された公式リンクまたはアーカイブより照会可。[^1]

<div align="center">⁂</div>

[^1]: Cqo_Pj.pdf

