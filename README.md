# FedGSDA
In real-world scenarios, graph data is often dispersed across various institutions, and collaborative training via federated graph learning emerges as a potent strategy for aggregating insights from diverse sources. While most existing research focuses on mitigating discrepancies in model parameters, this study takes a novel perspective, addressing challenges stemming from graph structure and node attribute distribution. Our approach, termed Federated Graph Learning with Structure and Node Distribution Augmentation under Non-IID Scenarios (FedGSDA), aims to amalgamate global insights into local models through contrastive learning, thereby bridging the gap between local and global distributions. Experimentation across four social datasets validates the efficacy of FedGSDA, achieving accuracies of 0.873, 0.704, 0.585, and 0.894, surpassing benchmarks like FedGCN. Ablation experiments from a structural standpoint effectively integrate global information into local models, while focusing on node attribute distribution reveals the benefits of contrasting intra-class nodes and inter-class centroids for improved class separability. Additionally, comparative evaluations of different federated learning algorithms under multi-client settings underscore the scalability of FedGSDA.

# Framework
![The Framework of FedGSDA](https://github.com/czfyvic/FedGSDA/tree/main/data/fig2.png)

# DataSet
The experimental dataset includes Cora, BlogCatalog, Flickr and Facebook, you can download it yourself on the Internet.

# Experimental environment
+ torch == 2.2.2
+ pandas == 1.22.0
+ networkx == 3.1
+ matplotlib == 3.7.1
+ numpy == 1.22.0

# Acknowledgement
This work was sponsored by the National Key Research and Development Program of China (No. 2018YFB0704400), Key Program of Science and Technology of Yunnan Province (No. 202002AB080001-2, 202102AB080019-3), Key Research Project of Zhejiang Laboratory (No.2021PE0AC02), Key Project of Shanghai Zhangjiang National Independent Innovation Demonstration Zone(No. ZJ2021-ZD-006). The authors gratefully appreciate the anonymous reviewers for their valuable comments.
