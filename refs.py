from shiny import ui


refs_page = ui.page_fluid(
    ui.HTML("<p>Hier finden Sie die Referenzen zu den verwendeten Datensätzen, Büchern aus dem Bereich des Maschinellen Lernens und den erwähnten Papern. </p>"),
    ui.HTML("<i>Datensätze</i>: <ul><li>Breast Cancer Wisconsin Dataset: Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B. </li><li>CIFAR-10: Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. https://www.cs.toronto.edu/~kriz/cifar.html</li></ul>"),
    ui.HTML("<i>Python packages</i>:<ul><li>Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011 </li><li>Ansel, J., Yang, E., He, H., Gimelshein, N., Jain, A., Voznesensky, M., Bao, B., Bell, P., Berard, D., Burovski, E., Chauhan, G., Chourdia, A., Constable, W., Desmaison, A., DeVito, Z., Ellison, E., Feng, W., Gong, J., Gschwind, M., Hirsh, B., Huang, S., Kalambarkar, K., Kirsch, L., Lazos, M., Lezcano, M., Liang, Y., Liang, J., Lu, Y., Luk, C., Maher, B., Pan, Y., Puhrsch, C., Reso, M., Saroufim, M., Siraichi, M. Y., Suk, H., Suo, M., Tillet, P., Wang, E., Wang, X., Wen, W., Zhang, S., Zhao, X., Zhou, K., Zou, R., Mathews, A., Chanan, G., Wu, P., & Chintala, S. (2024). PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation [Conference paper]. 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24). https://doi.org/10.1145/3620665.3640366</li></ul>"),
    ui.HTML("<i>Bücher & erwähnte Paper</i>:"), 
    ui.HTML("<ul><li>Lecun, Y.; Bottou, L.; Bengio, Y.; Haffner, P. (1998). 'Gradient-based learning applied to document recognition'. Proceedings of the IEEE. 86 (11): 2278–2324. doi:10.1109/5.726791. S2CID 14542261.</li></ul>"),
    ui.HTML("<ul><li>Bishop, C. M., & Nasrabadi, N. M. (2006). Pattern recognition and machine learning (Vol. 4, No. 4, p. 738). New York: springer.  </li></ul>"),
    ui.HTML("<ul><li>Goodfellow, I., Bengio, Y., Courville, A., & Bengio, Y. (2016). Deep learning (Vol. 1, No. 2). Cambridge: MIT press.  </li></ul>"),
    ui.HTML("<ul><li>Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: An introduction (Vol. 1, No. 1, pp. 9-11). Cambridge: MIT press. </li></ul>"),
    ui.HTML("<ul><li>Plaat, A. (2022). Deep reinforcement learning (Vol. 10, pp. 978-981). Singapore: Springer. </li></ul>"),
)