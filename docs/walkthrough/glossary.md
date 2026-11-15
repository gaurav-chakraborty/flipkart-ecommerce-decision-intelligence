# Statistical, Machine Learning, & Domain Glossary

Author: Gaurav Chakraborty  
Student ID: iitmd_dsai_2508432  
Course: Capstone Project — Flipkart E-Commerce Decision Intelligence

## 1. Statistical Inference & Regression Terminology

- **Bessel Correction ($ddof=1$)**: The use of $n - 1$ instead of $n$ in the denominator when calculating sample variance $s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$. It eliminates downward bias, yielding an unbiased estimator of population variance $\sigma^2$.
- **Standard Error (SE)**: The standard deviation of the sampling distribution of a statistic: $\text{SE}(\bar{x}) = \frac{s}{\sqrt{n}}$. Measures sample-to-sample variability of the sample mean.
- **Confidence Interval (95% CI)**: An interval estimate $[\bar{x} - 1.96 \cdot \text{SE}, \bar{x} + 1.96 \cdot \text{SE}]$ that, under repeated sampling from the population, encompasses the true population parameter in 95% of realized samples.
- **Two-Sided $z$-Test**: A statistical hypothesis test where the test statistic $z = \frac{\bar{x} - \mu_0}{\text{SE}}$ is compared against standard normal critical values $\pm 1.96$ to assess whether the sample mean differs significantly from null hypothesis value $\mu_0$.
- **Ordinary Least Squares (OLS)**: A parameter estimation method that minimizes the residual sum of squares $\text{RSS} = \sum (y_i - \hat{y}_i)^2$. Closed-form simple regression slope is $\hat{\beta}_1 = \frac{\text{Cov}(x, y)}{\text{Var}(x)}$ and intercept is $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$.
- **ANOVA Sum of Squares Decomposition**: The mathematical partitioning $TSS = ESS + RSS$, where:
  - Total Sum of Squares ($TSS = \sum (y_i - \bar{y})^2$) measures total target variance.
  - Explained Sum of Squares ($ESS = \sum (\hat{y}_i - \bar{y})^2$) measures variance explained by the model.
  - Residual Sum of Squares ($RSS = \sum (y_i - \hat{y}_i)^2$) measures unexplained error variance.
- **Coefficient of Determination ($R^2$)**: The proportion of target variance explained by the model: $R^2 = \frac{ESS}{TSS} = 1 - \frac{RSS}{TSS}$.
- **Adjusted $R^2$ ($R^2_{\text{adj}}$)**: A modified measure that penalizes the addition of non-informative explanatory features: $R^2_{\text{adj}} = 1 - (1 - R^2) \frac{n - 1}{n - p - 1}$.
- **Heteroscedasticity**: The violation of the classical linear regression assumption of constant error variance $\text{Var}(\epsilon_i) = \sigma^2$. Manifests as a residual "funnel" where variance expands with the fitted value.
- **Log-Log Transformation**: Fitting $\ln(y) = \beta_0 + \beta_1 \ln(x) + \epsilon$. Converts multiplicative, variance-expanding relationships into linear homoscedastic relationships; $\beta_1$ represents elasticity (% change in $y$ per 1% change in $x$).

## 2. Classification, ROC, & Business Cost Optimization

- **Stratified Sampling**: Partitioning a dataset such that every split (train and test) contains exactly the same class proportions as the full population. Prevents distribution shift.
- **Data Leakage**: The contamination of the training process with information from the test set. Prevented by fitting all scalers and transformations strictly on training rows only.
- **Logistic Sigmoid**: The mathematical activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$ that maps real-valued scores $z \in (-\infty, \infty)$ into valid probabilities $p \in (0, 1)$.
- **Binary Cross-Entropy (BCE) Loss**: The negative log-likelihood loss for binary classification: $L(y, p) = -\frac{1}{n} \sum [y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i)]$.
- **Batch Gradient Descent**: An optimization algorithm that iteratively updates model weights by taking steps proportional to the negative gradient evaluated across the entire training dataset: $w \leftarrow w - \eta \nabla_w L$.
- **Odds Ratio (OR)**: The factor by which the odds of an outcome change per unit increase in an explanatory predictor: $\text{OR}_j = \exp(w_j)$. For standardized features, it represents the multiplier per standard deviation change.
- **Confusion Matrix**: A $2 \times 2$ contingency table recording classification outcomes:
  - True Positives (TP): Correctly identified churners.
  - False Positives (FP): Non-churners incorrectly flagged (false alarms).
  - True Negatives (TN): Non-churners correctly identified as retained.
  - False Negatives (FN): Churners missed by the model.
- **Precision**: $\frac{TP}{TP + FP}$. The proportion of flagged customers who actually churn.
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$. The proportion of actual churners who are successfully flagged.
- **Specificity**: $\frac{TN}{TN + FP}$. The proportion of non-churners correctly left unflagged.
- **$F_1$-Score**: The harmonic mean of precision and recall: $F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$.
- **Receiver Operating Characteristic (ROC)**: A curve plotting True Positive Rate (Recall) vs False Positive Rate ($1 - \text{Specificity}$) as the decision threshold $\tau$ varies from 1.0 down to 0.0.
- **Asymmetric Business Cost**: An operational objective function where misclassification costs are unequal: $\text{Cost} = C_{\text{FA}} \cdot \text{FP} + C_{\text{Miss}} \cdot \text{FN}$. In this domain, $C_{\text{FA}} = \text{INR } 50$ (wasted discount voucher) and $C_{\text{Miss}} = \text{INR } 400$ (lost customer margin).
- **$K$-Nearest Neighbours (KNN)**: A non-parametric classifier that assigns labels based on the majority class among the $K$ closest training samples in standardized feature space.

## 3. Unsupervised Clustering & Dimensionality Reduction

- **$K$-Means Clustering**: An iterative partitioning algorithm that divides $N$ observations into $K$ clusters, assigning each point to the nearest centroid and recalculating centroids until convergence.
- **Within-Cluster Sum of Squares (WCSS)**: $\sum_{k=1}^K \sum_{i \in C_k} \|x_i - \mu_k\|^2$. Measures intra-cluster compactness. Monotonically decreases with increasing $K$.
- **Elbow Method (Second Differences)**: An objective criterion for selecting optimal $K$. The discrete second difference $\Delta_2(K) = \Delta_1(K-1) - \Delta_1(K)$ identifies the point of maximum curvature where additional clusters yield diminishing returns.
- **Hierarchical Agglomerative Clustering (HAC)**: A bottom-up clustering procedure that starts with $N$ singleton clusters and iteratively merges the closest pair of clusters.
- **Linkage Criteria**:
  - Single Linkage: Distance between closest points: $d(A, B) = \min_{a \in A, b \in B} d(a, b)$. Vulnerable to chaining.
  - Complete Linkage: Distance between furthest points: $d(A, B) = \max_{a \in A, b \in B} d(a, b)$. Yields compact, spherical clusters.
  - Average Linkage: Mean distance between all pairs: $d(A, B) = \frac{1}{|A||B|} \sum_{a, b} d(a, b)$. Compromise between single and complete.
- **Dendrogram Cut**: Slicing the hierarchical merge tree horizontally at a fixed distance threshold $H$ to extract discrete cluster partitions.
- **Principal Component Analysis (PCA)**: An orthogonal linear transformation that converts correlated features into linearly uncorrelated variables (principal components) ordered by variance explained.
- **Eigenvalues ($\lambda_j$)**: Roots of the characteristic polynomial $\det(R - \lambda I) = 0$. In PCA on a correlation matrix, $\lambda_j$ equals the variance explained by the $j$-th principal component, and $\sum \lambda_j = p$.
- **Factor Loadings**: The correlation between original variables and principal components: $L_{ij} = v_{ij} \sqrt{\lambda_j}$, reflecting variable contributions to components.

## 4. Decision Trees, Ensembles, & Deep Learning Architectures

- **Shannon Entropy**: A measure of impurity or information disorder in a distribution: $H(S) = -\sum p_i \log_2(p_i)$. Reaches maximum 1.0 for balanced binary outcomes and 0.0 for pure sets.
- **Gini Impurity**: An alternative impurity measure: $\text{Gini}(S) = 1 - \sum p_i^2 = 2p(1-p)$. Bounded in $[0, 0.5]$ for binary classification.
- **Information Gain (IG)**: The expected reduction in entropy achieved by partitioning a dataset on feature $X$: $\text{IG}(S, X) = H(S) - \sum \frac{|S_v|}{|S|} H(S_v)$.
- **Decision Stump**: A 1-level decision tree consisting of a single internal decision node and two leaf nodes. Serves as a canonical weak learner.
- **Majority Voting Ensemble**: An ensemble architecture that combines predictions from multiple heterogeneous base models (e.g. Logistic Regression, KNN, Decision Stump) and selects the class with the most votes.
- **AdaBoost**: An adaptive boosting algorithm that trains weak learners sequentially, increasing the sample weights of misclassified instances and weighting learners by their accuracy: $\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$.
- **Out-of-Bag (OOB) Samples**: In bootstrap aggregating (bagging / Random Forests), the subset of training samples not selected in a bootstrap sample of size $N$. Asymptotically approaches $1/e \approx 36.79\%$.
- **Convolutional Layer (Conv2D)**: A neural network layer applying learnable filter kernels across spatial dimensions: $\text{Params} = (K_h \cdot K_w \cdot C_{\text{in}} + 1) \cdot C_{\text{out}}$.
- **Max Pooling**: A non-parametric spatial downsampling operation that extracts the maximum value within localized receptive fields (e.g. $2 \times 2$). Contains zero learnable parameters.
- **Recurrent Neural Network (RNN)**: A sequential neural architecture maintaining a hidden state vector $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$.
