# Viva Voce Examination Preparation: 40 Core Defense Questions

Author: Gaurav Chakraborty  
Student ID: iitmd_dsai_2508432  
Course: Capstone Project — Flipkart E-Commerce Decision Intelligence

## 1. Part 1: Statistical Estimation & Regression (10 Questions)

### Q1.1: Why did you use Bessel correction ($ddof=1$) instead of $n$ for sample standard deviation?
**Answer**: Using $n$ in the denominator underestimates population variance because sample observations cluster around the sample mean $\bar{x}$ rather than the true population mean $\mu$. Dividing by the degrees of freedom $n - 1$ ($ddof=1$) provides an algebraically unbiased estimator $E[s^2] = \sigma^2$. For our $N=800$, sample SD is 9.1723.

### Q1.2: Why is a two-sided $z$-test justified rather than a Student's $t$-test for $H_0: \mu = 8.0$?
**Answer**: With sample size $N = 800$, the Central Limit Theorem guarantees that the sampling distribution of the mean is Gaussian regardless of underlying population skewness. Furthermore, with $N=800$, the $t$-distribution with 799 degrees of freedom is virtually indistinguishable from the standard normal distribution ($t_{0.975, 799} = 1.9629 \approx 1.9600$). Our test statistic $z = 2.8930 > 1.96$ rejects $H_0$ ($p = 0.0038$).

### Q1.3: In Task 1.3, why did you use `math.ceil` rather than standard rounding for sample size planning?
**Answer**: The sample size formula $n = \left(\frac{z_{\alpha/2} \cdot s}{E}\right)^2$ yields $1292.78$. Rounding down to 1292 would result in a margin of error strictly greater than $E = 0.50$ ($E_{1292} = 0.50015 > 0.50$). Taking the mathematical ceiling ($n = 1293$) guarantees that the margin of error remains strictly bounded within the desired tolerance.

### Q1.4: How did you verify that your simple linear regression implementation satisfied the ANOVA identity?
**Answer**: By definition of least squares with an intercept term, the residuals sum strictly to zero ($\sum e_i = 0$) and are orthogonal to fitted values ($\sum e_i \hat{y}_i = 0$). This guarantees the exact Pythagorean decomposition $TSS = ESS + RSS$. In our pipeline, $TSS = 67221.4285$, $ESS = 34925.6088$, and $RSS = 32295.8197$; their sum matches $TSS$ to within $10^{-12}$.

### Q1.5: Why did gradient descent require feature standardization on $X_4$, and how did you unstandardise parameters?
**Answer**: Unstandardised $X_4$ ranges from 3.7 to 43.1, producing elliptical loss contours where gradient descent oscillates violently or requires an imperceptibly tiny learning rate. Standardizing $X_4$ produces spherical contours allowing rapid convergence at $\eta = 0.1$. Because $y = w \cdot \frac{x - \bar{x}}{s_x} + b$, unstandardised parameters are recovered as $\text{Slope} = \frac{w}{s_x} = 0.6701$ and $\text{Intercept} = b - \text{Slope} \cdot \bar{x} = -3.6598$, matching OLS to 0.0000 gap.

### Q1.6: What statistical evidence proved that simple linear regression suffered from heteroscedasticity?
**Answer**: Sorting observations by fitted values $\hat{y}$ into lower and upper halves revealed that residual standard deviation expanded from 2.3155 in the lower half to 8.6891 in the upper half. The ratio $\frac{8.6891}{2.3155} = 3.7525$ heavily exceeds the standard diagnostic threshold of 2.0, confirming a widening residual funnel.

### Q1.7: Why did the log-log transformation stabilize residual variance, and what is its economic interpretation?
**Answer**: Log transformation compresses the right tail of positive, multiplicative variables. Fitting $\ln(\text{val}) = -2.4370 + 1.5030 \ln(X_4)$ produced lower and upper residual SDs of 0.5165 and 0.5369, yielding a spread ratio of 1.0396 (virtually 1.0). The slope $1.5030$ represents elasticity: a 1% increase in baseline order value yields a 1.503% increase in actual realized spending.

### Q1.8: Why did you compare Multiple Regression and Log-Log Regression using adjusted $R^2$ rather than raw $R^2$?
**Answer**: Raw $R^2$ monotonically increases with every added predictor, even purely random noise. Adjusted $R^2$ introduces a degrees-of-freedom penalty for model complexity: $R^2_{\text{adj}} = 1 - (1 - R^2) \frac{n - 1}{n - p - 1}$. Multiple regression with 7 predictors achieved $R^2_{\text{adj}} = 0.7403$ vs log-log $R^2_{\text{adj}} = 0.6561$.

### Q1.9: What edge case occurs if fitted values have massive ties during the residual funnel split?
**Answer**: If sorting fitted values using an unstable algorithm (e.g. quicksort), tied records at the median boundary shuffle non-deterministically across halves between operating systems. We enforce `kind='stable'` (mergesort), ensuring tied rows preserve original file order and produce byte-identical residual standard deviations.

### Q1.10: In multiple regression, why did you solve normal equations via `np.linalg.solve` instead of matrix inversion `np.linalg.inv`?
**Answer**: Matrix inversion $X^T X)^{-1}$ is computationally inefficient and numerically sensitive to near-collinearity. Solving the linear system $(X^T X) \beta = X^T y$ via LU/Cholesky decomposition avoids explicitly computing the inverse matrix, maintaining floating-point accuracy.

---

## 2. Part 2: Classification & Asymmetric Cost Optimization (10 Questions)

### Q2.1: Why was stratified sampling mandatory for the 70/30 train/test split?
**Answer**: In imbalanced datasets (21.25% churn, 170 positives), random sampling risks partition skew where the test set receives an unrepresentative churn rate. Stratification partitions both classes independently, guaranteeing exactly 119 churners in training (560 rows) and 51 churners in testing (240 rows), exactly matching the population proportion.

### Q2.2: Why must the feature scaler be fitted strictly on training data?
**Answer**: Fitting the scaler on the entire dataset incorporates test set mean and variance into training transformations, leaking future distribution information. Our implementation fits mean and sample SD on $X_{\text{train}}$ only, transforming $X_{\text{test}}$ with training parameters, ensuring zero data leakage.

### Q2.3: How does your logistic sigmoid prevent floating-point overflow?
**Answer**: Evaluating standard $\frac{1}{1 + e^{-z}}$ for large negative $z$ (e.g. $z = -800$) causes $e^{800}$ to overflow float64, raising RuntimeWarning and producing NaN. Our stable sigmoid branches: for $z \ge 0$, it computes $\frac{1}{1 + e^{-z}}$; for $z < 0$, it evaluates $\frac{e^z}{1 + e^z}$, guaranteeing numerical stability for all $z \in (-\infty, \infty)$.

### Q2.4: Why does binary cross-entropy require probability clipping?
**Answer**: If a model predicts $p = 0.0$ for an instance with true label $y = 1$, evaluating $\ln(p)$ results in $-\infty$, generating NaN during gradient descent. Clipping probabilities to $[10^{-15}, 1 - 10^{-15}]$ bounds maximum single-sample loss to $\approx 34.54$, keeping the loss function finite and differentiable.

### Q2.5: Why did learning rate $\eta = 0.1$ outperform $\eta = 0.01$ and $\eta = 1.0$?
**Answer**: Over 200 epochs, $\eta = 0.01$ converged too slowly (loss 0.5091). While $\eta = 1.0$ achieved slightly lower loss (0.3703 vs 0.3776), large step sizes risk overshooting and instability on less spherical loss landscapes. $\eta = 0.1$ provided smooth monotonic convergence without oscillations.

### Q2.6: How do you interpret the odds ratio of $X_3$ (discount share)?
**Answer**: The learned weight for standardized $X_3$ is $w_3 = +0.9013$, yielding an odds ratio $\text{OR} = \exp(0.9013) = 2.4629$. For every 1 standard deviation increase in discount order share, the odds of a customer churning increase by $146.29\%$ ($2.4629 - 1$), holding all other predictors constant.

### Q2.7: Why is blind majority accuracy (78.75%) a dangerous baseline in churn prediction?
**Answer**: A naive model that classifies all customers as retained achieves 78.75% accuracy simply because 78.75% of customers do not churn. However, its recall is 0.0%, failing to identify a single churner. Relying on accuracy masks catastrophic operational failure in imbalanced domains.

### Q2.8: How did you break ties during $K$-Nearest Neighbours cross-validation?
**Answer**: In 5-fold cross validation over odd $K \in [1, 21]$, if two candidate values of $K$ produce identical mean CV $F_1$-scores, the algorithm strictly selects the smaller $K$. Smaller $K$ retains higher localized sensitivity, whereas larger $K$ over-smoothes decision boundaries. In our data, $K=5$ achieved clear maximum CV $F_1 = 0.4541$.

### Q2.9: Why did optimal decision threshold shift from standard $\tau = 0.50$ down to $\tau^* = 0.15$?
**Answer**: Standard $\tau = 0.50$ assumes symmetric misclassification costs. However, our business reality is highly asymmetric: missing an actual churner costs INR 400 (lost customer margin), whereas sending a retention discount to a non-churner costs only INR 50. Lowering the threshold to $\tau^* = 0.15$ flags more borderline customers, raising recall from 47.06% to 88.24% and cutting test cost from INR 11,800 to INR 6,050.

### Q2.10: How does your grid search handle cost ties across candidate thresholds?
**Answer**: If two thresholds achieve identical total operational cost, the decision engine strictly selects the smaller threshold. In retention operations, ties favor aggressive customer capture (higher recall) over conservative flagging.

---

## 3. Part 3: Unsupervised Learning, Clustering, & PCA (10 Questions)

### Q3.1: Why did you initialize $K$-Means centroids using deterministic stride indices $0, \lfloor 800/K \rfloor, \dots$?
**Answer**: Random initialization (e.g. Forgy or $K$-Means++) produces non-deterministic cluster assignments that vary across runs, violating reproducibility. Using fixed evenly-spaced starting rows (`0, floor(800/K), 2*floor(800/K), ...`) guarantees deterministic convergence to identical cluster centers across all machines while spreading initial centroids across the dataset.

### Q3.2: How did you mathematically define the elbow point in $K$-Means?
**Answer**: Rather than subjective visual inspection, we compute the sequential WCSS drops $\Delta(K) = \text{WCSS}(K-1) - \text{WCSS}(K)$ and select the smallest $K$ where the next drop $\Delta(K+1) < 0.5 \cdot \Delta(K)$. From $K=2\to 3$, WCSS drops by $551.7543$, whereas from $K=3\to 4$ the drop falls to $252.8782 < 275.8772$, objectively identifying $K = 3$ as the elbow.

### Q3.3: What business personas emerged from the $K=3$ customer segmentation?
**Answer**:
- **Segment A (322 customers, 40.25%)**: Established loyal high-value shoppers. High tenure (44.07 months), high loyalty score (66.52), low complaints (0.76), ultra-low churn rate (1.86%), high order values (INR 1,353).
- **Segment B (255 customers, 31.88%)**: Moderate-tenure mid-tier shoppers. Average tenure (22.91 months), moderate discount share (0.3712), moderate complaints (1.76), moderate churn rate (23.53%).
- **Segment C (223 customers, 27.88%)**: At-risk early-life discount chasers. Short tenure (9.22 months), high complaint count (2.82), high discount share (0.5060), low loyalty score (38.78), high churn rate (46.64%).

### Q3.4: Why does single linkage clustering suffer from chaining?
**Answer**: Single linkage measures distance between the closest pair of points across clusters: $d(A, B) = \min_{a, b} d(a, b)$. A single intermediate data point between two distinct dense clusters can form a bridge, prematurely merging clusters into elongated chains. Cutting before the largest merge-height jump ($1.8404 \to 2.5015$, jump $0.6611$ at step 4) yields 2 clusters (a 5-point chain and 1 isolated point).

### Q3.5: How did the largest merge-height gap cut rule behave across Single, Complete, and Average linkage?
**Answer**: Across all three linkage rules on the first 6 rows, the largest merge-height jump occurs at merge step 4 (the final merge from 2 clusters down to 1): Single ($1.8404 \to 2.5015$), Complete ($2.8388 \to 4.4969$), and Average ($2.2537 \to 3.4573$). Cutting immediately before the largest jump yields 2 clusters for all three linkage methods.

### Q3.6: Why did you perform PCA on the correlation matrix rather than the covariance matrix?
**Answer**: Features $X_1 \dots X_5$ have vastly different measurement units and variances (e.g. loyalty score variance is $\approx 225$, whereas discount share variance is $\approx 0.05$). Performing PCA on unstandardised covariance would allow loyalty score to dominate the principal components purely due to unit scale. PCA on the correlation matrix standardises all features to unit variance, treating predictors equitably.

### Q3.7: What is the sum of eigenvalues in PCA, and why does this property hold?
**Answer**: The sum of eigenvalues of a matrix equals its trace. In a $p \times p$ correlation matrix, the diagonal elements represent the correlation of each variable with itself, which is identically 1.0. Therefore, $\text{Trace}(R) = p = 5.0$. In our decomposition, $\sum_{j=1}^5 \lambda_j = 2.6136 + 1.1828 + 0.5114 + 0.4055 + 0.2867 = 5.0000$.

### Q3.8: How did you resolve the eigenvector sign indeterminacy?
**Answer**: If $v$ is an eigenvector satisfying $R v = \lambda v$, then $-v$ is also a valid eigenvector because $R(-v) = \lambda(-v)$. To prevent sign flipping across linear algebra libraries, we enforced the deterministic convention: `if v[argmax(abs(v))] < 0: v = -v`. The element with the largest absolute loading is strictly forced to be positive.

### Q3.9: How many principal components are required to explain $\ge 80\%$ of total variance?
**Answer**: PC1 explains 52.27% ($\lambda_1 = 2.6136$), PC2 explains 23.66% ($\lambda_2 = 1.1828$), and PC3 explains 10.23% ($\lambda_3 = 0.5114$). The cumulative variance explained by the first three components is $52.27\% + 23.66\% + 10.23\% = 86.16\% \ge 80\%$. Thus, 3 components are required.

### Q3.10: How did PC1 score validate the $K$-Means customer segments?
**Answer**: Evaluating mean PC1 scores across segments revealed clean, monotonic separation: Segment A $= +1.7773$, Segment B $= -0.7831$, Segment C $= -1.6709$. PC1 effectively functions as a composite "Customer Relationship Health" index.

---

## 4. Part 4: Decision Trees, Ensembles, & Deep Architectures (10 Questions)

### Q4.1: Why does Shannon entropy use base 2 logarithms?
**Answer**: In information theory, using base 2 logarithms ($-\sum p_i \log_2 p_i$) measures uncertainty in "bits" (binary digits). A fair coin flip has entropy $-\frac{1}{2}\log_2(0.5) - \frac{1}{2}\log_2(0.5) = 1.0$ bit. Natural logarithms ($−\ln$) scale entropy by $\ln(2) \approx 0.6931$, violating the bit convention.

### Q4.2: Why did candidate split S4 ($X_1 \le 24$) achieve the highest Information Gain?
**Answer**: Customer tenure ($X_1 \le 24$ months) is the single strongest binary split discriminator of churn among the five candidates. Splitting at 24 months separates the 560 training rows into 312 early-tenure customers (109 churners, 34.94% churn rate) and 248 established customers (10 churners, 4.03% churn rate), maximizing Information Gain ($\text{IG} = 0.1252$) and Gini gain ($0.0606$).

### Q4.3: Why did you construct a 1-level decision stump instead of a fully grown decision tree?
**Answer**: Fully grown decision trees on 560 training samples overfit complex noisy interactions, exhibiting high variance. A 1-level stump represents an ultra-interpretable weak learner with zero variance, functioning as the foundational building block for AdaBoost and majority voting ensembles.

### Q4.4: In majority voting, why did you compare hard voting against soft probability voting?
**Answer**: Hard voting takes discrete integer predictions and counts votes, discarding prediction confidence. Soft voting averages continuous predicted probabilities before applying a threshold. In our test set, soft voting achieved a lower operational cost (INR 7,650 vs INR 8,650) because confidence averaging mitigated extreme confident errors from individual weak models.

### Q4.5: In AdaBoost, what is the mathematical derivation of weak learner weight $\alpha$?
**Answer**: AdaBoost minimizes the exponential loss $L = \sum_{i=1}^n \exp(-y_i F_m(x_i))$. Differentiating with respect to $\alpha_m$ and setting to zero yields $\alpha_m = \frac{1}{2} \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)$, where $\epsilon_m$ is the weighted training error. If a model has 50% error (random guessing), $\alpha = 0$; as $\epsilon \to 0$, $\alpha \to \infty$.

### Q4.6: Why is the total weight of misclassified samples after AdaBoost reweighting exactly 0.5?
**Answer**: The weight update rule scales misclassified samples by $e^\alpha = \sqrt{\frac{1-\epsilon}{\epsilon}}$ and correct samples by $e^{-\alpha} = \sqrt{\frac{\epsilon}{1-\epsilon}}$. The unnormalized weight of misclassified samples is $\epsilon \cdot \sqrt{\frac{1-\epsilon}{\epsilon}} = \sqrt{\epsilon(1-\epsilon)}$, and for correct samples is $(1-\epsilon) \cdot \sqrt{\frac{\epsilon}{1-\epsilon}} = \sqrt{\epsilon(1-\epsilon)}$. Because both sums are equal, normalising by their sum forces each group to have total weight $0.50$.

### Q4.7: What is the mathematical derivation of the Out-of-Bag (OOB) sample fraction in Random Forests?
**Answer**: In a bootstrap sample of size $N$ drawn with replacement, the probability that a specific row is not selected on a single draw is $1 - \frac{1}{N}$. Over $N$ independent draws, the probability of never being selected is $\left(1 - \frac{1}{N}\right)^N$. Taking the calculus limit as $N \to \infty$, $\lim_{N \to \infty} \left(1 - \frac{1}{N}\right)^N = e^{-1} \approx 0.367879$. For our training set $N = 560$, the exact finite value is $(559/560)^{560} = 0.3676$.

### Q4.8: In the CNN parameter calculation, why does Conv1 with 6 filters have 156 parameters?
**Answer**: The input has 1 channel (grayscale $28 \times 28$). Each of the 6 filters has spatial dimensions $5 \times 5$. The number of weights per filter is $5 \times 5 \times 1 = 25$, plus 1 bias parameter per filter. Therefore, each filter has $25 + 1 = 26$ parameters. Across 6 filters, the total learnable parameters are $6 \times 26 = 156$.

### Q4.9: Why does Max Pooling have exactly zero trainable parameters?
**Answer**: Max pooling is a deterministic fixed mathematical operation: it slides a window (e.g. $2 \times 2$) across feature maps and outputs the maximum numerical value. Because it contains no learnable weights or biases, its parameter count is identically zero.

### Q4.10: Why does flattening a $128 \times 128 \times 3$ image into a 1,024-unit dense layer require over 50 million parameters?
**Answer**: A $128 \times 128 \times 3$ RGB image flattens to $49,152$ input features. Connecting all $49,152$ inputs to $1,024$ hidden units creates a weight matrix of size $49,152 \times 1,024 = 50,331,648$ weights plus $1,024$ biases, totaling $50,332,672$ parameters in a single layer. This demonstrates why convolutional weight sharing and pooling are essential for spatial data.
