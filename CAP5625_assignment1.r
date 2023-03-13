library(tidyverse)

set.seed(NULL)

lambdas <- 10^seq(-2, 4)
alpha <- 0.1 / max(lambdas)
nIterations <- 1e5
nFolds <- 5

Credit <- read_csv("Credit_N400_p9.csv")

Credit <- Credit %>%
  mutate(Gender = ifelse(Gender == "Female", 1, 0),
         Student = ifelse(Student == "Yes", 1, 0),
         Married = ifelse(Married == "Yes", 1, 0))

X <- Credit %>% select(-Balance) %>% as.matrix()
Y <- Credit %>% select(Balance) %>% as.matrix()

features <- colnames(X)
N <- nrow(X)
p <- ncol(X)

# Standardize X and center Y
meanX <- colMeans(X)
sdX <- apply(X, 2, sd) 
X <- sweep(sweep(X, 2, meanX), 2, FUN = "/", sdX)
Y <- Y - mean(Y)

# Get path of betas across all tuning parameters
allBetas <- matrix(0, nrow = length(lambdas), ncol = p)
for(lambdaIndex in 1:length(lambdas)) {
  lambda <- lambdas[lambdaIndex]

  beta <- runif(p, -1, 1) # Initialize parameter vector in U[-1,1]
  
  # Perform gradient descent
  for(iter in 1:nIterations) {
    beta <- beta - 2 * alpha * (lambda * beta - t(X) %*% (Y - X %*% beta))
  }
  
  for(j in 1:p) {
    allBetas[lambdaIndex, j] = beta[j]
  }
}

# Collect values for ease of plotting
paramCol <- c()
lambdaCol <- c()
estimatesCol <- c()
for(i in 1:length(lambdas)) {
  for(j in 1:p) {
    paramCol <- c(paramCol, features[j])
    lambdaCol <- c(lambdaCol, log10(lambdas[i]))
    estimatesCol <- c(estimatesCol, allBetas[i,j])
  }
}
betasWithLambda <- tibble(Parameter = paramCol, Lambda = lambdaCol, Estimates = estimatesCol)

# Run cross validation across all tuning parameters
CVerror <- rep(0, length(lambdas))
foldid <- sample(rep(seq(nFolds), length = N)) # Randomly assign observations to folds
for(fold in 1:nFolds) {
  # Get validation and training sets for current fold
  valX <- X[foldid == fold,]
  valY <- Y[foldid == fold,]
  trainX <- X[foldid != fold,]
  trainY <- Y[foldid != fold,]
  
  # Standardize trainX and valX, center trainY and valY
  meanX <- colMeans(trainX)
  sdX <- apply(trainX, 2, sd) 
  meanY <- mean(trainY)
  trainX <- sweep(sweep(trainX, 2, meanX), 2, FUN = "/", sdX)
  valX <- sweep(sweep(valX, 2, meanX), 2, FUN = "/", sdX)
  trainY <- trainY - meanY
  valY <- valY - meanY
  
  for(lambdaIndex in 1:length(lambdas)) {
    lambda <- lambdas[lambdaIndex]
  
    beta <- runif(p, -1, 1) # Initialize parameter vector in U[-1,1]
    
    # Perform gradient descent
    for(iter in 1:nIterations) {
      beta <- beta - 2 * alpha * (lambda * beta - t(trainX) %*% (trainY - trainX %*% beta))
    }
    
    # Compute CV MSE for current fold
    CVerror[lambdaIndex] = CVerror[lambdaIndex] + mean( t(valY - valX %*% beta) %*% (valY - valX %*% beta) / length(valY) )
  }
}
CVerror = CVerror / nFolds # Mean CV MSE across folds

CVresults <- tibble(Lambda = c(log10(lambdas)), CVerror = c(CVerror))

# Deliverable 1
pdf("ProgrammingAssignment1_Deliverable1.pdf")
ggplot(betasWithLambda) +
  geom_line(aes(Lambda, Estimates, color = Parameter, linetype = Parameter), size = 1) +
  xlab("Log10(Lambda)") +
  ylab("Parameter estimates") +
  theme_bw() +
  theme(aspect.ratio = 1,
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
dev.off()

# Deliverable 2
pdf("ProgrammingAssignment1_Deliverable2.pdf")
ggplot(CVresults, aes(Lambda, CVerror)) +
  geom_line(size = 1) +
  xlab("Log10(Lambda)") + 
  ylab("Cross validation MSE") +
  theme_bw() +
  theme(aspect.ratio = 1,
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
dev.off()

# Deliverable 3
lambdaMin <- lambdas[ which(CVerror == min(CVerror), arr.ind = TRUE) ]

# Deliverable 4
bestBetas <- allBetas[ which(CVerror == min(CVerror), arr.ind = TRUE), ] 