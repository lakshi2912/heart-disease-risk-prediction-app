# Load required libraries
library(ggplot2)
library(dplyr)

# Assuming your dataset is heart_data and outcome variable is 'target'
# target: 1 = disease, 0 = no disease

# -------------------------------
# 1. Histograms for Continuous Variables
# -------------------------------
continuous_vars <- c("age", "trestbps", "chol", "thalach", "oldpeak")

for (var in continuous_vars) {
  p <- ggplot(heart_data, aes(x = .data[[var]])) +
    geom_histogram(bins = 30, fill = "skyblue", color = "black", alpha = 0.7) +
    geom_density(aes(y = ..count..), color = "red", size = 1) +
    labs(title = paste("Histogram and Density of", var),
         x = var, y = "Count") +
    theme_minimal()
  print(p)
}

# -------------------------------
# 2. Boxplots for Continuous vs Target
# -------------------------------
for (var in continuous_vars) {
  p <- ggplot(heart_data, aes(x = factor(target), y = .data[[var]], fill = factor(target))) +
    geom_boxplot(alpha = 0.6) +
    labs(title = paste("Boxplot of", var, "by Heart Disease Status"),
         x = "Heart Disease (0 = No, 1 = Yes)", y = var) +
    scale_fill_manual(values = c("0" = "skyblue", "1" = "tomato"),
                      labels = c("No Disease", "Disease")) +
    theme_minimal()
  print(p)
}

# -------------------------------
# 3. Bivariate Statistical Tests
# -------------------------------

# For continuous predictors: independent t-tests
for (var in continuous_vars) {
  test <- t.test(heart_data[[var]] ~ heart_data$target)
  cat("T-test for", var, "\n")
  print(test)
  cat("\n----------------------------\n")
}

# For categorical predictors: Chi-square tests
categorical_vars <- c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")

for (var in categorical_vars) {
  tab <- table(heart_data[[var]], heart_data$target)
  chi_test <- chisq.test(tab)
  cat("Chi-square test for", var, "\n")
  print(chi_test)
  cat("\n----------------------------\n")
}
