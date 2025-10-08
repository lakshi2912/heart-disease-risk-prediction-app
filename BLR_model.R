## =========================================================
## HEART: Bayesian Logistic Regression with Random Intercept
## Full pipeline: preprocess -> compare RE -> select -> plots
## =========================================================

# Packages
library(dplyr)
library(ggplot2)
library(rstanarm)
library(bayesplot)
library(loo)
library(pROC)
library(tidyr)
library(caret)
set.seed(123)

#---------------------------------------
# 1) Data loading & preprocessing
#---------------------------------------
heart <- read.csv("heart.csv")

# Check overall structure
str(heart)

# Summary of all variables
summary(heart)

# If you want levels for categorical variables:
sapply(heart, function(x) if(is.factor(x)) levels(x) else NA)

# Alternatively, to see unique values of each variable:
sapply(heart, unique)

# For detailed summary of factor variables with counts:
lapply(heart, function(x) if(is.factor(x)) table(x) else NULL)

# Ensure outcome is numeric 0/1
# (rstanarm is fine with 0/1 numeric or factor with levels c(0,1))
if (is.factor(heart$target)) {
  heart$target <- as.numeric(as.character(heart$target))
}
heart$target <- as.integer(heart$target)

# Standardize continuous predictors (create *_z columns)
num_vars <- c("age", "trestbps", "chol", "thalach", "oldpeak")
heart <- heart %>%
  mutate(across(all_of(num_vars), ~ as.numeric(scale(.)), .names = "{.col}_z"))

# Categorical predictors as factors
cat_vars <- c("sex","cp","fbs","restecg","exang","slope","ca","thal")
heart <- heart %>% mutate(across(all_of(cat_vars), ~ as.factor(.)))

# Sanity check for NAs
print(colSums(is.na(heart)))

#---------------------------------------
# 2) Fit candidate random-intercept models
#    Compare with LOO (PSIS)
#---------------------------------------
group_vars <- c("sex","cp","fbs","restecg","exang","slope","ca","thal")

# Fixed-effect terms (character vector so we can remove the grouping var when needed)
fixed_terms <- c("age_z","trestbps_z","chol_z","thalach_z","oldpeak_z",
                 "sex","cp","fbs","restecg","exang","slope","ca","thal")

# Storage
models <- list()
loos   <- list()
loo_table <- data.frame(Group = character(),
                        elpd_loo = numeric(),
                        p_loo = numeric(),
                        LOOIC = numeric(),
                        stringsAsFactors = FALSE)

for (g in group_vars) {
  # Remove the grouping variable from fixed effects to avoid redundancy
  fe_terms <- setdiff(fixed_terms, g)
  rhs <- paste(c(fe_terms, paste0("(1 | ", g, ")")), collapse = " + ")
  fml <- as.formula(paste("target ~", rhs))
  
  cat("Fitting: (1 |", g, ")\n")
  fit <- stan_glmer(
    formula = fml,
    data = heart,
    family = binomial(link = "logit"),
    prior = normal(0, 2.5, autoscale = TRUE),
    prior_intercept = normal(0, 5, autoscale = TRUE),
    prior_covariance = decov(regularization = 2),
    chains = 4, iter = 4000, warmup = 1000, refresh = 0, seed = 123
  )
  
  models[[g]] <- fit
  
  # PSIS-LOO
  loo_g <- loo(fit)
  loos[[g]] <- loo_g
  
  # Store proper LOOIC (not elpd!)
  loo_table <- rbind(loo_table, data.frame(
    Group    = g,
    elpd_loo = loo_g$estimates["elpd_loo", "Estimate"],
    p_loo    = loo_g$estimates["p_loo",    "Estimate"],
    LOOIC    = loo_g$estimates["looic",    "Estimate"]
  ))
}

# Rank by LOOIC (lower = better)
loo_table <- loo_table %>% arrange(LOOIC)
print(loo_table)

#---------------------------------------
# 3) Pairwise model comparison (recommended)
#---------------------------------------
comp <- loo_compare(loos)  
print(comp)

# Build a readable comparison table with ΔLOOIC relative to the best
best_name <- rownames(comp)[1]
best_loo  <- loos[[best_name]]
best_looic <- best_loo$estimates["looic","Estimate"]

loo_table <- loo_table %>%
  mutate(`ΔLOOIC` = LOOIC - min(LOOIC)) %>%
  arrange(`ΔLOOIC`)

print(loo_table)

cat("\nBest random-intercept grouping variable (from loo_compare):", best_name, "\n")

best_model <- models[[best_name]]

summary(best_model)

#posterior_interval(best_model, prob = 0.95)

summary(best_model, probs = c(0.025, 0.5, 0.975))





#---------------------------------------
# 4) Recompute LOO with PSIS-k diagnostics & exact refits if needed
#---------------------------------------
# This triggers exact LOO refits if any Pareto k > 0.7
loo_best <- loo(best_model, k_threshold = 0.7)
print(loo_best)
# Pareto-k plot
plot(loo_best)

#---------------------------------------
# 5) Posterior predictive checks (model adequacy)
#---------------------------------------
# Density overlay (y vs. replicated y)
pp_check(best_model, type = "dens_overlay", ndraws = 100)
# Bars for binary outcome
pp_check(best_model, type = "bars", ndraws = 100)


#---------------------------------------
# 6) Save the final model for app / downstream use
#---------------------------------------
saveRDS(best_model, file = paste0("heart_model_final_",best_name,".rds"))
cat("Saved final model as: heart_model_final_",best_name, ".rds\n", sep = "")

#---------------------------------------
# 7) Graphs for the paper
#    (a) LOOIC with SE bars
#---------------------------------------
# Extract SE for LOOIC properly
get_looic_se <- function(loo_obj) {
  # LOOIC = -2 * elpd_loo; Var(LOOIC) = 4 * Var(elpd_loo)
  se_elpd <- loo_obj$estimates["elpd_loo","SE"]
  2 * se_elpd
}

loo_table$SE_LOOIC <- sapply(loo_table$Group, function(g) get_looic_se(loos[[g]]))

ggplot(loo_table, aes(x = reorder(Group, LOOIC), y = LOOIC)) +
  geom_col(width = 0.6) +
  geom_errorbar(aes(ymin = LOOIC - SE_LOOIC, ymax = LOOIC + SE_LOOIC), width = 0.15) +
  coord_flip() +
  labs(title = "Model comparison by LOOIC (lower is better)",
       x = "Random-intercept grouping variable",
       y = "LOOIC") +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())+
  theme_minimal(base_size = 13)

ggplot(loo_table, aes(x = reorder(Group, LOOIC), y = LOOIC)) +
  geom_col(width = 0.6, fill = "lightblue") +   # �� set bar color
  geom_errorbar(aes(ymin = LOOIC - SE_LOOIC, ymax = LOOIC + SE_LOOIC),
                width = 0.15, color = "black") + # optional: keep error bars visible
  coord_flip() +
  labs(title = "Model comparison by LOOIC",
       x = "Random-intercept grouping variable",
       y = "LOOIC") +
  theme_minimal(base_size = 13) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())

ggplot(loo_table, aes(x = reorder(Group, LOOIC), y = LOOIC)) +
  geom_col(width = 0.6, fill = "skyblue1") +   # �� lighter shade of blue
  geom_errorbar(aes(ymin = LOOIC - SE_LOOIC, ymax = LOOIC + SE_LOOIC),
                width = 0.15, color = "black") +
  coord_flip() +
  labs(title = "Model comparison by LOOIC (lower is better)",
       x = "Random-intercept grouping variable",
       y = "LOOIC") +
  theme_minimal(base_size = 13) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())


#---------------------------------------
#    (b) Fixed-effect intervals (posterior) for best model
#---------------------------------------
# Convert posterior draws to matrix
post <- as.matrix(best_model)

#---------------------------------------
#    (c) Random-intercept caterpillar plot
#---------------------------------------

# Extract posterior draws for random intercepts
post <- as.data.frame(as.matrix(best_model))

# Random intercepts are named like: b[(Intercept) ca:0], b[(Intercept) ca:1], ...
ri_names <- grep("^b\\[\\(Intercept\\) ca:", colnames(post), value = TRUE)

# Compute posterior summaries: mean, 2.5%, 97.5%
ri_summary <- post %>%
  select(all_of(ri_names)) %>%
  summarise(across(everything(), list(
    mean = ~mean(.),
    l95  = ~quantile(., 0.025),
    u95  = ~quantile(., 0.975)
  ), .names = "{.col}_{.fn}")) %>%
  pivot_longer(everything(),
               names_to = c("level", ".value"),
               names_pattern = "b\\[\\(Intercept\\) ca:(.)\\]_(.*)") %>%
  mutate(level = as.factor(level))

# Caterpillar plot
ggplot(ri_summary, aes(x = reorder(level, mean), y = mean)) +
  geom_point(size = 3, color = "blue") +
  geom_errorbar(aes(ymin = l95, ymax = u95), width = 0.2, color = "blue") +
  coord_flip() +
  labs(
    title = "Random Intercepts by ca (posterior mean & 95% CrI)",
    x = "ca (Number of Major Vessels)",
    y = "Random intercept (log-odds scale)"
  ) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank())
theme_minimal(base_size = 13)

#---------------------------------------
# 8) (Optional) ROC/AUC on training data
#    Uses posterior mean predicted probability
#---------------------------------------
pred_prob_draws <- posterior_epred(best_model)   # draws x N
prob_mean <- colMeans(pred_prob_draws)
roc_obj <- roc(response = heart$target, predictor = prob_mean)
plot(roc_obj, main = paste0("ROC (AUC = ", round(auc(roc_obj), 3), ")"))
abline(a = 0, b = 1, lty = 2, col = "gray")

#---------------------------------------
# 9) Concise textual outputs to paste in Results
#---------------------------------------
cat("\n--- Summary for Results section ---\n")
cat("Best random-intercept grouping:", best_name, "\n")
cat("Best model LOOIC:", round(loo_best$estimates["looic","Estimate"], 2), 
    " (MCSE:", round(loo_best$estimates["looic","SE"], 2), ")\n")
cat("All Pareto k < 0.7? ", all(loo_best$diagnostics$pareto_k < 0.7,na.rm=TRUE), "\n")
cat("AUC (training): ", round(auc(roc_obj), 3), "\n")

# Assuming your model is saved as 'best_model'
posterior_preds <- posterior_linpred(best_model, transform = TRUE)
# transform = TRUE converts log-odds to probabilities

mean_probs <- apply(posterior_preds, 2, mean)

pred_class <- ifelse(mean_probs > 0.5, 1, 0)

observed <-heart$target
accuracy <- mean(pred_class == observed)
accuracy

confusionMatrix(factor(pred_class), factor(observed))
