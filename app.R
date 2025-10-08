# app.R
# Improved Thesis-Ready Shiny App (single-file)
# Bayesian Logistic Regression-Based Personalized Health Advisory System
# Make sure 'heart_model_final_ca.rds' is in the same folder (or adjust model_path)

# ===== Packages =====
library(shiny)
library(shinydashboard)
library(ggplot2)
library(pROC)
library(rstanarm)
library(dplyr)
library(tidyr)
library(glue)
library(flexdashboard) # for gauge
# optional for PDF: rmarkdown
# library(rmarkdown)

# ===== Config =====
model_path <- "heart_model_final_ca.rds"    # adjust if needed
# Optional precomputed posterior draws file (speeds up prediction on hosting)
posterior_draws_path <- "posterior_draws_ca.rds"  # optional: save posterior_epred(model, draws = <large>) offline

# How many posterior draws to use for on-the-fly posterior_epred calls (lower= faster)
DEFAULT_POSTERIOR_DRAWS <- 600

# ===== Helper: safe model load =====
safe_load_model <- function(path) {
  if (!file.exists(path)) stop(glue("Model file not found at: {path}"))
  obj <- readRDS(path)
  # basic check
  if (!("stanreg" %in% class(obj))) {
    warning("Loaded object does not look like an rstanarm model (class: {paste(class(obj), collapse=', ')}). Proceeding anyway.")
  }
  obj
}

# ===== Try to load model (with friendly error) =====
model <- tryCatch(
  safe_load_model(model_path),
  error = function(e) {
    message("Failed to load model: ", e$message)
    NULL
  }
)
if (is.null(model)) {
  stop("Model could not be loaded. Place the model RDS at '", model_path, "' and restart the app.")
}

# ===== Helper: mean/sd for scaling (from model$data) =====
m_sd <- function(x) list(m = mean(x, na.rm = TRUE), s = sd(x, na.rm = TRUE))
ms <- list(
  age      = m_sd(model$data$age),
  trestbps = m_sd(model$data$trestbps),
  chol     = m_sd(model$data$chol),
  thalach  = m_sd(model$data$thalach),
  oldpeak  = m_sd(model$data$oldpeak)
)

# ===== Baseline values (for contributions) =====
baseline <- list(
  age = median(model$data$age, na.rm = TRUE),
  trestbps = median(model$data$trestbps, na.rm = TRUE),
  chol = median(model$data$chol, na.rm = TRUE),
  thalach = median(model$data$thalach, na.rm = TRUE),
  oldpeak = median(model$data$oldpeak, na.rm = TRUE),
  sex = levels(model$data$sex)[1],
  cp = levels(model$data$cp)[1],
  fbs = levels(model$data$fbs)[1],
  restecg = levels(model$data$restecg)[1],
  exang = levels(model$data$exang)[1],
  slope = levels(model$data$slope)[1],
  ca = levels(model$data$ca)[1],
  thal = levels(model$data$thal)[1]
)

# ===== Recommendation master list =====
recommendations_master <- c(
  "Maintain a balanced diet rich in fruits, vegetables, and whole grains.",
  "Engage in at least 150 minutes of moderate aerobic activity per week.",
  "Avoid smoking and limit alcohol consumption.",
  "Monitor blood pressure and cholesterol regularly.",
  "Maintain a healthy weight (BMI 18.5–24.9) where appropriate.",
  "Manage stress with relaxation techniques (meditation, yoga).",
  "Ensure 7–8 hours of quality sleep each night.",
  "Follow up with regular medical check-ups and blood tests.",
  "Adhere to prescribed medications and clinical advice.",
  "Reduce salt and processed food intake; prefer unsaturated fats."
)

# ===== Utility: transform a raw row to model features (z-scored + factors) =====
mk_newdata_from_raw <- function(raw_row) {
  data.frame(
    age_z      = (as.numeric(raw_row$age)      - ms$age$m)      / ifelse(ms$age$s == 0, 1, ms$age$s),
    trestbps_z = (as.numeric(raw_row$trestbps) - ms$trestbps$m) / ifelse(ms$trestbps$s == 0, 1, ms$trestbps$s),
    chol_z     = (as.numeric(raw_row$chol)     - ms$chol$m)     / ifelse(ms$chol$s == 0, 1, ms$chol$s),
    thalach_z  = (as.numeric(raw_row$thalach)  - ms$thalach$m)  / ifelse(ms$thalach$s == 0, 1, ms$thalach$s),
    oldpeak_z  = (as.numeric(raw_row$oldpeak)  - ms$oldpeak$m)  / ifelse(ms$oldpeak$s == 0, 1, ms$oldpeak$s),
    sex        = factor(as.character(raw_row$sex),     levels = levels(model$data$sex)),
    cp         = factor(as.character(raw_row$cp),      levels = levels(model$data$cp)),
    fbs        = factor(as.character(raw_row$fbs),     levels = levels(model$data$fbs)),
    restecg    = factor(as.character(raw_row$restecg), levels = levels(model$data$restecg)),
    exang      = factor(as.character(raw_row$exang),   levels = levels(model$data$exang)),
    slope      = factor(as.character(raw_row$slope),   levels = levels(model$data$slope)),
    ca         = factor(as.character(raw_row$ca),      levels = levels(model$data$ca)),
    thal       = factor(as.character(raw_row$thal),    levels = levels(model$data$thal)),
    stringsAsFactors = FALSE
  )
}

# ===== Optionally load precomputed posterior draws to speed up predictions =====
precomputed_post_draws <- NULL
if (file.exists(posterior_draws_path)) {
  # Expect an object like a matrix of dims draws x Ntrain OR a list with function to sample
  try({
    precomputed_post_draws <- readRDS(posterior_draws_path)
    message("Loaded precomputed posterior draws from: ", posterior_draws_path)
  }, silent = TRUE)
}

# ===== UI =====
ui <- dashboardPage(
  dashboardHeader(title = "Heart Disease Risk Advisory (Bayesian)"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("About", tabName = "about", icon = icon("info-circle")),
      menuItem("Patient View", tabName = "patient", icon = icon("user")),
      menuItem("Clinician View", tabName = "clinician", icon = icon("stethoscope"))
    )
  ),
  dashboardBody(
    tabItems(
      # ABOUT
      tabItem(tabName = "about",
              fluidRow(
                box(title = "About this application", width = 12, status = "info", solidHeader = TRUE,
                    h3("Bayesian Logistic Regression-Based Personalized Health Advisory System"),
                    p("This application predicts a patient's probability of heart disease using a Bayesian logistic regression model (rstanarm)."),
                    p("Key features:"),
                    tags$ul(
                      tags$li("Posterior predictive distribution of risk (with 95% credible interval)."),
                      tags$li("Top contributing factors compared to a baseline patient."),
                      tags$li("Simple personalized recommendations."),
                      tags$li("Clinician diagnostics: ROC and calibration and posterior summaries.")
                    ),
                    br(),
                    h4("Variable descriptions and coding:"),
                    tags$ul(
                      tags$li("**sex**: 0 = Female, 1 = Male"),
                      tags$li("**cp (Chest pain type)**: 0 = Typical angina, 1 = Atypical angina, 2 = Non-anginal pain, 3 = Asymptomatic"),
                      tags$li("**fbs (Fasting blood sugar)**: 0 = ≤120 mg/dl, 1 = >120 mg/dl"),
                      tags$li("**restecg (Resting ECG)**: 0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy"),
                      tags$li("**exang (Exercise induced angina)**: 0 = No, 1 = Yes"),
                      tags$li("**slope (Slope of peak exercise ST)**: 0 = Upsloping, 1 = Flat, 2 = Downsloping"),
                      tags$li("**ca (Number of major vessels)**: 0 = none, 1 = one, 2 = two, 3 = three, 4 = four"),
                      tags$li("**thal (Thalassemia)**: 0 = Normal, 1 = Fixed defect, 2 = Reversible defect, 3 = Other")
                    ),
                    p(strong("Notes:")),
                    p("• The app is intended for educational / research demonstrations. Clinical decisions require comprehensive evaluation and qualified medical advice."),
                    p("• For hosting (shinyapps.io), heavy stanfit objects may be slow to load; consider precomputing draws offline and saving to 'posterior_draws_ca.rds'.")
                )
              )
      ),
      
      # PATIENT
      tabItem(tabName = "patient",
              fluidRow(
                box(title = "Patient information", width = 4, status = "primary", solidHeader = TRUE,
                    helpText("Enter patient-level predictors and click 'Predict'."),
                    
                    numericInput("age", "Age (years)", value = 55, min = 18, max = 120, step = 1),
                    selectInput("sex", "Sex", choices = setNames(levels(model$data$sex), levels(model$data$sex))),
                    selectInput("cp", "Chest pain type", choices = setNames(levels(model$data$cp), levels(model$data$cp))),
                    numericInput("trestbps", "Resting blood pressure (mmHg)", value = round(ms$trestbps$m), min = 50, max = 300, step = 1),
                    numericInput("chol", "Serum cholesterol (mg/dl)", value = round(ms$chol$m), min = 50, max = 800, step = 1),
                    selectInput("fbs", "Fasting blood sugar > 120 mg/dl", choices = setNames(levels(model$data$fbs), levels(model$data$fbs))),
                    selectInput("restecg", "Resting ECG", choices = setNames(levels(model$data$restecg), levels(model$data$restecg))),
                    numericInput("thalach", "Max heart rate achieved", value = round(ms$thalach$m), min = 30, max = 300, step = 1),
                    selectInput("exang", "Exercise induced angina", choices = setNames(levels(model$data$exang), levels(model$data$exang))),
                    numericInput("oldpeak", "ST depression (oldpeak)", value = round(ms$oldpeak$m,2), min = 0, max = 10, step = 0.1),
                    selectInput("slope", "Slope of peak exercise ST", choices = setNames(levels(model$data$slope), levels(model$data$slope))),
                    selectInput("ca", "Number of major vessels (ca)", choices = setNames(levels(model$data$ca), levels(model$data$ca))),
                    selectInput("thal", "Thalassemia", choices = setNames(levels(model$data$thal), levels(model$data$thal))),
                    hr(),
                    numericInput("draws", "Posterior draws for prediction (lower = faster)", value = DEFAULT_POSTERIOR_DRAWS, min = 50, max = 2000, step = 50),
                    radioButtons("pred_type", "Prediction Type:",
                                 choices = c("Marginal (default)" = "marginal",
                                             "Conditional (if ca group exists)" = "conditional"),
                                 selected = "marginal"),
                    
                    actionButton("predictBtn", "Predict risk", class = "btn-success"),
                    br(), br(),
                    downloadButton("downloadReport", "Download report (PDF or TXT)")
                ),
                box(title = "Prediction & interpretation", width = 8, status = "success", solidHeader = TRUE,
                    uiOutput("riskText"),
                    gaugeOutput("riskGauge", height = "220px"),
                    plotOutput("riskDensity", height = "260px"),
                    hr(),
                    h4("Top contributing factors (vs baseline)"),
                    tableOutput("contributors_tbl")
                )
              ),
              fluidRow(
                box(title = "Personalized recommendations", width = 12, status = "warning", solidHeader = TRUE,
                    uiOutput("adviceText"))
              )
      ),
      
      # CLINICIAN
      tabItem(tabName = "clinician",
              fluidRow(
                box(title = "ROC curve (training data)", width = 6, status = "info", solidHeader = TRUE,
                    plotOutput("rocPlot", height = 320),
                    uiOutput("aucValue")
                ),
                box(title = "Calibration by decile (training data)", width = 6, status = "info", solidHeader = TRUE,
                    plotOutput("calPlot", height = 320)
                )
              ),
              fluidRow(
                box(title = "Posterior summary (selected population-level betas)", width = 12, status = "primary", solidHeader = TRUE,
                    verbatimTextOutput("posteriorSummary")
                )
              )
      )
    )
  )
)

# ===== SERVER =====
server <- function(input, output, session) {
  
  # small validator for key inputs
  validate_inputs <- function() {
    shiny::validate(
      need(!is.null(input$age) && input$age > 0 && input$age < 120, "Enter a valid age."),
      need(!is.null(input$trestbps) && input$trestbps > 30 && input$trestbps < 300, "Enter a valid resting BP."),
      need(!is.null(input$chol) && input$chol > 50 && input$chol < 1000, "Enter a valid cholesterol."),
      need(!is.null(input$thalach) && input$thalach > 30 && input$thalach < 300, "Enter a valid max HR.")
    )
  }
  
  # build raw row from inputs
  build_raw_input <- reactive({
    list(
      age = input$age,
      trestbps = input$trestbps,
      chol = input$chol,
      thalach = input$thalach,
      oldpeak = input$oldpeak,
      sex = as.character(input$sex),
      cp = as.character(input$cp),
      fbs = as.character(input$fbs),
      restecg = as.character(input$restecg),
      exang = as.character(input$exang),
      slope = as.character(input$slope),
      ca = as.character(input$ca),
      thal = as.character(input$thal)
    )
  })
  
  # function to compute posterior draws for a single newdata row
  compute_posterior_draws <- function(nd_row, draws = DEFAULT_POSTERIOR_DRAWS) {
    # If user supplied precomputed draws for training data and newdata is identical to some training rows,
    # we could reuse. But general approach: call posterior_epred with specified draws.
    # Use tryCatch to gracefully handle failures on hosting.
    out <- tryCatch({
      # if user provided a precomputed draws object that supports sampling, you could use it here.
      posterior_epred(model, newdata = nd_row, draws = draws)
    }, error = function(e) {
      # fallback attempt with fewer draws
      tryCatch({
        posterior_epred(model, newdata = nd_row, draws = min(200, draws))
      }, error = function(e2) {
        stop("Posterior prediction failed: ", e2$message)
      })
    })
    out
  }
  
  # Observe predict button
  observeEvent(input$predictBtn, {
    # validate
    validate_inputs()
    # convert raw to model's expected newdata
    raw <- build_raw_input()
    nd <- mk_newdata_from_raw(as.data.frame(raw, stringsAsFactors = FALSE))
    
    # use precomputed draws if available and appropriate (not implemented as automatic mapping here)
    draws_to_use <- as.integer(input$draws)
    draws_to_use <- ifelse(is.na(draws_to_use) || draws_to_use <= 0, DEFAULT_POSTERIOR_DRAWS, draws_to_use)
    
    # ---- Marginal vs Conditional predictions ----
    draws_mat <- NULL
    try({
      if (input$pred_type == "marginal") {
        # Marginal: integrates over random intercepts
        draws_mat <- posterior_linpred(model, newdata = nd,
                                       transform = TRUE, re.form = NA,
                                       draws = draws_to_use)
      } else {
        # Conditional: uses group-specific intercept if ca exists
        draws_mat <- tryCatch({
          posterior_linpred(model, newdata = nd,
                            transform = TRUE, re.form = NULL,
                            draws = draws_to_use)
        }, error = function(e) {
          showNotification("Group-specific ca not found. Using marginal prediction.", type = "warning")
          posterior_linpred(model, newdata = nd,
                            transform = TRUE, re.form = NA,
                            draws = draws_to_use)
        })
      }
    }, silent = TRUE)
    
    if (is.null(draws_mat)) {
      # show user-friendly error in UI
      output$riskText <- renderUI({
        div(style = "color: red;", "Prediction failed — posterior prediction could not be computed. Check server logs.")
      })
      return(invisible(NULL))
    }
    
    
    risk_draws <- as.numeric(draws_mat[,1])
    risk_mean <- mean(risk_draws)
    risk_ci <- quantile(risk_draws, c(0.025, 0.975))
    
    risk_category <- if (risk_mean < 0.30) "Low risk (<30%)" else if (risk_mean < 0.60) "Moderate risk (30–60%)" else "High risk (>60%)"
    
    # Render summary UI
    output$riskText <- renderUI({
      HTML(glue(
        "<h4>Predicted risk: <b>{round(100*risk_mean,1)}%</b></h4>
         <small>95% CrI: {round(100*risk_ci[1],1)}% – {round(100*risk_ci[2],1)}%</small>
         <p><b>Category:</b> {risk_category}</p>"
      ))
    })
    
    # Gauge
    output$riskGauge <- renderGauge({
      gauge(risk_mean*100, min = 0, max = 100,
            sectors = gaugeSectors(success = c(0,30), warning = c(30,60), danger = c(60,100)))
    })
    
    # Posterior density plot
    output$riskDensity <- renderPlot({
      ggplot(tibble::tibble(risk = risk_draws), aes(x = risk)) +
        geom_density(fill = "#66c2a5", alpha = 0.6) +
        geom_vline(xintercept = risk_mean, linetype = "dashed") +
        geom_vline(xintercept = risk_ci, linetype = "dotted") +
        scale_x_continuous(labels = scales::percent_format(accuracy = 1)) +
        labs(title = "Posterior Distribution of Patient Risk", x = "Risk (probability)", y = "Density") +
        theme_minimal(base_size = 13)
    })
    
    # Feature contributions (simple one-at-a-time vs baseline)
    predictors <- c("age","trestbps","chol","thalach","oldpeak","sex","cp","fbs","restecg","exang","slope","ca","thal")
    raw_inputs <- as.data.frame(raw, stringsAsFactors = FALSE)
    contributions <- sapply(predictors, function(var) {
      tmp <- raw_inputs
      tmp[[var]] <- baseline[[var]]
      nd_var <- mk_newdata_from_raw(tmp)
      mean(posterior_epred(model, newdata = nd_var, draws = min(250, draws_to_use))[,1])
    }, simplify = TRUE)
    deltas <- risk_mean - as.numeric(contributions)
    contrib_df <- tibble::tibble(
      variable = predictors,
      delta = deltas
    ) %>% mutate(direction = ifelse(delta > 0, "Increases risk", "Decreases risk"),
                 abs_delta = abs(delta)) %>% arrange(desc(abs_delta))
    
    top3 <- head(contrib_df, 3)
    
    output$contributors_tbl <- renderTable({
      top3 %>% mutate(delta_pct = round(delta * 100, 2),
                      abs_delta_pct = paste0(round(abs_delta * 100, 2), "%")) %>%
        select(variable, delta_pct, direction, abs_delta_pct) %>%
        dplyr::rename(Variable = variable, `Δ Probability (%)` = delta_pct, Effect = direction, `|Δ| (%)` = abs_delta_pct)
    }, digits = 3)
    
    # Recommendations (simple rule-based)
    recs <- c()
    if (risk_mean >= 0.70) recs <- c(recs, "Immediate cardiology consultation recommended.")
    else if (risk_mean >= 0.50) recs <- c(recs, "Schedule a cardiology appointment soon for further evaluation.")
    else recs <- c(recs, "Maintain healthy habits and regular annual checkups.")
    
    if (as.numeric(raw_inputs$chol) >= 240) recs <- c(recs, "High cholesterol — consider dietary changes and discuss lipid-lowering therapy with your clinician.")
    if (as.numeric(raw_inputs$trestbps) >= 140) recs <- c(recs, "Elevated resting blood pressure — regular monitoring and hypertension management recommended.")
    if (as.numeric(raw_inputs$thalach) < 120) recs <- c(recs, "Low maximum heart rate — consider graded exercise testing.")
    if (as.numeric(raw_inputs$exang) %in% c(1,"1")) recs <- c(recs, "Exercise-induced angina reported — avoid strenuous activity until clinically cleared.")
    if (as.numeric(raw_inputs$oldpeak) >= 2) recs <- c(recs, "Marked ST depression — consider further cardiac testing (stress imaging).")
    
    if (length(recs) < 5) recs <- unique(c(recs, head(recommendations_master, 5 - length(recs))))
    if (length(recs) > 10) recs <- head(recs, 10)
    
    output$adviceText <- renderUI({
      HTML(paste0("<ul>", paste0("<li>", recs, "</li>", collapse = ""), "</ul>"))
    })
    # Downloadable report (PDF via rmarkdown if available, else text)
    # We will create a small Rmd on the fly if rmarkdown is available
    output$downloadReport <- downloadHandler(
      filename = function() {
        paste0("heart_risk_report_", Sys.Date(), if (requireNamespace("rmarkdown", quietly = TRUE)) ".pdf" else ".txt")
      },
      content = function(file) {
        # data snapshot to include in report
        snapshot <- list(
          risk_mean = risk_mean,
          risk_ci = risk_ci,
          risk_category = risk_category,
          raw_inputs = raw_inputs,
          top3 = top3,
          recs = recs,
          draws = risk_draws
        )
        
        if (requireNamespace("rmarkdown", quietly = TRUE)) {
          # Create temporary Rmd
          tmp_rmd <- tempfile(fileext = ".Rmd")
          rmd_text <- c(
            "---",
            "title: \"Heart Disease Risk Report\"",
            "output: pdf_document",
            "---",
            "",
            "```{r setup, include=FALSE}",
            "knitr::opts_chunk$set(echo=FALSE, warning=FALSE, message=FALSE)",
            "library(ggplot2); library(knitr); library(scales)",
            "```",
            "",
            "## Summary",
            "",
            glue::glue("**Predicted risk:** {round(100*snapshot$risk_mean,1)}%  "),
            glue::glue("**95% CrI:** {round(100*snapshot$risk_ci[1],1)}% – {round(100*snapshot$risk_ci[2],1)}%  "),
            glue::glue("**Category:** {snapshot$risk_category}"),
            "",
            "## Patient inputs",
            "```{r}",
            "df <- data.frame(",
            glue::glue("Age = {snapshot$raw_inputs$age}, Sex = '{snapshot$raw_inputs$sex}',"),
            glue::glue("CP = '{snapshot$raw_inputs$cp}', Trestbps = {snapshot$raw_inputs$trestbps}, Chol = {snapshot$raw_inputs$chol},"),
            glue::glue("FBS = '{snapshot$raw_inputs$fbs}', RestECG = '{snapshot$raw_inputs$restecg}', Thalach = {snapshot$raw_inputs$thalach},"),
            glue::glue("Exang = '{snapshot$raw_inputs$exang}', Oldpeak = {snapshot$raw_inputs$oldpeak},"),
            glue::glue("Slope = '{snapshot$raw_inputs$slope}', CA = '{snapshot$raw_inputs$ca}', Thal = '{snapshot$raw_inputs$thal}')"),
            "kable(df)",
            "```",
            "",
            "## Posterior risk distribution",
            "```{r}",
            "d <- data.frame(risk = as.numeric(snapshot$draws))",
            "ggplot(d, aes(x = risk)) + geom_density(fill = '#66c2a5', alpha = 0.6) +",
            "  geom_vline(xintercept = mean(d$risk), linetype = 2) +",
            "  scale_x_continuous(labels = percent_format(accuracy = 1)) +",
            "  labs(x='Risk (probability)', y='Density') + theme_minimal(base_size=12)",
            "```",
            "",
            "## Top contributing factors",
            "```{r}",
            "t <- snapshot$top3",
            "t$`Δ Probability (%)` <- round(t$delta*100, 2)",
            "t$`|Δ| (%)` <- paste0(round(abs(t$delta)*100,2),'%')",
            "kable(t[,c('variable','Δ Probability (%)','direction','|Δ| (%)')], col.names = c('Variable','Δ Probability (%)','Effect','|Δ| (%)'))",
            "```",
            "",
            "## Recommendations",
            "```{r}",
            "cat(paste0('- ', unlist(snapshot$recs), collapse='\\n'))",
            "```"
          )
          writeLines(rmd_text, tmp_rmd, useBytes = TRUE)
          # render to output file
          # rmarkdown::render may require pandoc; hosting may or may not have it
          tryCatch({
            rmarkdown::render(tmp_rmd, output_file = file, quiet = TRUE, envir = new.env(parent = globalenv()))
          }, error = function(e) {
            # fallback to text
            writeLines(capture.output(print(snapshot)), file)
          })
        } else {
          # fallback: write readable text
          txt <- c(
            "Heart Disease Risk Report",
            "-------------------------",
            paste0("Predicted risk: ", round(100 * snapshot$risk_mean, 2), "%"),
            paste0("95% CrI: ", round(100 * snapshot$risk_ci[1], 2), "% - ", round(100 * snapshot$risk_ci[2], 2), "%"),
            paste0("Category: ", snapshot$risk_category),
            "",
            "Patient inputs:",
            paste0("  Age: ", snapshot$raw_inputs$age),
            paste0("  Sex: ", snapshot$raw_inputs$sex),
            paste0("  CP: ", snapshot$raw_inputs$cp),
            paste0("  Resting BP: ", snapshot$raw_inputs$trestbps),
            paste0("  Cholesterol: ", snapshot$raw_inputs$chol),
            paste0("  Thalach: ", snapshot$raw_inputs$thalach),
            paste0("  Exang: ", snapshot$raw_inputs$exang),
            paste0("  Oldpeak: ", snapshot$raw_inputs$oldpeak),
            "",
            "Top contributing factors:",
            paste0("  ", snapshot$top3$variable, ": ΔProb=", round(snapshot$top3$delta*100,2), "% (", snapshot$top3$direction, ")", collapse = "\n"),
            "",
            "Recommendations:",
            paste0("- ", snapshot$recs, collapse = "\n")
          )
          writeLines(txt, con = file)
        }
      }
    ) 
  }) # observeEvent predictBtn
  
  # ===== Clinician plots & posterior summary (based on training data contained in model$data) =====
  output$rocPlot <- renderPlot({
    if (!"target" %in% names(model$data)) {
      plot.new(); text(0.5, 0.5, "Training outcome 'target' not found in model$data")
      return()
    }
    pred_draws <- tryCatch(posterior_epred(model, newdata = model$data), error = function(e) NULL)
    if (is.null(pred_draws)) {
      plot.new(); text(0.5, 0.5, "Posterior predictions unavailable for clinician ROC (model may be too large to sample here).")
      return()
    }
    pred_mean <- colMeans(pred_draws)
    roc_obj <- roc(response = model$data$target, predictor = pred_mean)
    plot(roc_obj, main = glue::glue("ROC (AUC = {round(auc(roc_obj), 3)})"))
  })
  
  output$aucValue <- renderUI({
    if (!"target" %in% names(model$data)) return(NULL)
    pred_draws <- tryCatch(posterior_epred(model, newdata = model$data), error = function(e) NULL)
    if (is.null(pred_draws)) return(HTML("<b>AUC:</b> unavailable"))
    pred_mean <- colMeans(pred_draws)
    roc_obj <- roc(response = model$data$target, predictor = pred_mean)
    HTML(glue("<b>AUC:</b> {round(auc(roc_obj), 3)}"))
  })
  
  output$calPlot <- renderPlot({
    if (!"target" %in% names(model$data)) {
      plot.new(); text(0.5, 0.5, "Training outcome 'target' not found in model$data")
      return()
    }
    pred_draws <- tryCatch(posterior_epred(model, newdata = model$data), error = function(e) NULL)
    if (is.null(pred_draws)) {
      plot.new(); text(0.5, 0.5, "Calibration unavailable (posterior predictions not computable).")
      return()
    }
    pred_mean <- colMeans(pred_draws)
    df <- tibble(y = as.numeric(model$data$target), p = pred_mean) %>%
      mutate(bin = ntile(p, 10)) %>%
      group_by(bin) %>%
      summarise(pred = mean(p), obs = mean(y), .groups = "drop")
    ggplot(df, aes(x = pred, y = obs)) +
      geom_point(size = 2) + geom_line() +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
      coord_equal(xlim = c(0,1), ylim = c(0,1)) +
      labs(title = "Calibration by Decile (training data)", x = "Mean predicted", y = "Observed proportion") +
      theme_minimal(base_size = 13)
  })
  
  output$posteriorSummary <- renderPrint({
    # print population-level coefficients from stanreg
    post <- as.matrix(model)
    betas <- grep("^b\\[", colnames(post), value = TRUE)
    if (length(betas) == 0) {
      print("No population-level coefficients found in model object (names differ).")
      return()
    }
    summary_df <- data.frame(
      parameter = betas,
      mean = apply(post[, betas, drop = FALSE], 2, mean),
      l95 = apply(post[, betas, drop = FALSE], 2, quantile, 0.025),
      u95 = apply(post[, betas, drop = FALSE], 2, quantile, 0.975)
    )
    print(summary_df)
  })
}

# ===== Run the app =====
shinyApp(ui = ui, server = server)


