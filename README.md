# Will a Kickstarter project get funded? Predicting success at launch

A team project from the Neuefishe Data Science and AI bootcamp. We asked whether a Kickstarter campaign's success can be predicted **at the moment it launches**, using only information the creator knows on day one.

> **My role:** This was a team project. I built the modelling part: model comparison, hyperparameter tuning, final evaluation, threshold analysis and feature importance (`04_Training_the_models_final.ipynb`). Data cleaning and preprocessing were done in the team (`kickstarter_eda_preprocessing.ipynb`).

## The question

Kickstarter is all-or-nothing: if a campaign misses its funding goal, the creator gets nothing. So can we tell, before a single backer arrives, which projects are likely to succeed, and what drives that?

## Data

- The public Kaggle *Kickstarter Projects* dataset (`ks-projects-201801.csv`).
- **366,690 projects** after cleaning, about **36% successful**, so the classes are imbalanced.
- Features, all known at launch: funding goal (log10, in USD), category, country, campaign duration, launch year, month, day, weekday and hour, and the length of the project name.
- Cancelled projects count as failures, because a cancelled campaign was not funded.

## Making sure the results are honest

A model like this can look much better than it really is, so most of the care went into avoiding that:

- **Data leakage removed.** Columns such as *number of backers* and *amount pledged* are only known after the campaign ends. With them in, the AUC jumps to about 0.99, which is a fake result. They were removed, along with a column that encoded the final state.
- **Related projects kept together.** Many creators relaunch the same project. Attempts at the same project share a group ID, and `GroupShuffleSplit` keeps each group in a single split, so the model is never tested on a project it has effectively already seen.
- **Clean evaluation design.** The data was split into training (68%), validation (12%) and test (20%) sets. Models were compared and tuned on validation only, and the test set was used exactly once at the end.
- **Sensible metric.** Always predicting "failure" already gives 64% accuracy, so accuracy is misleading here. The main metric is **F1** for the successful class, reported together with precision, recall, accuracy and ROC AUC.

## Models and results

Four models were compared, each with class weights balanced for the minority class:

| Model (validation set, after tuning) | F1 | Precision | Recall | Accuracy | ROC AUC |
|---|---|---|---|---|---|
| Baseline (always "failure") | 0.00 | 0.00 | 0.00 | 0.63 | – |
| Logistic regression | 0.589 | 0.528 | 0.665 | 0.660 | 0.724 |
| Random forest | 0.613 | 0.542 | 0.707 | 0.674 | 0.748 |
| **Gradient boosting** | **0.625** | **0.554** | **0.716** | **0.685** | **0.762** |

Gradient boosting was selected as the final model. **On the test set it reaches F1 = 0.62 and ROC AUC = 0.76.** The training, validation and test scores differ by only about 0.01, so the model does not overfit.

The three models perform similarly, and even the training F1 stays around 0.63. In other words, the limit is the information in the features, not the choice of algorithm.

## What drives success?

I used **permutation importance**: each feature is shuffled in turn, and the drop in test ROC AUC shows how much the model relies on it.

| Feature | Drop in AUC |
|---|---|
| Funding goal | 0.102 |
| Category | 0.071 |
| Campaign duration | 0.031 |
| Launch year | 0.025 |
| Name length | 0.015 |
| Launch hour, country, month, weekday, day | < 0.012 each |

**The size of the goal and the type of project matter far more than when you launch.** Lower goals succeed much more often, and categories such as games, comics and theatre do better than technology or food.

## Choosing the decision threshold

The model leans towards optimism (precision of about 0.55). Raising the decision threshold trades recall for precision:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 | 0.548 | 0.717 | 0.621 |
| 0.60 | 0.606 | 0.546 | 0.575 |
| 0.70 | 0.677 | 0.334 | 0.447 |

Which threshold is right depends on the use. A platform screening projects would want something different from a creator checking their own odds.

## Limitations and next steps

- The data only describes a project at launch. The real drivers, such as the pitch, video, reward tiers and the creator's audience, aren't included.
- Next steps:
  - add text features from the project name or description, after removing the "(Canceled)" suffix that would leak the outcome;
  - choose the threshold on the validation set;
  - test on later years only, so the evaluation mimics real use.

## Repository

| File | Contents |
|---|---|
| `kickstarter_eda_preprocessing.ipynb` | Data checks, leakage analysis, cleaning and preprocessing (team) |
| `04_Training_the_models_final.ipynb` | Modelling, tuning, evaluation, threshold and feature importance (my part) |
| `Outliers Will Your Kickstarter Get Funded.pptx` | Final team presentation |
| `pyproject.toml`, `uv.lock` | Python environment |

**To run it:** download `ks-projects-201801.csv` from Kaggle into `data/`, install the environment with `uv sync`, then run the preprocessing notebook followed by the modelling notebook.

*Built with Python, pandas and scikit-learn.*
