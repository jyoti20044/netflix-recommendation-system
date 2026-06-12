# Netflix Recommendation System
![Python](https://img.shields.io/badge/Python-3.x-blue)
![ML](https://img.shields.io/badge/ML-Scikit--Surprise-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

A personalized movie recommendation system built on the Netflix Prize Dataset
using collaborative filtering and matrix factorization techniques.

## Project Structure
| Notebook | Description |
|----------|-------------|
| `netflix_eda.ipynb` | Exploratory Data Analysis |
| `netflix_models.ipynb` | Model building — SVD and Item-Based CF |
| `netflix_improvements.ipynb` | SVD tuning, SVD++, explainability |
| `netflix_evaluation.ipynb` | RMSE, MAE, MAP@10 evaluation |

## Models Implemented
| Model | RMSE | Notes |
|-------|------|-------|
| Item-Based CF | 1.1636 | Baseline |
| SVD (default) | 0.9867 | Best performer |
| SVD++ | 0.9930 | Enhanced SVD with implicit feedback |

## Key Features
- 3 models compared — Item-CF, SVD, SVD++
- Mandatory metrics — RMSE and MAP@10 both implemented
- Explainable recommendations — why each movie was recommended
- Cold start handling — Bayesian average scoring for new users
- Popularity bias analysis — coverage and diversity metrics

## Tech Stack
- Python 3.x
- scikit-surprise (SVD, SVD++, KNNBasic)
- pandas, numpy, matplotlib, seaborn

## How to Run

### 1. Install dependencies
pip install scikit-surprise pandas numpy matplotlib seaborn
### 2. Download dataset
- Kaggle: https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
- Place `combined_data_1.txt` and `movie_titles.csv` in project folder

### 3. Run notebooks in order
netflix_eda.ipynb
netflix_models.ipynb
netflix_improvements.ipynb
netflix_evaluation.ipynb
###Results
SVD achieves RMSE of 0.9867
SVD MAP@10 = 0.6822
Item-CF MAP@10 = 0.5811
Data sparsity > 99%
Cold start problem addressed using Bayesian average scoring
SVD++ needs more data to outperform standard SVD

##Evaluation Setup:
- Train/Test Split: 80% train · 20% test (Surprise `train_test_split`)
- Relevance threshold: rating ≥ 3.5 → considered relevant for MAP@10
  
## Dataset
- Source: Netflix Prize Dataset (Kaggle)
- Subset: 2M ratings out of 100M total
- Users: 480,189 | Movies: 17,770
- Rating scale: 1-5 stars
