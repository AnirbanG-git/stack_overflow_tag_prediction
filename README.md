# Project Name
**Multiclass Multilabel Prediction for Stack Overflow Questions**

## General Information

### Overview
This project aims to predict tags for Stack Overflow questions, a vital task to efficiently categorize content for easier navigation and assistance. Utilizing text data from questions, the goal is to apply machine learning models to accurately predict associated tags.

### Dataset
The dataset comprises Stack Overflow questions and can be downloaded [here](https://www.dropbox.com/s/5721wcs2guuykzl/stacksample.zip?dl=0).

### Objective
To predict multiple tags associated with the questions posted on Stack Overflow.

### Methodology
1. Data was filtered to focus on the top 10 most occurring tags for model building.
2. The project initially explores basic text data preprocessing before diving into complex models.
3. Models developed include GRU and Bidirectional GRU, using GloVe embeddings for text representation.

## Technologies Used

This project leverages a variety of technologies, libraries, and frameworks:

- **Conda**: 23.5.2
- **Python**: 3.8.18
- **NumPy**: 1.22.3
- **Pandas**: 2.0.3
- **Matplotlib**:
  - Core: 3.7.2
  - Inline: 0.1.6
- **Seaborn**: 0.12.2
- **Scikit-learn**: 1.3.0
- **TensorFlow and Related Packages**:
  - TensorFlow Dependencies: 2.9.0
  - TensorFlow Estimator: 2.13.0 (via PyPI)
  - TensorFlow for macOS: 2.13.0 (via PyPI)
  - TensorFlow Metal: 1.0.1 (for GPU acceleration on macOS, via PyPI)
- **NLTK**: 3.8.1
- **Beautiful Soup**: 4.12.3
- **WordCloud**: 1.9.2

### Key Libraries and Their Roles:

#### Natural Language Processing (NLP) Libraries:
- **NLTK**: Utilized for text processing and analysis tasks such as tokenization, stemming, and lemmatization.
- **Beautiful Soup**: Employed for HTML parsing and cleaning, essential for processing web-sourced textual data.
- **WordCloud**: Used to generate visual word clouds from text data, aiding in the visual analysis of text features.

#### Machine Learning and Deep Learning Libraries:
- **Scikit-learn**: Provides tools for data preprocessing, model selection, and evaluation metrics, supporting a wide range of machine learning tasks.
- **TensorFlow (including Keras)**: The backbone for building and training advanced neural network models, including GRU and Bidirectional GRU architectures, to handle complex text classification challenges.

#### Data Visualization Libraries:
- **Matplotlib** and **Seaborn**: Integral for creating a wide array of data visualizations, from simple plots to complex heatmaps, to analyze model performance and explore data characteristics.


## Conclusions
In this analysis, we explored the application of GloVe embeddings in conjunction with two types of neural network models: a GRU model and a Bidirectional GRU model, aimed at classifying text data into specific categories. Our objective was to determine the effectiveness of these models on a multi-class text classification task. The performance of each model was evaluated based on Precision, Recall, and F1 Score, considering both macro-average and specific metrics per category.

**GRU Model Performance:**
The GRU model achieved a macro-average precision of approximately 0.873, recall of 0.821, and an F1 score of 0.843. This model showed particularly strong performance in categorizing 'android' and 'python' questions, with precision scores of 0.96 for both categories. However, it was less effective in handling 'html' questions, indicating potential areas for model improvement.

**Bidirectional GRU Model Performance:**
The Bidirectional GRU model, on the other hand, presented a slightly different performance profile. It achieved a macro-average precision of about 0.889, recall of 0.805, and an F1 score of 0.842. This model excelled in the 'python' category with a high precision score of 0.97 but showed limitations in adequately capturing the 'html' and 'jquery' categories, as evidenced by their lower recall rates.

**Comparative Insights:**
- **Precision and Recall Trade-off:** Both models demonstrated a trade-off between precision and recall, a common phenomenon in classification tasks. The Bidirectional GRU model exhibited higher precision overall, suggesting it is less prone to false positives, whereas the GRU model showed better recall, indicating its strength in identifying relevant instances across categories.

- **Model Complexity and Performance:** The Bidirectional GRU model, with its ability to capture information from both past and future states, did not significantly outperform the standard GRU model in terms of F1 score. This suggests that for this specific task, the additional complexity of a bidirectional architecture might not yield proportional benefits in performance, considering the computational overhead.

- **Category-Specific Performance:** Both models struggled with the 'html' category, which might indicate challenges related to the specificity of language used in HTML-related questions or the similarity of HTML questions with other categories, requiring further investigation into feature engineering or model architecture adjustments.

The analysis demonstrates the efficacy of using GloVe embeddings with GRU-based models for text classification tasks. While both models achieved commendable performance, there remains room for optimization, especially in balancing precision and recall and enhancing category-specific accuracy. Future efforts should aim to refine these models, considering the trade-offs between model complexity and performance, to achieve more robust and computationally efficient text classification systems.

## Project Files
- `stack_overflow_tag_pred.ipynb`: Jupyter notebook containing EDA, data preprocessing steps, the model training and evaluation process.

## Future Work
- Experiment with additional models and hyperparameter tuning to improve accuracy.
- Extend the model to predict beyond the top 10 tags.

## Contact
Created by [@AnirbanG-git] - feel free to contact me!
