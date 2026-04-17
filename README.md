# Smart Portfolio Builder

A Streamlit app that builds a stock portfolio automatically from a pool of S&P 500 stocks using:
- machine learning return forecasts
- Black-Litterman portfolio optimization
- a user-selected risk profile
- a user budget in USD

## Repo structure

```text
portfolio_app/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── features.parquet
└── src/
    ├── black_litterman.py
    ├── data_loader.py
    └── ml_views.py
```

## What the app does

1. Loads the historical stock feature dataset from `data/features.parquet`
2. Trains Random Forest and XGBoost models to generate return views and confidence scores
3. Builds a Black-Litterman posterior return estimate
4. Optimizes a long-only portfolio subject to the selected risk profile
5. Converts the weights into a dollar allocation and whole-share recommendation
6. Displays user-friendly charts and downloadable output

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud deployment

1. Push this folder to a GitHub repository.
2. In Streamlit Cloud, create a new app and connect the repo.
3. Set the main file path to:

```text
app.py
```

4. Deploy.

## Notes

- The current dataset contains 22 stocks from the available universe in `features.parquet`.
- If you later expand the parquet to 30 stocks, the app will pick them up automatically.
- The app is fully self-contained and does not depend on live APIs.
