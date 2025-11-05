# Setup

Install dependencies:

```bash
pip3 install -r requirements.txt
```

## Running

1. Clean data (optional):

```bash
python3 cleaning_data.py
```

2. Train models:

```bash
python3 machine_learning/train_models.py
```

3. Run web app:

```bash
streamlit run web_app/app.py
```

Or if streamlit command doesn't work:

```bash
python3 -m streamlit run web_app/app.py
```
