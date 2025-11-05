import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Spam Detector", layout="wide")


@st.cache_resource
def load_models():
    with open('machine_learning/models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    models = {}
    model_files = {
        'Logistic Regression': 'logistic_regression_model.pkl',
        'Naive Bayes': 'naive_bayes_model.pkl',
        'Random Forest': 'random_forest_model.pkl',
        'SVM': 'svm_model.pkl'
    }

    for name, filename in model_files.items():
        with open(f'machine_learning/models/{filename}', 'rb') as f:
            models[name] = pickle.load(f)

    with open('machine_learning/models/model_results.pkl', 'rb') as f:
        results = pickle.load(f)

    return vectorizer, models, results


try:
    vectorizer, models, results = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    error_msg = str(e)

st.title("Spam Message Detector")

if not models_loaded:
    st.error(
        "Models not found. Please train the models first by running train_models.py")
    st.stop()

st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose model:", list(models.keys()))

tab1, tab2 = st.tabs(["Check Message", "Model Stats"])

with tab1:
    st.subheader("Enter a message to check")

    test_messages = {
        "Test 1 - Prize Winner": "WINNER!! You have won a £1000 cash prize! Call 09061701461 to claim your prize now!",
        "Test 2 - Free Offer": "Congratulations! You've been selected for a FREE iPhone 15. Click here: bit.ly/free-phone",
        "Test 3 - Urgent Account": "URGENT: Your bank account has been suspended. Verify your details immediately at secure-bank-login.com",
        "Test 4 - Normal Message": "Hey, are we still meeting for lunch tomorrow at 1pm? Let me know if you need to reschedule.",
        "Test 5 - Friend Message": "Thanks for helping me move last weekend! I really appreciate it. Let's grab coffee soon."
    }

    st.write("**Quick Test Messages:**")
    cols = st.columns(5)
    for idx, (label, message) in enumerate(test_messages.items()):
        with cols[idx]:
            if st.button(label, key=f"test_{idx}", use_container_width=True):
                st.session_state.message_area = message

    user_input = st.text_area("Message:", height=150,
                              placeholder="Type your message here...",
                              key="message_area")

    if st.button("Check"):
        if user_input.strip():
            input_vec = vectorizer.transform([user_input])
            model = models[selected_model]
            prediction = model.predict(input_vec)[0]
            proba = model.predict_proba(input_vec)[0]

            if prediction == 1:
                st.error("SPAM")
            else:
                st.success("NOT SPAM")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Spam probability", f"{proba[1]:.2%}")
            with col2:
                st.metric("Ham probability", f"{proba[0]:.2%}")

            import matplotlib.pyplot as plt

            st.subheader("Prediction Confidence")
            fig1, ax1 = plt.subplots(figsize=(8, 3))

            categories = ['Ham (Not Spam)', 'Spam']
            probabilities = [proba[0], proba[1]]
            colors = ['#2ecc71', '#e74c3c']

            bars = ax1.barh(categories, probabilities, color=colors)
            ax1.set_xlabel('Probability')
            ax1.set_xlim([0, 1])
            ax1.set_title(f'{selected_model} Confidence Scores')

            for bar, prob in zip(bars, probabilities):
                width = bar.get_width()
                ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                         f'{prob:.2%}', va='center', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()

            with st.expander("All model predictions"):
                predictions = []
                all_spam_probs = []
                all_ham_probs = []
                model_names = []

                for name, m in models.items():
                    pred = m.predict(input_vec)[0]
                    prob = m.predict_proba(input_vec)[0]
                    predictions.append({
                        'Model': name,
                        'Result': 'Spam' if pred == 1 else 'Ham',
                        'Confidence': f"{max(prob):.2%}"
                    })
                    model_names.append(name)
                    all_ham_probs.append(prob[0])
                    all_spam_probs.append(prob[1])

                st.table(pd.DataFrame(predictions))

                st.subheader("All Models Comparison")
                fig2, ax2 = plt.subplots(figsize=(10, 5))

                x = range(len(model_names))
                width = 0.35

                bars1 = ax2.bar([i - width/2 for i in x], all_ham_probs, width,
                                label='Ham Probability', color='#3498db')
                bars2 = ax2.bar([i + width/2 for i in x], all_spam_probs, width,
                                label='Spam Probability', color='#e67e22')

                ax2.set_ylabel('Probability')
                ax2.set_title('Prediction Probabilities Across All Models')
                ax2.set_xticks(x)
                ax2.set_xticklabels(model_names, rotation=15, ha='right')
                ax2.legend()
                ax2.set_ylim([0, 1.1])

                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        if height > 0.05:
                            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                     f'{height:.2f}', ha='center', va='bottom', fontsize=8)

                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
        else:
            st.warning("Please enter a message")

with tab2:
    st.subheader("Model Performance")

    perf_data = []
    for name, metrics in results.items():
        perf_data.append({
            'Model': name,
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1': f"{metrics['f1_score']:.3f}"
        })

    df_perf = pd.DataFrame(perf_data)
    st.dataframe(df_perf, hide_index=True)

    best = df_perf.loc[df_perf['Accuracy'].astype(float).idxmax()]
    st.info(f"Best model: {best['Model']} (Accuracy: {best['Accuracy']})")

    st.divider()

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 5))

    model_names = list(results.keys())
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    x = range(len(model_names))
    bar_width = 0.2

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, metric in enumerate(metrics_to_plot):
        values = [results[m][metric] for m in model_names]
        offset = (i - 1.5) * bar_width
        ax.bar([p + offset for p in x], values, bar_width,
               label=metric.replace('_', ' ').title(), color=colors[i])

    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim([0.5, 1.05])
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accuracy Rankings")

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        acc_vals = [results[m]['accuracy'] for m in model_names]
        colors_gradient = plt.cm.Blues([0.4, 0.5, 0.6, 0.7])

        bars = ax2.barh(model_names, acc_vals, color=colors_gradient)
        ax2.set_xlabel('Accuracy')
        ax2.set_xlim([0.9, 1.0])

        for bar in bars:
            width = bar.get_width()
            ax2.text(width - 0.005, bar.get_y() + bar.get_height()/2,
                     f'{width:.3f}', ha='right', va='center', color='white', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col2:
        st.subheader("F1-Score Distribution")

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        f1_vals = [results[m]['f1_score'] for m in model_names]
        colors_pie = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

        wedges, texts, autotexts = ax3.pie(f1_vals, labels=model_names, autopct='%1.1f%%',
                                           colors=colors_pie, startangle=90)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax3.set_title('F1-Score Proportion')
        st.pyplot(fig3)
        plt.close()

    st.divider()
    st.subheader("Training Results")

    try:
        from PIL import Image
        comparison_img = Image.open('report/model_comparison.png')
        st.image(comparison_img, caption='Model Comparison (from training)',
                 use_container_width=True)
    except:
        st.warning(
            "Training comparison chart not found. Run train_models.py first.")

    st.subheader("Confusion Matrices")
    try:
        cm_img = Image.open('report/confusion_matrices.png')
        st.image(cm_img, caption='Confusion Matrices for All Models',
                 use_container_width=True)
    except:
        st.warning("Confusion matrix chart not found.")
