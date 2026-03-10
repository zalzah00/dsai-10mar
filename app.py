import joblib
import pandas as pd
import gradio as gr

# Load model
artifact = joblib.load("credit_model.joblib")
model = artifact["model"]
age_col = artifact["age_col"]
credit_col = artifact["credit_col"]

def predict_credit(age, credit_amount):
    # Fixed the missing closing bracket here
    data = pd.DataFrame([[age, credit_amount]], columns=[age_col, credit_col])

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    label = "Good Credit" if pred == 1 else "Bad Credit"
    return f"{label} | Probability of Good Credit: {prob:.3f}"

demo = gr.Interface(
    fn=predict_credit,
    inputs=[gr.Number(label="Age"), gr.Number(label="Credit Amount")],
    outputs="text",
    title="German Credit Prediction",
    description="Predict creditability using Age and Credit Amount"
)

# Crucial for Render: Bind to 0.0.0.0 and port 7860
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
