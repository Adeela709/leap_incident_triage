import joblib
from fastapi import FastAPI


from fastapi.responses import HTMLResponse

model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ready"}

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h1>Incident Severity Triage</h1>
            <p>API is running.</p>
        </body>
    </html>
    """