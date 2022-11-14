import uvicorn as uvicorn

from API import app, get_model_response

model_name = "Spam Detector"
version = "1.0.0"

@app.get('/')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }


@app.get('/health')
async def service_health():
    """Return service health"""
    return {
        "ok"
    }


@app.post('/predict')
async def model_predict(input : str):
    """Predict with input"""
    response = get_model_response(input)
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
