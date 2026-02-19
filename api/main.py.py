# ============================================================================
# 11. CRIAÇÃO DA API FASTAPI
# ============================================================================

api_code = '''
# ============================================================================
# TRACK&CARE INFERENCE API
# FastAPI para predição de localização de pacientes
# ============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import Optional

app = FastAPI(title="Track&Care Inference API", version="1.0.0")

# ============================================================================
# MODELOS DE DADOS
# ============================================================================

class TelemetryInput(BaseModel):
    imei: str
    sensor_latlong: str
    sensor_room: str
    rssi: float
    timestamp: Optional[str] = None

class PredictionOutput(BaseModel):
    predicted_room: str
    confidence: float
    model_version: str
    imei_normalized: bool

# ============================================================================
# CARREGAMENTO DO MODELO
# ============================================================================

MODEL_PATH = "models/"
model = None
label_encoder = None
imei_stats = None

def load_model():
    """Carrega o modelo e preprocessadores"""
    global model, label_encoder, imei_stats
    
    try:
        model = joblib.load(os.path.join(MODEL_PATH, "best_model.pkl"))
        label_encoder = joblib.load(os.path.join(MODEL_PATH, "label_encoder.pkl"))
        imei_stats = joblib.load(os.path.join(MODEL_PATH, "imei_stats.pkl"))
        print("✅ Modelo carregado com sucesso")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        raise

# Carregar modelo na inicialização
@app.on_event("startup")
async def startup_event():
    load_model()

# ============================================================================
# FUNÇÕES DE PREPROCESSAMENTO
# ============================================================================

def normalize_rssi(imei: str, rssi: float) -> float:
    """
    Normaliza RSSI baseado nas estatísticas do IMEI
    
    Estratégia para novos IMEIs:
    - Se IMEI conhecido: usa mean/std específicos
    - Se IMEI desconhecido: usa média global (fallback)
    """
    if imei in imei_stats.index:
        mean = imei_stats.loc[imei, 'mean']
        std = imei_stats.loc[imei, 'std']
    else:
        # Fallback para IMEIs não vistos
        mean = imei_stats['mean'].mean()
        std = imei_stats['std'].mean()
    
    if std == 0 or np.isnan(std):
        std = 1
    
    normalized = (rssi - mean) / std
    
    # Clip para evitar outliers extremos
    normalized = np.clip(normalized, -5, 5)
    
    return normalized

def extract_hour(timestamp: Optional[str]) -> int:
    """Extrai hora do timestamp"""
    if timestamp:
        try:
            return pd.to_datetime(timestamp).hour
        except:
            pass
    return 12  # Default: meio-dia

def preprocess_input(input_data: TelemetryInput) -> pd.DataFrame:
    """Preprocessa input para predição"""
    
    # Normalizar RSSI
    rssi_normalized = normalize_rssi(input_data.imei, input_data.rssi)
    
    # Extrair features
    hour = extract_hour(input_data.timestamp)
    
    # RSSI Binning
    if input_data.rssi <= -90:
        rssi_binned = 0
    elif input_data.rssi <= -70:
        rssi_binned = 1
    elif input_data.rssi <= -50:
        rssi_binned = 2
    else:
        rssi_binned = 3
    
    # Sensor Room Encoding (mapeamento fixo)
    room_mapping = {
        'Enfermaria': 0,
        'Recepção': 1,
        'Corredor': 2,
        'Arredores': 3
    }
    sensor_room_encoded = room_mapping.get(input_data.sensor_room, 1)
    
    # Interação
    rssi_room_interaction = rssi_normalized * sensor_room_encoded
    
    # Criar DataFrame
    features = pd.DataFrame([{
        'rssi_normalized': rssi_normalized,
        'sensor_room_encoded': sensor_room_encoded,
        'hour': hour,
        'rssi_binned': rssi_binned,
        'rssi_room_interaction': rssi_room_interaction
    }])
    
    return features, rssi_normalized

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Track&Care Inference API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: TelemetryInput):
    """
    Prediz a sala do paciente baseado na telemetria
    
    Validações:
    - RSSI deve estar em range físico (-120 a 0)
    - IMEI deve ser fornecido
    """
    try:
        # Validação de RSSI
        if input_data.rssi > 0 or input_data.rssi < -120:
            raise HTTPException(
                status_code=400, 
                detail=f"RSSI inválido: {input_data.rssi}. Range esperado: -120 a 0 dBm"
            )
        
        # Preprocessamento
        features, rssi_normalized = preprocess_input(input_data)
        
        # Predição
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities))
        
        # Decode da predição
        predicted_room = label_encoder.inverse_transform([prediction])[0]
        
        return PredictionOutput(
            predicted_room=predicted_room,
            confidence=round(confidence, 4),
            model_version="1.0.0",
            imei_normalized=rssi_normalized != input_data.rssi
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

@app.post("/predict/batch")
async def predict_batch(inputs: list[TelemetryInput]):
    """Predição em batch para múltiplas leituras"""
    results = []
    for input_data in inputs:
        try:
            result = await predict(input_data)
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e)})
    return {"predictions": results, "count": len(results)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

# Salvar código da API
with open('api.py', 'w', encoding='utf-8') as f:
    f.write(api_code)

print("✅ API FastAPI criada: api.py")