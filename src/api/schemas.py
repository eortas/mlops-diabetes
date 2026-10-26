from pydantic import BaseModel, Field, ConfigDict

class DiabetesFeatures(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 0.038076,
                "sex": 0.050680,
                "bmi": 0.061696,
                "bp": 0.021872,
                "s1": -0.044223,
                "s2": -0.034821,
                "s3": -0.043401,
                "s4": -0.002592,
                "s5": 0.019907,
                "s6": -0.017646
            }
        }
    )

    age: float = Field(..., description="Edad normalizada del paciente")
    sex: float = Field(..., description="Sexo normalizado del paciente")
    bmi: float = Field(..., description="Indice de masa corporal normalizado")
    bp: float = Field(..., description="Presion arterial promedio normalizada")
    s1: float = Field(..., description="Medida de suero s1 tc")
    s2: float = Field(..., description="Medida de suero s2 ldl")
    s3: float = Field(..., description="Medida de suero s3 hdl")
    s4: float = Field(..., description="Medida de suero s4 tch")
    s5: float = Field(..., description="Medida de suero s5 ltg")
    s6: float = Field(..., description="Medida de suero s6 glu")

class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Progresion estimada de la diabetes")
    model_version: str = Field(..., description="Version del modelo utilizado")
    model_name: str = Field(..., description="Nombre del modelo en el registro")

class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    alias: str
    metrics: dict
    run_id: str
