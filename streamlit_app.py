import streamlit as st
from pipelines.deployment_pipeline import prediction_service_loader
import pandas as pd
from docker_services import MongoOperations
from steps import DataTransformConfig
import numpy as np
from sklearn.preprocessing import LabelEncoder

def prepare_data(data: pd.DataFrame) -> np.ndarray:
    label_encoder = LabelEncoder()
    data["famhist_index"] = label_encoder.fit_transform(data["famhist"])
    data = data.drop(["famhist"], axis=1)
    mongo_oper = MongoOperations()
    standard_scaler = mongo_oper.read_algorithm_from_mongo("standard_scaler")
    data = standard_scaler.transform(data)
    if DataTransformConfig().fine_transforming == True:
        transformation_algorithm = mongo_oper.read_algorithm_from_mongo("transformation_algorithm")
        data = transformation_algorithm.transform(data)
    return data

def main():
    st.title("End to end Heart Disease Diagnosis Pipeline")

    st.header("Patient healt parameters:")
    sbp = st.number_input("Systolic blood pressure")
    tobacco = st.number_input("Cumulative tobacco (kg)")
    ldl = st.number_input("Low density lipoprotein cholesterol level")
    adiposity = st.number_input("Severe overweight")
    famhist = st.selectbox("Family history of heart disease", ["Present", "Absent"])
    typea = st.number_input("Type-A behavior")
    obesity = st.number_input("Excessive fat accumulation")
    alcohol = st.number_input("Current alcohol consumption")
    age = st.number_input("Age at onset")

    if st.button("Predict"):
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False
        )
        if service is None:
            st.write("Can not find service, the pipeline needs to be run first, to create a service.")
            main()
        data = pd.DataFrame({
            "sbp": [sbp],
            "tobacco": [tobacco],
            "ldl": [ldl],
            "adiposity": [adiposity],
            "famhist": [famhist],
            "typea": [typea],
            "obesity": [obesity],
            "alcohol": [alcohol],
            "age": [age]
        })
        prepared_data = prepare_data(data=data)
        prediction = service.predict(prepared_data)
        st.success("1 - Heart Disease\n"
                   "0 = Helthy Patient\n"
                   f"Your results: {prediction}")


if __name__ == "__main__":
    main()
