import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("recycle_dataset_200.csv")

# Label encoders
le_object = LabelEncoder()
le_material = LabelEncoder()
le_recyclable = LabelEncoder()

df["Object_Code"] = le_object.fit_transform(df["Object_Name"])
df["Material_Code"] = le_material.fit_transform(df["Material"])
df["Recyclable_Code"] = le_recyclable.fit_transform(df["Recyclable"])

# Train model
X = df[["Object_Code", "Material_Code"]]
y = df["Recyclable_Code"]
model = DecisionTreeClassifier()
model.fit(X, y)

# Streamlit UI
st.title("♻️ Recyclable Finder (Auto Material Detection)")
st.write("Enter an object name, and we'll detect the material and predict recyclability.")

# User input
object_input = st.text_input("Enter Object Name (e.g., Bottle, Can, Plastic Wrapper)")

# Predict if object is known
if st.button("Check Recyclability"):
    if object_input in df["Object_Name"].values:
        # Get the corresponding material
        material_detected = df[df["Object_Name"] == object_input]["Material"].values[0]

        # Encode both
        obj_code = le_object.transform([object_input])[0]
        mat_code = le_material.transform([material_detected])[0]

        # Predict
        input_data = pd.DataFrame([[obj_code, mat_code]], columns=["Object_Code", "Material_Code"])
        pred = model.predict(input_data)
        label = le_recyclable.inverse_transform(pred)[0]

        # Output
        st.info(f"🔍 Detected Material: **{material_detected}**")
        if label == "Yes":
            st.success(f"✅ '{object_input}' made of '{material_detected}' is RECYCLABLE.")
        else:
            st.error(f"❌ '{object_input}' made of '{material_detected}' is NOT recyclable.")

    else:
        st.warning("⚠️ This object is not in the dataset. Try another example.")
