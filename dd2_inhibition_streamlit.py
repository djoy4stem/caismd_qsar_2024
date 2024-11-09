import streamlit as st
from streamlit_ketcher import st_ketcher
from io import StringIO

import pandas as pd
from lib import predict


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    if not df is None:
        return df.to_csv().encode("utf-8")


# st.image("figs/mosquito.jpeg", caption="mosquito")
# st.sidebar.image("figs/mosquito.jpeg", caption=None, use_container_width=True)

# Create two columns, with the image on the right
col1, col2 = st.columns([3, 1])  # Make the left column wider than the right

# Place main content in the left column
with col1:
    st.write("""
    ## DD2 Inhibition Prediction App
    This application helps you predict whether a given compound inhibits the growth 
    of the P. falciparum's DD2 parasite, which is responsible for the Malaria disease.
    Inhibiting molecules could be considered as possible anti-malarial drugs.
    """)

# Place the image in the upper-right column
with col2:
    st.image("assets/mosquito.jpeg", caption=None, width=150)  # Adjust width as needed


input_type_ = st.radio(
    "Select an input mode",
    ["SMILES", "CSV File", "Drawing"]

)

dd2_df = None
predictions = None
smiles_column='SMILES'

if input_type_ == "SMILES":
    smiles_input = st.text_input("Enter a valid SMILES")
    if smiles_input:
        # smile_code = st_ketcher(smiles_input)
        dd2_df = pd.DataFrame([smiles_input], columns=[smiles_column])


elif input_type_ == "CSV File":

    uploaded_file = st.file_uploader("Upload a CSV a file")
    

    if uploaded_file is not None:
        # To read file and return a pandas dataframe:
        dd2_df = pd.read_csv(uploaded_file)

        st.text('Displaying the first two rows...')
        st.write(dd2_df.iloc[:2])


elif input_type_ == "Drawing":
    smiles_input = st_ketcher(height=400)
    if smiles_input:
        dd2_df = pd.DataFrame([smiles_input], columns=[smiles_column])   

if not dd2_df is None:
    predictions = predict.featurize_and_predict_from_smiles(dd2_df, smiles_col="SMILES")
    if not predictions is None:
        predictions = predictions.astype({col: str for col in predictions.select_dtypes(bool).columns})



        # predictions_csv = convert_df(predictions)

        st.download_button(
        label="Download data as CSV",
        data=convert_df(predictions),
        file_name="dd2_predictions.csv",
        mime="text/csv",
        )

        st.text('\nBelow are the first 10 predictions...')
        st.write(predictions.iloc[:10])
    else:
        st.markdown(
                    "<p style='color: red;'>The computation failed. Make sure to enter valid SMILES of organic molecules.</p>",
                    unsafe_allow_html=True
                )

else:
    if input_type_ == "SMILES" and not st.session_state.get("smiles_input"):
        st.warning("Please enter a valid SMILES string to proceed.")
    elif input_type_ == "Upload CSV" and not uploaded_file:
        st.warning("Please upload a CSV file to proceed.")


