import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Productivity Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "final_model.pkl"
X_TRAIN_PATH = BASE_DIR / "data" / "x_train.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD TRAINING DATA
# ============================================================

@st.cache_data
def load_training_data():
    return pd.read_csv(X_TRAIN_PATH)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

try:

    model = load_model()
    X_train = load_training_data()

except FileNotFoundError as e:

    st.error(
        f"Required file was not found:\n\n{e}"
    )

    st.stop()

except Exception as e:

    st.error(
        f"Unable to load model or training data:\n\n{e}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("📊 Employee Productivity Predictor")

st.markdown(
    """
    Predict employee productivity using a trained
    machine learning model.
    """
)

st.divider()


# ============================================================
# GET MODEL FEATURES
# ============================================================

if hasattr(model, "feature_names_in_"):

    features = list(model.feature_names_in_)

else:

    features = list(X_train.columns)


# ============================================================
# CHECK MODEL FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in X_train.columns
]

if missing_features:

    st.error(
        "The following model features are missing "
        "from x_train.csv:"
    )

    st.write(missing_features)

    st.stop()


# ============================================================
# CATEGORICAL / ORDINAL FEATURES
# ============================================================

categorical_features = [
    "education_level",
    "stress_level",
    "project_complexity",
    "manager_rating"
]


# ============================================================
# INPUT SECTION
# ============================================================

st.header("👤 Employee Information")

st.caption(
    "Enter or select all employee information before "
    "making a prediction."
)


# ============================================================
# STORE INPUT VALUES
# ============================================================

input_values = {}


# ============================================================
# TWO COLUMN LAYOUT
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# CREATE INPUT FIELDS
# ============================================================

for i, feature in enumerate(features):

    column = col1 if i % 2 == 0 else col2

    with column:

        feature_name = (
            feature
            .replace("_", " ")
            .title()
        )


        # ====================================================
        # CATEGORICAL / ORDINAL FEATURES
        # ====================================================

        if feature in categorical_features:

            categories = (
                X_train[feature]
                .dropna()
                .unique()
                .tolist()
            )

            # Sort numeric categories correctly
            try:
                categories = sorted(categories)
            except TypeError:
                pass


            value = st.selectbox(
                feature_name,
                options=categories,
                index=None,
                placeholder=f"Select {feature_name}",
                key=f"select_{feature}"
            )


        # ====================================================
        # NUMERICAL FEATURES
        # ====================================================

        else:

            feature_data = X_train[feature].dropna()

            min_value = feature_data.min()
            max_value = feature_data.max()


            # -----------------------------------------------
            # INTEGER FEATURE
            # -----------------------------------------------

            if pd.api.types.is_integer_dtype(
                feature_data
            ):

                value = st.number_input(
                    feature_name,
                    min_value=int(min_value),
                    max_value=int(max_value),
                    value=None,
                    step=1,
                    placeholder=f"Enter {feature_name}",
                    key=f"number_{feature}"
                )


            # -----------------------------------------------
            # FLOAT FEATURE
            # -----------------------------------------------

            else:

                value = st.number_input(
                    feature_name,
                    min_value=float(min_value),
                    max_value=float(max_value),
                    value=None,
                    step=0.01,
                    placeholder=f"Enter {feature_name}",
                    key=f"number_{feature}"
                )


        input_values[feature] = value


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.write("")

predict_button = st.button(
    "🔮 Predict Productivity",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # ========================================================
    # CHECK FOR EMPTY INPUTS
    # ========================================================

    missing_values = [
        feature
        for feature, value in input_values.items()
        if value is None
    ]


    if missing_values:

        st.warning(
            "⚠️ Please enter/select all employee "
            "information before making a prediction."
        )

        missing_names = [
            feature
            .replace("_", " ")
            .title()
            for feature in missing_values
        ]

        st.write(
            "**Missing fields:**"
        )

        for name in missing_names:

            st.write(
                f"- {name}"
            )


    else:

        # ====================================================
        # CREATE INPUT DATAFRAME
        # ====================================================

        input_data = pd.DataFrame(
            [input_values],
            columns=features
        )


        # ====================================================
        # ENSURE COLUMN ORDER
        # ====================================================

        input_data = input_data[features]


        # ====================================================
        # MAKE PREDICTION
        # ====================================================

        try:

            prediction = model.predict(
                input_data
            )[0]


            # =================================================
            # PREDICTION RESULT
            # =================================================

            st.divider()

            st.header("🎯 Prediction Result")


            # =================================================
            # PRODUCTIVITY SCORE
            # =================================================

            st.metric(
                label="Predicted Productivity Score",
                value=f"{prediction:.2f}"
            )


            # =================================================
            # PRODUCTIVITY CATEGORY
            # =================================================

            if prediction >= 80:

                st.success(
                    "🟢 High Productivity\n\n"
                    "The employee is predicted to have "
                    "a high productivity level."
                )


            elif prediction >= 60:

                st.warning(
                    "🟡 Moderate Productivity\n\n"
                    "The employee is predicted to have "
                    "a moderate productivity level."
                )


            else:

                st.error(
                    "🔴 Low Productivity\n\n"
                    "The employee is predicted to have "
                    "a low productivity level."
                )


            # =================================================
            # ENTERED INFORMATION
            # =================================================

            with st.expander(
                "🔍 View Entered Employee Information"
            ):

                display_data = input_data.T.reset_index()

                display_data.columns = [
                    "Feature",
                    "Value"
                ]

                display_data["Feature"] = (
                    display_data["Feature"]
                    .str.replace("_", " ")
                    .str.title()
                )

                st.dataframe(
                    display_data,
                    use_container_width=True,
                    hide_index=True
                )


        except Exception as e:

            st.error(
                f"Prediction failed:\n\n{e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Employee Productivity Prediction | "
    "Machine Learning Project"
)