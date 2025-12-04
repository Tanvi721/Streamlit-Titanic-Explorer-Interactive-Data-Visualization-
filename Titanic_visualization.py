import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic Dashboard", layout="wide")

# Load Titanic dataset
df = sns.load_dataset("titanic")

st.title("🚢 Titanic Dataset Visualization (10+ Graphs using Seaborn)")
st.write("Interactive dashboard created using **Seaborn** and **Streamlit**.")

# Sidebar filters
st.sidebar.header("Filters")
sex_filter = st.sidebar.multiselect("Select Gender:", df['sex'].dropna().unique(),
                                    default=df['sex'].dropna().unique())
class_filter = st.sidebar.multiselect("Select Class:", df['class'].dropna().unique(),
                                      default=df['class'].dropna().unique())

filtered_df = df[(df['sex'].isin(sex_filter)) & (df['class'].isin(class_filter))]


# Tabs
tabs = st.tabs([
    "1) Age Distribution",
    "2) Gender Count",
    "3) Class Count",
    "4) Survival Count",
    "5) Survival by Gender",
    "6) Survival by Class",
    "7) Fare Distribution",
    "8) Age vs Fare Scatter",
    "9) Pairplot (Selected Columns)",
    "10) Correlation Heatmap"
])

# -------------------------------------------------------
# 1. Age Distribution
# -------------------------------------------------------
with tabs[0]:
    st.subheader("Age Distribution (Histogram + KDE)")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df['age'].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

# -------------------------------------------------------
# 2. Gender Count
# -------------------------------------------------------
with tabs[1]:
    st.subheader("Passenger Count by Gender")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x="sex", ax=ax)
    st.pyplot(fig)

# -------------------------------------------------------
# 3. Class Count
# -------------------------------------------------------
with tabs[2]:
    st.subheader("Passenger Count by Class")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x="class", ax=ax)
    st.pyplot(fig)

# -------------------------------------------------------
# 4. Survival Count
# -------------------------------------------------------
with tabs[3]:
    st.subheader("Survival Count")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x="survived", ax=ax)
    ax.set_xticklabels(["Not Survived", "Survived"])
    st.pyplot(fig)

# -------------------------------------------------------
# 5. Survival by Gender
# -------------------------------------------------------
with tabs[4]:
    st.subheader("Survival Rate by Gender")
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_df, x="sex", y="survived", ax=ax)
    st.pyplot(fig)

# -------------------------------------------------------
# 6. Survival by Class
# -------------------------------------------------------
with tabs[5]:
    st.subheader("Survival Rate by Passenger Class")
    fig, ax = plt.subplots()
    sns.barplot(data=filtered_df, x="class", y="survived", ax=ax)
    st.pyplot(fig)

# -------------------------------------------------------
# 7. Fare Distribution
# -------------------------------------------------------
with tabs[6]:
    st.subheader("Fare Distribution")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x="class", y="fare", ax=ax)
    st.pyplot(fig)

# -------------------------------------------------------
# 8. Age vs Fare Scatterplot
# -------------------------------------------------------
with tabs[7]:
    st.subheader("Age vs Fare (Survival Highlighted)")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x="age", y="fare",
                    hue="survived", palette="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------------------------------------------
# 9. Pairplot (Optional)
# -------------------------------------------------------
with tabs[8]:
    st.subheader("Pairplot – Age, Fare, Survived")
    fig = sns.pairplot(filtered_df[['age', 'fare', 'survived']].dropna(),
                       hue="survived", diag_kind="kde")
    st.pyplot(fig)

# -------------------------------------------------------
# 10. Correlation Heatmap
# -------------------------------------------------------
with tabs[9]:
    st.subheader("Correlation Heatmap")
    num_df = filtered_df.select_dtypes(include=['int64', 'float64'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)