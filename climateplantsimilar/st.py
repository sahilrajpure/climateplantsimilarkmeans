import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv("climateplantsimilar/plants_survival_dataset_cleaned.csv")

# Select features for clustering and recommendation
features = ["Temperature (°C)", "Humidity (%)", "Sunlight Hours per Day", "Wind Speed (km/h)"]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df_scaled)

# Train Nearest Neighbors model
knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
knn.fit(df_scaled)

# Define Recommendation Function
def recommend_plants(temp, humidity, sunlight, wind_speed):
    input_data = scaler.transform([[temp, humidity, sunlight, wind_speed]])
    distances, indices = knn.kneighbors(input_data)
    return df.iloc[indices[0]], input_data

# CSS
st.markdown(
    """
    <style>
        .green-box {
    position: absolute;
    top: 5px;
    left: -60px;
    right: -60px;
    border: 6px solid green;
    box-sizing: border-box;
    padding: 10px;
    background-color: white;
    box-shadow: 0px 0px 15px rgba(0, 128, 0, 0.5);
    display: flex;
    height: calc(550vh - 10px);
    flex-direction: column;
    align-items: center;
    justify-content: center;
}

/* Adjust styles for mobile view */
@media (max-width: 680px) {
    .green-box {
        left: -30px;
        right: -30px;
        height: calc(550vh - 10px);
    }
}

        /* Center the logo and make it responsive */
        .logo-container {
            text-align: center;
            margin-bottom: 20px;
        }

        /* Aligns everything properly */
        .stApp {
            display: flex;
            justify-content: center;
        }

        .stMarkdown {
            width: 100%;
        }

        /* Styling for text display */
        .text-container {
            font-size: 18px;
            font-weight: bold;
            color: green;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Open the green box container
st.markdown('<div class="green-box">', unsafe_allow_html=True)

# Display the logo
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image("climateplantsimilar/logoheade.png", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Add a green line after the image
st.markdown("<hr style='border: 3px solid green; width: 550px; margin: 15px;'>", unsafe_allow_html=True)

# App Title and Description
st.markdown("<h2 style='text-align: center;'>🌿 Plant Suitability Recommender</h3>", unsafe_allow_html=True)
st.write("Enter your region's climate conditions to get the best plant recommendations.")

# User Inputs
temp = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
sunlight = st.slider("Sunlight Hours per Day", min_value=0.0, max_value=12.0, value=6.0)
wind_speed = st.slider("Wind Speed (km/h)", min_value=0.0, max_value=50.0, value=10.0)

# Display user input values dynamically
st.markdown("""
    <div style="display: flex; justify-content: center; text-align: center; flex-direction: column;">
        <p>🌡️ <b>Selected Temperature:</b> {temp}°C</p>
        <p>💧 <b>Selected Humidity:</b> {humidity}%</p>
        <p>☀️ <b>Selected Sunlight Hours:</b> {sunlight} hrs</p>
        <p>🌬️ <b>Selected Wind Speed:</b> {wind_speed} km/h</p>
    </div>
""".format(temp=temp, humidity=humidity, sunlight=sunlight, wind_speed=wind_speed), unsafe_allow_html=True)



# Close the green box container
st.markdown('</div>', unsafe_allow_html=True)

# Recommend Button
if st.button("Get Plant Recommendations 🌱"):
    recommended_plants, input_point = recommend_plants(temp, humidity, sunlight, wind_speed)
    
    st.success("🌿 Recommended Plants for Your Climate:")
    
    for _, plant in recommended_plants.iterrows():
        st.write(f"**- {plant['Plant Name']}**")
        image_path = f"C:/Users/Sahil/Downloads/plant_images/{plant['Plant Name'].replace(' ', '_')}.jpg"
        if os.path.exists(image_path):
            st.image(image_path, caption=plant["Plant Name"], use_container_width=True)

    # Visual 1: Cluster Plot
    st.subheader("📌 Understanding the K-Means Cluster Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df_scaled[:, 0], y=df_scaled[:, 1], hue=df["Cluster"], palette="viridis", alpha=0.6)
    plt.scatter(input_point[:, 0], input_point[:, 1], color='red', marker='X', s=200, label="User Input")
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title("K-Means Clustering with User Input Highlighted")
    plt.legend()
    st.pyplot(fig)

    # Explanation for Cluster Plot
    st.write("""
    1. Each **color represents a distinct cluster** of plants based on climate conditions.  
    2. The **X-axis (Temperature) and Y-axis (Humidity)** help differentiate plant groups.  
    3. The **User Input (red 'X')** shows where your climate fits among existing plant data.  
    4. Plants with **similar climate conditions** are grouped together.  
    5. If your input falls near a cluster center, it means many plants thrive in that range.  
    6. Clusters **overlapping** indicate plants that can survive in multiple conditions.  
    7. If the input is isolated, it may suggest fewer suitable plant options.  
    8. The **K-Means algorithm dynamically groups plants** based on climate similarities.  
    """)

    # Visual 2: Heatmap
    st.subheader("📌 Understanding the Feature Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    correlation_matrix = df[features + ["Cluster"]].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax2)
    plt.title("Feature Correlation Heatmap")
    st.pyplot(fig2)

    # Explanation for Heatmap
    st.write("""
    1. The heatmap shows **how climate factors influence plant clustering**.  
    2. **Values close to +1** mean two features are **highly correlated**.  
    3. **Values close to -1** mean two features are **negatively correlated**.  
    4. Dark red squares indicate **strong positive relationships** (e.g., Temperature & Sunlight).  
    5. Dark blue squares indicate **strong negative relationships**.  
    6. If a feature has **low correlation with Cluster**, it means it does not strongly affect plant groups.  
    7. This heatmap helps in **identifying key factors** that define plant survival in different regions.  
    8. **Understanding feature correlation can improve the recommendation accuracy.**  
    """)

    # Visual 3: Bar Chart for Top 5 Recommended Plants
    st.subheader("📌 Understanding the Top 5 Recommended Plants Bar Chart")
    filtered_features = ["Temperature (°C)", "Humidity (%)", "Sunlight Hours per Day", "Wind Speed (km/h)"]

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    recommended_plants_filtered = recommended_plants.melt(id_vars="Plant Name", value_vars=filtered_features, var_name="Feature", value_name="Value")
    sns.barplot(x="Feature", y="Value", hue="Plant Name", data=recommended_plants_filtered, palette="husl", ax=ax3)
    
    plt.xticks(rotation=45)
    plt.title("Climate Conditions of Top 5 Recommended Plants")
    plt.ylabel("Value")
    plt.xlabel("Climate Factors")
    st.pyplot(fig3)

    # Explanation for Bar Chart
    st.write("""
    1. This bar chart compares the climate conditions of the **top 5 recommended plants**,  
    2. Each **colored bar represents a plant**, while the X-axis shows different climate factors.  
    3. The **height of the bars indicates the required value** for each climate factor.  
    4. Plants with **similar values to the user input** are more suitable for survival.  
    5. This visualization helps in identifying **which plant closely matches the given conditions**.  
    6. If a plant has **higher or lower values in multiple factors**, it may need **specific adjustments** to thrive.  
    7. The bar chart provides a **quick and effective comparison of plant adaptability**.  
    8. This allows users to understand **why certain plants are recommended based on the given climate**.  
    """)


