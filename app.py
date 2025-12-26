import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ast 
import numpy as np

st.set_page_config(page_title="Movie Performance Dashboard", layout="wide")

# --- Data Loading  ---
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    
    # Standardize title column
    if "title_x" in df.columns and "title" not in df.columns:
        df.rename(columns={"title_x": "title"}, inplace=True)
    elif "title_x" in df.columns and "title" in df.columns:
        # If both exist, use 'title' and drop 'title_x'
        df.drop(columns=["title_x"], inplace=True)

    # Convert strings
    for col in ["Genres_names", "main_production_company", "languages","All_cast","production_Countries"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else ([] if pd.isna(x) else x)
            )
    
    # Ensure release month/year are correct types
    df["release_month"] = pd.to_numeric(df["release_month"], errors='coerce').astype("Int64")
    df["release_year"] = pd.to_numeric(df["release_year"], errors='coerce').astype("Int64")
    
    # Calculate profit on the original dataframe
    df["profit"] = df["revenue"] - df["budget"]

    return df

try:
    df_original = load_data("Data/movies_clean.csv")
except FileNotFoundError:
    st.error("Error: The data file 'Data/movies_clean.csv' was not found. Please make sure the file is in the correct directory.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.title("🎬 Filter Movies")

# Years filter
years = sorted(df_original["release_year"].dropna().unique())
years.insert(0, "All")

# Months filter
months = sorted(df_original["release_month"].dropna().unique())
months.insert(0, "All")

# Explode lists for filter options only
df_exploded_genres = df_original.explode("Genres_names").dropna(subset=["Genres_names"])
df_exploded_companies = df_original.explode("main_production_company").dropna(subset=["main_production_company"])
df_exploded_countries = df_original.explode("production_Countries").dropna(subset=["production_Countries"])

companies = sorted(df_exploded_companies["main_production_company"].unique()) if "main_production_company" in df_exploded_companies.columns else []
countries = sorted(df_exploded_countries["production_Countries"].unique()) if "production_Countries" in df_exploded_countries.columns else []
genres = sorted(df_exploded_genres["Genres_names"].unique()) if "Genres_names" in df_exploded_genres.columns else []


selected_year = st.sidebar.selectbox("Select Year", years)
selected_month = st.sidebar.selectbox("Select Release Month", months)
selected_genre = st.sidebar.multiselect("Select Genre(s)", genres)
selected_company = st.sidebar.multiselect("Select Company(s)", companies)
selected_country = st.sidebar.multiselect("Select Country(s)", countries)
# Preview
if df_original.empty:
    st.warning("⚠️ No movies match the current filter selections.")
else:
    st.dataframe(
        df_original[["title", "release_year", "revenue", "profit", "vote_average", "popularity","Genres_names","main_production_company","All_cast","Director"]]
        .head(5)
    )


# --- Filtering  ---
def filter_data(df):
    filtered_data = df.copy()
    
    if selected_year != "All":
        filtered_data = filtered_data[filtered_data["release_year"] == selected_year]
        
    if selected_month != "All":
        filtered_data = filtered_data[filtered_data["release_month"] == selected_month]
    
    if selected_genre:
        filtered_data = filtered_data[filtered_data["Genres_names"].apply(lambda x: any(item in selected_genre for item in x))]
        
    if selected_company:
        filtered_data = filtered_data[filtered_data["main_production_company"].apply(lambda x: any(item in selected_company for item in x))]
        
    if selected_country:
        filtered_data = filtered_data[filtered_data["production_Countries"].apply(lambda x: any(c in x for c in selected_country))]
    
    return filtered_data

filtered_df = filter_data(df_original)
# --- Main Metrics Display ---
st.title("📊 Movie Performance Dashboard")
st.markdown("---")

if not filtered_df.empty:
    # Calculate metrics
    total_movies = filtered_df.shape[0]
    avg_imdb_rating = filtered_df['vote_average'].mean()
    total_votes = filtered_df['vote_count'].sum()
    
    # Handle cases where revenue might be all NaN or empty
    if not filtered_df['revenue'].dropna().empty:
        max_rev_movie_row = filtered_df.loc[filtered_df['revenue'].idxmax()]
        max_revenue_movie = f"{max_rev_movie_row['title']} (${max_rev_movie_row['revenue']:,.0f})"
        
        min_rev_movie_row = filtered_df.loc[filtered_df['revenue'].idxmin()]
        min_revenue_movie = f"{min_rev_movie_row['title']} (${min_rev_movie_row['revenue']:,.0f})"
    else:
        max_revenue_movie = "N/A"
        min_revenue_movie = "N/A"

    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Total Movies", value=f"{total_movies:,}")
    with col2:
        st.metric(label="Average IMDb Rating", value=f"{avg_imdb_rating:.2f} ⭐")
    with col3:
        st.metric(label="Total Votes", value=f"{total_votes:,.0f}")
    with col4:
        st.metric(label="Max Revenue Movie", value=max_revenue_movie)
    with col5:
        st.metric(label="Min Revenue Movie", value=min_revenue_movie)

else:
    # Display placeholder metrics if no data
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Total Movies", value="N/A")
    with col2:
        st.metric(label="Average IMDb Rating", value="N/A")
    with col3:
        st.metric(label="Total Votes", value="N/A")
    with col4:
        st.metric(label="Max Revenue Movie", value="N/A")
    with col5:
        st.metric(label="Min Revenue Movie", value="N/A")
    st.warning("⚠️ No movies match the current filter selections.")

st.markdown("---")


st.subheader("Filtered Data Preview")
if not filtered_df.empty:
    st.dataframe(
        filtered_df[["title", "release_year", "revenue", "profit", "vote_average", "popularity", "Genres_names", "main_production_company", "All_cast", "Director"]]
        .head(5)
    )
# Tabs
tab1, tab2, tab3, tab4 , tab5= st.tabs([
    "💰 Financial & Correlations", 
    "⭐ Popularity & Ratings", 
    "⏱ Runtime", 
    "🏢 Production Companies & Languages",
    "🎭 Cast & Directors Insights"
])

# ---------------- Tab 1: Profit Insights ----------------
with tab1:
    st.title("🎥 Box Office Insights")
    st.markdown("##### Insights into budget, revenue, ratings, and more...")

    # Prepare base data for this tab (only movies with financial data)
    base = filtered_df.copy()

    if base.empty:
        st.warning("⚠️ No movies match the current filter selections for financial analysis.")
    else:
        # Top 10 movies by profit
        top10_movies_profit = base.sort_values("profit", ascending=False).head(10)

        # Top 10 genres by average profit
        base_exploded_genres = base.explode("Genres_names").dropna(subset=["Genres_names"])
        avg_profit_genre = (
            base_exploded_genres.groupby("Genres_names")["profit"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        # Runtime stats
        runtime_stats = base[base["runtime"].notna()].copy()
        runtime_stats["runtime_category"] = pd.cut(
            runtime_stats["runtime"],
            bins=[0, 100, 140, 300],
            labels=["Short", "Medium", "Long"],
            include_lowest=True
        )

        # Subplots (2x2) - Profit Insights
        fig1 = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Top 10 Movies by Profit",
                "Top 10 Genres by Average Profit",
                "Runtime vs Profit ",
                "Profit vs Popularity"
            ),
            horizontal_spacing=0.25, vertical_spacing=0.25
        )

        # Subplot 1: Movies by Profit
        if not top10_movies_profit.empty:
            fig1.add_trace(
                go.Bar(
                    x=top10_movies_profit["title"],
                    y=top10_movies_profit["profit"],
                    marker_color="teal",
                    text=top10_movies_profit["profit"].round(0),
                    textposition="auto",
                    name="Movies"
                ),
                row=1, col=1
            )

        # Subplot 2: Genres by Avg Profit
        if not avg_profit_genre.empty:
            fig1.add_trace(
                go.Bar(
                    x=avg_profit_genre.index,
                    y=avg_profit_genre.values,
                    marker_color="steelblue",
                    text=avg_profit_genre.values.round(0),
                    textposition="auto",
                    name="Genres"
                ),
                row=1, col=2
            )

        # Subplot 3: Runtime vs Profit
        if not runtime_stats.empty:
            fig1.add_trace(
                go.Box(
                    x=runtime_stats["runtime_category"],
                    y=runtime_stats["profit"],
                    marker_color="royalblue",
                    name="Runtime Category"
                ),
                row=2, col=1
            )

        # Subplot 4: Profit vs Popularity
        fig1.add_trace(
            go.Scatter(
                x=runtime_stats["popularity"],   
                y=runtime_stats["profit"],       
                mode="markers",
                marker=dict(color="darkorange", size=8, opacity=0.7),
                text=runtime_stats["title"],     
                hoverinfo="text+x+y",
                name="Profit vs Popularity"
            ),
            row=2, col=2
        )

        fig1.update_xaxes(title_text="Popularity", row=2, col=2)
        fig1.update_yaxes(title_text="Profit", row=2, col=2)


        fig1.update_layout(
            height=1200, width=1200,
            title_text="Profit Insights",
            font=dict(size=14, color="black"),
            margin=dict(t=80, b=80, l=60, r=60)
        )
        fig1.update_xaxes(tickangle=-45)

        st.plotly_chart(fig1, use_container_width=True)
        
        # ---------------- (Revenue,Budget & Correlations) ----------------
        
        financial_df = ["popularity", "budget", "revenue", "profit", "vote_count", "runtime"]
        corr_matrix = base[financial_df].corr()

        # 1) Top Months by Average Profit
        avg_profit_month = base.groupby("release_month")["profit"].mean().sort_values(ascending=False)

        # 2) Top 10 Genres combination by Average Profit 
    
        base["Genres_combo"] = base["Genres_names"].apply(lambda x: " | ".join(sorted(x)) if isinstance(x, list) else x)
        top10_genres_profit_combo = (
            base.groupby("Genres_combo")["profit"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        # 3) Average Revenue & Budget by Genre 
        avg_rev_budget_genre = (
            base_exploded_genres.groupby("Genres_names")[["revenue", "budget"]]
            .mean()
            .sort_values("revenue", ascending=False)
            .head(10)
        )

        # Subplots (2x2) - Revenue Insights
        fig2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Top Months by Average Profit",
                "Top 10 Genres combination by Average Profit",
                "Top 10 Genres by Average Revenue & Budget",
                "Financial Metrics Correlation Heatmap"
            ),
            horizontal_spacing=0.30, vertical_spacing=0.30
        )

        # Subplot 1
        if not avg_profit_month.empty:
            fig2.add_trace(go.Bar(y=avg_profit_month.index.astype(str), x=avg_profit_month.values, marker_color="seagreen", orientation="h"), row=1, col=1)
            fig2.update_yaxes(title_text="Release Month", row=1, col=1)
            fig2.update_xaxes(title_text="Average Profit ($)", row=1, col=1)
            
        # Subplot 2
        if not top10_genres_profit_combo.empty:
            fig2.add_trace(go.Bar(x=top10_genres_profit_combo.index, y=top10_genres_profit_combo.values, name="Profit", marker_color="mediumseagreen"), row=1, col=2)

        # Subplot 3
        if not avg_rev_budget_genre.empty:
            fig2.add_trace(go.Bar(x=avg_rev_budget_genre.index, y=avg_rev_budget_genre["revenue"], name="Revenue", marker_color="dodgerblue"), row=2, col=1)
            fig2.add_trace(go.Bar(x=avg_rev_budget_genre.index, y=avg_rev_budget_genre["budget"], name="Budget", marker_color="orange"), row=2, col=1)

        # Subplot 4
        if not corr_matrix.empty:
            fig2.add_trace(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="RdBu",
                zmin=-1, zmax=1,
                text=corr_matrix.round(2).values,
                texttemplate="%{text}",
                hoverongaps=False
            ), row=2, col=2)

        fig2.update_layout(
            height=1200, width=1400,
            title_text="Revenue, Profit & Budget Insights",
            font=dict(size=14, color="black"), 
            margin=dict(t=80, b=80, l=60, r=60),
            barmode="group"
        )
        fig2.update_xaxes(tickangle=-45)

        st.plotly_chart(fig2, use_container_width=True)


# ---------------- Tab 2: Popularity & Ratings Insights ---------------
with tab2:
    st.subheader("⭐ Popularity & Ratings Insights")
    
    df2 = filtered_df.copy()
    
    if df2.empty:
        st.warning("⚠️ No movies match the current filter selections.")
    else:
        # --- Popularity Insights ---
        
        # 1) Top 10 Movies by Popularity
        top10_pop_movies = df2.sort_values("popularity", ascending=False).head(10)

        # 2) Top 10 Genres by Average Popularity
        df2["Genres_combo"] = df2["Genres_names"].apply(lambda x: " | ".join(x) if isinstance(x, list) else x)
        df2_exploded_genres = df2.explode("Genres_names").dropna(subset=["Genres_names"])
        avg_pop_genre = (
            df2_exploded_genres.groupby("Genres_names")["popularity"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

    
        # 3) Popularity Trend by Release Year
        avg_pop_year = df2.groupby("release_year")["popularity"].mean()

       
        fig_pop = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Top 10 Movies by Popularity",
                "Top 10 Genres by Average Popularity",
                "Popularity Trend by Release Year"
            ),
            horizontal_spacing=0.25, vertical_spacing=0.25
        )

        if not top10_pop_movies.empty:
            fig_pop.add_trace(
                go.Bar(x=top10_pop_movies["title"], y=top10_pop_movies["popularity"], marker_color="purple"),
                row=1, col=1
            )

        if not avg_pop_genre.empty:
            fig_pop.add_trace(
                go.Bar(x=avg_pop_genre.index, y=avg_pop_genre.values, marker_color="darkcyan"),
                row=1, col=2
            )

        if not avg_pop_year.empty:
            fig_pop.add_trace(
                go.Scatter(x=avg_pop_year.index, y=avg_pop_year.values, mode="lines+markers", line=dict(color="orange")),
                row=2, col=1
            )

        fig_pop.update_layout(
            height=1200, width=1400,
            title_text="Popularity Insights",
            font=dict(size=14, color="black")
        )

        fig_pop.update_xaxes(tickangle=-30)
        fig_pop.layout["xaxis3"].update(domain=[0, 1]) 

        st.plotly_chart(fig_pop, use_container_width=True)

        # --- Vote Insights ---
        df_vote_credible = df2[df2["vote_count"] >= 100].copy()
        
        top10_vote_movies = df_vote_credible.sort_values("vote_average", ascending=False).head(10)

        avg_vote_genre_combo = (
        df2.groupby("Genres_combo")["vote_average"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )



        df_vote_pop = df2[(df2["vote_average"] > 0) & (df2["popularity"] > 0)]
        avg_vote_year = df2.groupby("release_year")["vote_average"].mean()

        fig_vote = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Top 10 Movies by Average Rating (Min 100 Votes)",
                "Top 10 Genres combinations by Average Rating",
                "Vote Average vs Popularity Scatter",
                "Vote Average Trend by Release Year"
            ),
            horizontal_spacing=0.25, vertical_spacing=0.25
        )

        if not top10_vote_movies.empty:
            fig_vote.add_trace(go.Bar(x=top10_vote_movies["title"], y=top10_vote_movies["vote_average"], marker_color="seagreen"), row=1, col=1)
        
        if not avg_vote_genre_combo.empty:
            fig_vote.add_trace(go.Bar(x=avg_vote_genre_combo.index, y=avg_vote_genre_combo.values, marker_color="darkorange"), row=1, col=2)
        
        if not df_vote_pop.empty:
            fig_vote.add_trace(go.Scatter(x=df_vote_pop["vote_average"], y=df_vote_pop["popularity"], mode="markers", marker=dict(color="purple", size=6), hovertext=df_vote_pop["title"], hoverinfo="text+x+y"), row=2, col=1)
            fig_vote.update_xaxes(title_text="Vote Average", row=2, col=1)
            fig_vote.update_yaxes(title_text="Popularity", row=2, col=1)
        
        if not avg_vote_year.empty:
            fig_vote.add_trace(go.Scatter(x=avg_vote_year.index, y=avg_vote_year.values, mode="lines+markers", line=dict(color="blue")), row=2, col=2)

        fig_vote.update_layout(height=1200, width=1400, template="plotly_dark", title_text="Vote Insights", font=dict(size=14, color="black"))
        fig_vote.update_xaxes(tickangle=-30)

        st.plotly_chart(fig_vote, use_container_width=True)


with tab3:
    st.subheader("⏱ Runtime Insights")
    
    df_runtime = filtered_df[filtered_df["runtime"] > 0].copy()

    if df_runtime.empty:
        st.warning("⚠️ No movies with valid runtime data match the current filter selections.")
    else:
        df_runtime_votes = df_runtime[(df_runtime["runtime"] > 0) & (df_runtime["vote_count"] > 0)]

        runtime_stats = df_runtime_votes.copy()
        runtime_stats["Runtime_Category"] = pd.cut(
            runtime_stats["runtime"],
            bins=[0, 90, 120, 150, 180, float("inf")],
            labels=["Short (<90)", "Medium (90-120)", "Long (120-150)",
                    "Very Long (150-180)", "Epic (>180)"]
        )

        runtime_counts = runtime_stats["Runtime_Category"].value_counts().sort_index()
        avg_runtime_year = runtime_stats.groupby("release_year")["runtime"].mean()

        df_runtime_exploded_genres = df_runtime.explode("Genres_names").dropna(subset=["Genres_names"])

        avg_runtime_genre = (
            df_runtime_exploded_genres.groupby("Genres_names")["runtime"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Top 10 Genres by Avg Runtime",
                "Movies Count by Runtime Category",
                "Runtime vs Votes"
            ),specs=[[{}, {}], [{"colspan": 2}, None]],
            vertical_spacing=0.25,
            horizontal_spacing=0.25
        )

        # Subplot 1: Runtime vs Genre
        if not avg_runtime_genre.empty:
            fig.add_trace(go.Bar(
                x=avg_runtime_genre.index,
                y=avg_runtime_genre.values,
                marker_color="royalblue",
                text=avg_runtime_genre.values.round(1),
                textposition="auto",
                name="Avg Runtime"
            ), row=1, col=1)

        # Subplot 2: Movies Count by Runtime Category
        if not runtime_counts.empty:
            fig.add_trace(go.Bar(
                x=runtime_counts.index,
                y=runtime_counts.values,
                marker_color="darkorange",
                text=runtime_counts.values,
                textposition="auto",
                name="Count"
            ), row=1, col=2)

        # Subplot 3: Runtime vs Vote Count
        if not df_runtime_votes.empty:
          fig.add_trace(go.Scatter(
           x=df_runtime_votes["runtime"],
           y=df_runtime_votes["vote_count"],
           mode="markers",
           marker=dict(color="gold", size=6),
           hovertext=df_runtime_votes["title"],
           hoverinfo="text+x+y",
           name="Movies"
            ), row=2, col=1)


        fig.update_layout(
            height=1200, width=1400,
            title_text="Runtime Insights",
            font=dict(size=14, color="black"),
            margin=dict(t=80, b=80, l=60, r=60),
        )
        fig.update_xaxes(tickangle=-45)


        if not avg_runtime_year.empty:
          fig_runtime_trend = go.Figure()
          fig_runtime_trend.add_trace(go.Scatter(
        x=avg_runtime_year.index,
        y=avg_runtime_year.values,
        mode="lines+markers",
        line=dict(color="purple"),
        marker=dict(size=8),
        name="Trend"
    ))

          fig_runtime_trend.update_layout(
        title="Average Runtime Trend by Year",
        xaxis_title="Release Year",
        yaxis_title="Average Runtime (minutes)",
        height=600, width=1000,
        font=dict(size=14, color="black"),
        margin=dict(t=60, b=60, l=60, r=60)
    )

   


    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig_runtime_trend, use_container_width=True)

    # ----------------Production Companies & Languages ----------------
with tab4:
    st.subheader("🏢 Production Companies & 🌍 Languages Insights")

    if filtered_df.empty:
        st.warning("⚠️ No movies match the current filter selections.")
    else:
        df_filtered = filtered_df.copy()

        # --- Production Companies ---
        

        df_companies_exploded = df_filtered.explode("main_production_company").dropna(subset=["main_production_company"])

        if not df_companies_exploded.empty:
            company_stats = df_companies_exploded.groupby("main_production_company").agg({
                "vote_average": ["mean", "count", "max"],
                "profit": "mean",
                "revenue": "mean",
                "budget": "mean"
            }).round(2)

          
            company_stats.reset_index(inplace=True)
            company_stats.columns = [
                "Company", "Avg_Rating", "Movie_Count", "Max_Rating",
                "Avg_Profit", "Avg_Revenue", "Avg_Budget"
            ]

         
            company_stats["Avg_Profit/Movie_Count"] = company_stats["Avg_Profit"] / company_stats["Movie_Count"]

           
            company_stats = company_stats[company_stats["Movie_Count"] >= 2].sort_values(
                "Avg_Profit/Movie_Count", ascending=False
            )

           
            top_companies = company_stats.head(10)

           
            fig = px.bar(
                top_companies,
                x="Company",
                y="Avg_Profit/Movie_Count",
                color="Avg_Rating",
                hover_data=["Movie_Count", "Max_Rating", "Avg_Revenue", "Avg_Budget"],
                title="Production Company Performance: Avg Profit per Movie",
                labels={"Avg_Profit/Movie_Count": "Avg Profit per Movie", "Avg_Rating": "Average Rating"},
                color_continuous_scale="Viridis"
            )

            fig.update_layout(
                template="plotly_dark",
                height=700,
                width=1200,
                margin=dict(t=100, b=80, l=60, r=60),
                font=dict(size=14, color="black")
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("⚠️ No production company data available after filtering.")

        # --- Languages Insights ---
        df_lang_exploded = df_filtered.explode("languages").dropna(subset=["languages"])

        if not df_lang_exploded.empty:
            
            top_languages_count = df_lang_exploded["languages"].value_counts().head(5)
            avg_revenue_language = df_lang_exploded.groupby("languages")["revenue"].mean().sort_values(ascending=False).head(10)
            avg_rating_language = df_lang_exploded.groupby("languages")["vote_average"].mean().sort_values(ascending=False).head(10)
            avg_pop_language = df_lang_exploded.groupby("languages")["popularity"].mean().sort_values(ascending=False).head(10)

            fig_languages = make_subplots(
                rows=2, cols=2,specs=[[{"type": "domain"}, {"type": "xy"}],[{"type": "xy"}, {"type": "xy"}]],

                subplot_titles=(
                    "Top 10 Languages by Number of Movies",
                    "Top 10 Languages by Avg Revenue",
                    "Top 10 Languages by Avg Rating",
                    "Top 10 Languages by Average Popularity"
                ),
                vertical_spacing=0.18,
                horizontal_spacing=0.12
            )

            if not top_languages_count.empty:
              fig_languages.add_trace(go.Pie(labels=top_languages_count.index,values=top_languages_count.values,hole=0.4,marker=dict(colors=["seagreen"]),textinfo="label+value+percent",name="Movies"),row=1, col=1)
            
            if not avg_revenue_language.empty:
                fig_languages.add_trace(go.Bar(x=avg_revenue_language.index, y=avg_revenue_language.values, marker_color="dodgerblue", text=avg_revenue_language.values.round(0), textposition="auto", name="Revenue"), row=1, col=2)
            
            if not avg_rating_language.empty:
                fig_languages.add_trace(go.Scatter(x=avg_rating_language.index, y=avg_rating_language.values, mode="markers+lines", marker=dict(size=10, color="orchid"), name="Rating"), row=2, col=1)
            
            if not avg_pop_language.empty:
                fig_languages.add_trace(go.Bar(x=avg_pop_language.index,y=avg_pop_language.values,marker_color="mediumorchid",text=avg_pop_language.values.round(0),textposition="auto",name="Popularity"),row=2, col=2)

            fig_languages.update_layout(
                height=1100, width=1300,
                title_text="Languages Insights",
                margin=dict(t=100, b=80, l=60, r=60),
                font=dict(size=14, color="black") 
            )

            st.plotly_chart(fig_languages, use_container_width=True)

# ---------------- Concentration of Production insights ----------------

            df_exploded_countries = df_original.explode("production_Countries").dropna(subset=["production_Countries"])
            country_counts = (
                df_exploded_countries["production_Countries"]
                .value_counts()
                .reset_index()
            )

            country_counts.columns = ["Country", "Movie_Count"]

            top10_countries = country_counts.head(10)

            fig_countries = px.bar(
                top10_countries,
                x="Country",
                y="Movie_Count",
                title="Top 10 Production Countries by Number of Movies",
                labels={"Country": "Production Country", "Movie_Count": "Number of Movies"},
                color="Movie_Count",
                color_continuous_scale="Viridis"
            )

            fig_countries.update_layout(
                height=700, width=1000,
                font=dict(size=14, color="black"),
                margin=dict(t=60, b=60, l=60, r=60)
            )

            st.plotly_chart(fig_countries, use_container_width=True)
        else:
            
            st.warning("⚠️ No language data available after filtering.")

# ----------------Directors & Cast Insights ----------------
with tab5:
    st.subheader("🎭 Cast & Actors Insights")
    
    df5 = filtered_df.copy()
    
    if df5.empty:
        st.warning("⚠️ No movies match the current filter selections.")
    else:
        # 1) Cast Size
        df5["cast_size"] = df5["All_cast"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        df5["cast_category"] = pd.cut(df5["cast_size"], bins=[0,3,7,50], labels=["Small","Medium","Large"])
        
        # 2) Top 10 Actors by Profit
        df5_exploded_cast = df5.explode("All_cast").dropna(subset=["All_cast"])
        top10_actors_profit = (
        df5_exploded_cast.groupby("All_cast")["profit"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        )
        # 3) Top 10 Actors by Average Vote Rating
        top10_actors_rating = (
            df5_exploded_cast.groupby("All_cast")["vote_average"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )
        
        # 4) Top 10 Actors by Genre Preference
        df5_exploded_genres = df5.explode("All_cast").explode("Genres_names").dropna(subset=["All_cast","Genres_names"])
        actor_genre_counts = (
            df5_exploded_genres.groupby(["All_cast","Genres_names"])
            .size()
            .reset_index(name="Count")
        )
      
        actor_genre_pref = actor_genre_counts.loc[actor_genre_counts.groupby("All_cast")["Count"].idxmax()]
        top10_actor_genre_pref = actor_genre_pref.sort_values("Count", ascending=False).head(10)
        
        # Subplots 2 x 2
        fig_cast = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Cast Size vs Profit",
                "Top 10 Actors by Profit",
                "Top 10 Actors by Average Rating",
                "Top 10 Actors Genre Preference"
            ),
            horizontal_spacing=0.25, vertical_spacing=0.25
        )
        
        # Subplot 1
        fig_cast.add_trace(
            go.Box(x=df5["cast_category"], y=df5["profit"], marker_color="royalblue"),
            row=1, col=1
        )
        
        # Subplot 2
        
        fig_cast.add_trace(
         go.Bar(
        x=top10_actors_profit.index,
        y=top10_actors_profit.values,
        marker_color="seagreen",
        name="Top Actors by Profit"),row=1, col=2)

        fig_cast.update_xaxes(title_text="Actor", row=1, col=2)
        fig_cast.update_yaxes(title_text="Average Profit", row=1, col=2)

        
        # Subplot 3
        if not top10_actors_rating.empty:
            fig_cast.add_trace(
                go.Bar(
                    y=top10_actors_rating.index,
                    x=top10_actors_rating.values,
                    marker_color="violet",
                    textposition="auto",
                    name="Top Actors by Rating",orientation="h"
                ),
                row=2, col=1
            )
        
        # Subplot 4 (Top 10 Actors Genre Preference)
        if not top10_actor_genre_pref.empty:
            fig_cast.add_trace(
                go.Bar(
                    x=top10_actor_genre_pref["All_cast"],
                    y=top10_actor_genre_pref["Count"],
                    marker_color="darkorange",
                    text=top10_actor_genre_pref["Genres_names"],
                    textposition="auto",
                    name="Top Actors Genre Preference"
                ),
                row=2, col=2
            )
        
        fig_cast.update_layout(
            height=1000, width=1100,
            title_text="Cast & Actors Insights",
            template="plotly_dark",
            font=dict(size=12, color="black")
        )
        
        st.plotly_chart(fig_cast, use_container_width=True)

        # --- Directors Insights (Radar Chart) ---
        st.subheader("🎬 Top Directors Performance")
        
        df_director = df5.copy()
        df_director_grouped = df_director.groupby("Director")[["revenue","profit","vote_average"]].mean()
        
        # Normalize 0-1
        df_director_grouped["Revenue_Score"] = df_director_grouped["revenue"] / df_director_grouped["revenue"].max()
        df_director_grouped["Profit_Score"] = df_director_grouped["profit"] / df_director_grouped["profit"].max()
        df_director_grouped["Rating_Score"] = df_director_grouped["vote_average"] / df_director_grouped["vote_average"].max()
        
        # Composite Index
        df_director_grouped["Performance_Index"] = (
            df_director_grouped["Revenue_Score"]*0.4 +
            df_director_grouped["Profit_Score"]*0.3 +
            df_director_grouped["Rating_Score"]*0.3
        )
        
        top_directors = df_director_grouped.nlargest(5, "Performance_Index").reset_index()
        
        fig_directors = go.Figure()
        categories = ['Revenue', 'Profit', 'Rating']
        
        for idx, row in top_directors.iterrows():
            values = [row['Revenue_Score'], row['Profit_Score'], row['Rating_Score']]
            fig_directors.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=row['Director'],
                line=dict(width=2)
            ))
        
        fig_directors.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            showlegend=True,
            title="Multi-Dimensional Performance: Top 5 Directors",
            template="plotly_dark",
            height=700
        )
        
        st.plotly_chart(fig_directors, use_container_width=True)