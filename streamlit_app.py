import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


st.set_page_config(layout="wide")
st.title("Dashboard interactivo de rendimiento de equipos")

# --- Cargar datos ---
@st.cache_data
def load_data():
    df = pd.read_excel("Team_Stats.xlsx")
    df['Equipo'] = df['Equipo'].astype(str).str.strip()
    df['Competición'] = df['Competición'].astype(str).str.strip()
    return df

df = load_data()

# --- Métricas para radar ---
metricas = [
    "Entradas a ras de suelo logradas",
    "Despejes",
    "Interceptaciones",
    "Duelos defensivos ganados",
    "Duelos aereos ganados"
]
metricas = [m for m in metricas if m in df.columns]

# --- Pestañas principales ---
tab1, tab2, tab3, tab4 = st.tabs(["Radar comparativo defensivo", "Gráficos generales de equipos", "Percentiles", "Graficos jugadores"])

# ====== TAB 1: RADAR ======
with tab1:
    st.header("Radar de percentiles defensivos (por competición y equipo)")
    # --- Selección de competición ---
    competiciones = df['Competición'].value_counts().index.tolist()
    competiciones_validas = [c for c in competiciones if df[df['Competición'] == c]['Equipo'].nunique() > 2]
    competicion = st.selectbox("Competición", competiciones_validas)
    # --- Selección de equipos ---
    equipos = sorted(df[df['Competición'] == competicion]['Equipo'].unique())
    equipo_1 = st.selectbox("Equipo 1", equipos, index=0)
    equipo_2 = st.selectbox("Equipo 2", equipos, index=1 if len(equipos) > 1 else 0)
    if equipo_1 == equipo_2:
        st.warning("Selecciona dos equipos diferentes para comparar.")
    else:
        df_comp = df[df['Competición'] == competicion]
        df_equipo = df_comp.groupby("Equipo")[metricas].mean()
        percentiles = df_equipo.rank(pct=True) * 100
        valores1 = percentiles.loc[equipo_1].values
        valores2 = percentiles.loc[equipo_2].values
        N = len(metricas)
        valores1 = np.concatenate((valores1, [valores1[0]]))
        valores2 = np.concatenate((valores2, [valores2[0]]))
        angulos = np.linspace(0, 2 * np.pi, N + 1)
        fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
        colores = ['royalblue', 'crimson']
        ax.fill(angulos, valores1, color=colores[0], alpha=0.6, label=equipo_1)
        ax.plot(angulos, valores1, color=colores[0], linewidth=2)
        ax.fill(angulos, valores2, color=colores[1], alpha=0.5, label=equipo_2)
        ax.plot(angulos, valores2, color=colores[1], linewidth=2)
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(metricas, fontsize=12)
        ax.set_yticklabels([])
        for j, (angulo, v1, v2) in enumerate(zip(angulos, valores1, valores2)):
            if j < N:
                ax.text(angulo-0.08, v1 + 3, f"{int(v1)}", color=colores[0], ha='right', va='center', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, edgecolor=colores[0]))
                ax.text(angulo+0.08, v2 + 3, f"{int(v2)}", color=colores[1], ha='left', va='center', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6, edgecolor=colores[1]))
        ax.set_title(f"Comparativa Percentil defensivo\n({competicion})", size=15, y=1.13)
        ax.legend(loc='upper right', bbox_to_anchor=(1.22, 1.1))
        plt.tight_layout()
        st.pyplot(fig)
    st.caption("Radar de percentiles defensivos, calculados respecto a todos los equipos de la competición seleccionada.")

# ====== TAB 2: GRÁFICOS GENERALES ======
with tab2:
    st.header("Gráficos de rendimiento de equipos")
    # --- 1. Goles a favor por equipo ---
    st.subheader("1. Promedio de Goles a Favor por Equipo")
    goles_equipo = df.groupby('Equipo')['Goles a favor'].mean().sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(10,4))
    sns.barplot(x=goles_equipo.index, y=goles_equipo.values, ax=ax1)
    ax1.set_title('Promedio de Goles a Favor por Equipo')
    ax1.set_xlabel('Equipo')
    ax1.set_ylabel('Goles a Favor (Promedio)')
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    # --- 2. Goles a favor vs. Goles recibidos ---
    st.subheader("2. Goles a Favor vs. Goles Recibidos (por equipo, promedio)")
    goles = df.groupby('Equipo')[['Goles a favor', 'Goles recibidos']].mean()
    fig2, ax2 = plt.subplots(figsize=(5,5))
    sns.scatterplot(x=goles['Goles a favor'], y=goles['Goles recibidos'], hue=goles.index, s=100, ax=ax2)
    ax2.set_title('Goles a Favor vs. Goles Recibidos por Equipo')
    ax2.set_xlabel('Goles a Favor (Promedio)')
    ax2.set_ylabel('Goles Recibidos (Promedio)')
    ax2.legend(title='Equipo', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig2)

    # --- 3. Posesión media por equipo ---
    st.subheader("3. Posesión Media por Equipo")
    posesion = df.groupby('Equipo')['% Posesión'].mean().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(10,4))
    sns.barplot(x=posesion.index, y=posesion.values, ax=ax3)
    ax3.set_title('Posesión Media por Equipo')
    ax3.set_xlabel('Equipo')
    ax3.set_ylabel('% Posesión (Promedio)')
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

    # --- 4. Tiros a portería vs. Goles a favor ---
    st.subheader("4. Tiros a Portería vs. Goles a Favor (efectividad ofensiva)")
    ofensiva = df.groupby('Equipo')[['Tiros a porteria', 'Goles a favor']].sum()
    fig4, ax4 = plt.subplots(figsize=(6,4))
    sns.scatterplot(data=ofensiva, x='Tiros a porteria', y='Goles a favor', hue=ofensiva.index, s=100, ax=ax4)
    ax4.set_title('Tiros a Portería vs Goles a Favor por Equipo')
    ax4.set_xlabel('Tiros a Portería (Total)')
    ax4.set_ylabel('Goles a Favor (Total)')
    ax4.legend(title='Equipo', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig4)

    # --- 5. Ranking por xG promedio ---
    st.subheader("5. xG Promedio por Equipo")
    if "xG" in df.columns:
        xg = df.groupby('Equipo')['xG'].mean().sort_values(ascending=False)
        fig5, ax5 = plt.subplots(figsize=(10,4))
        sns.barplot(x=xg.index, y=xg.values, ax=ax5)
        ax5.set_title('xG Promedio por Equipo')
        ax5.set_xlabel('Equipo')
        ax5.set_ylabel('xG (Promedio)')
        ax5.tick_params(axis='x', rotation=45)
        st.pyplot(fig5)
    else:
        st.info("No hay columna 'xG' en el dataset.")


# ====== TAB 3: PERCENTILES ======
with tab3:
    st.header("Gráfico de percentiles por jugador")
    
    # --- Carga de datos ---
    df1 = pd.read_excel('Goalkeeper_Stats_WyScout_2.xlsx')
    df2 = pd.read_excel('WyScout Search results.xlsx')

    df1["Dif CG-xCG"] = df1["xCG"] - df1["Conceded Goals"]
    df2["Clean sheets ratio"] = df2["Clean sheets"] / df2["Matches played"]

    min_matches = st.slider("Mínimo de partidos jugados", min_value=1, max_value=20, value=5, key="slider_min_matches")
    nombres_validos = df1["Nombre"].value_counts()
    nombres_validos = nombres_validos[nombres_validos >= min_matches].index
    df_filtrado = df1[df1["Nombre"].isin(nombres_validos)]

    df_avg = df_filtrado.groupby("Nombre").mean(numeric_only=True).reset_index()
    df1 = df_avg.drop(['Level'], axis=1, errors='ignore')

    df_merged = pd.merge(
        df1, df2, left_on='Nombre', right_on='Player', how='inner', suffixes=('', '_df2')
    )
    cols_to_drop = [col for col in df_merged.columns if col.endswith('_df2')]
    df_merged = df_merged.drop(columns=cols_to_drop + ['Player'], errors='ignore')
    df_merged = df_merged.rename(columns={'Nombre': 'Player'})
    df_percentiles = df_merged
    df_percentiles = df_percentiles[~df_percentiles["Player"].isin(["B. Kamara", "L. Carević", "K. Pirić"])]

    # --- KPIs DERIVADAS ---
    def safe_div(df, a, b):
        if a not in df or b not in df:
            return np.nan
        return np.where((df[b]!=0) & (~df[b].isna()), (df[a]*100)/df[b], np.nan)

    df_percentiles["Total Actions Successful %"] = safe_div(df_percentiles, "Total Actions Successful", "Total Actions")
    df_percentiles["Passes Accurate %"] = safe_div(df_percentiles, "Passes Accurate", "Passes")
    df_percentiles["Long Passes Accurate %"] = safe_div(df_percentiles, "Long Passes Accurate", "Long Passes")
    df_percentiles["Aerial Duels Won %"] = safe_div(df_percentiles, "Aerial Duels Won", "Aerial Duels")
    df_percentiles["Save Rate %"] = safe_div(df_percentiles, "Saves", "Shots Against")
    df_percentiles["Saves With Reflexes %"] = safe_div(df_percentiles, "Saves With Reflexes", "Saves")
    df_percentiles["Clean Sheets Ratio"] = safe_div(df_percentiles, "Clean sheets", "Matches played")
    if "xG Against" in df_percentiles.columns and "Conceded Goals" in df_percentiles.columns:
        df_percentiles["Prevented Goals"] = df_percentiles["xG Against"] - df_percentiles["Conceded Goals"]

    # --- Todas las métricas posibles ---
    all_metrics = [
        'Total Actions Successful %',
        'Passes Accurate %',
        'Long Passes Accurate %',
        'Aerial Duels Won %',
        'Exits',
        'Save Rate %',
        'Saves With Reflexes %',
        'Prevented Goals',
        'Clean Sheets Ratio',
        'Dif CG-xCG'
    ]
    # Solo las que existen y tienen al menos 1 dato
    metrics_avail = [m for m in all_metrics if m in df_percentiles.columns and df_percentiles[m].notnull().sum() > 0]

    # --- Selección de jugador y métricas ---
    col1, col2 = st.columns(2)
    with col1:
        jugador = st.selectbox("Jugador", sorted(df_percentiles['Player'].dropna().unique()))
    with col2:
        selected_metrics = st.multiselect(
            "Selecciona las métricas", metrics_avail,
            default=[m for m in metrics_avail[:6]]  # Default primeras 6 que existan
        )

    if not selected_metrics:
        st.error("Selecciona al menos una métrica.")
        st.stop()

    player = df_percentiles[df_percentiles['Player'] == jugador]
    if player.empty:
        st.error(f"El jugador {jugador} no fue encontrado en el DataFrame.")
        st.stop()

    # ========== 1. GRAFICO DE BARRAS DE PERCENTILES ==========

    # Calcula percentil para cada métrica seleccionada
    bars = []
    percentiles = []
    player_vals = []
    for metric in selected_metrics:
        stat = df_percentiles[metric].astype(float).dropna()
        if stat.empty or player[metric].isna().all():
            percentiles.append(np.nan)
            player_vals.append(np.nan)
        else:
            val = player[metric].values[0]
            player_vals.append(val)
            try:
                perc = stats.percentileofscore(stat, val)
            except:
                perc = np.nan
            percentiles.append(perc)
        bars.append(metric)

    # Ordena de mayor a menor percentil para mejor visualización
    order = np.argsort(percentiles)[::-1]
    bars = np.array(bars)[order]
    percentiles = np.array(percentiles)[order]
    player_vals = np.array(player_vals)[order]

    fig, ax = plt.subplots(figsize=(12, max(4, len(bars)*0.7)))
    bars_display = bars if isinstance(bars, (list, np.ndarray)) else [bars]
    # Color por percentil
    colors = []
    for p in percentiles:
        if np.isnan(p):
            colors.append("#cccccc")
        elif p >= 70:
            colors.append("#7ED957")  # verde
        elif p >= 40:
            colors.append("#F4E04D")  # amarillo
        elif p >= 20:
            colors.append("#FFA500")  # naranja
        else:
            colors.append("#F46E6E")  # rojo

    ax.barh(bars, percentiles, color=colors, edgecolor='black')
    for i, (v, p) in enumerate(zip(player_vals, percentiles)):
        label = f"{v:.2f}" if not np.isnan(v) else "NA"
        ax.text(p+2 if not np.isnan(p) else 2, i, label, va='center', ha='left', fontsize=12, color="#444")

    ax.set_xlim(-2, 110)
    ax.set_xlabel("Percentil (dentro de todos los porteros)")
    ax.set_title(f"Perfil de percentiles - {jugador}", fontsize=18, weight='bold', pad=15)
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    # ========== 2. MINI-GRAFICOS DISTRIBUCION KDE/HIST ==========

    n_stats = len(selected_metrics)
    n_cols = 3
    n_rows = int(np.ceil(n_stats / n_cols))
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows*3.5))
    axes = np.array(axes).reshape(-1)
    col = "#8897f4"

    for i, ax in enumerate(axes):
        if i < n_stats:
            metric = selected_metrics[i]
            stat = df_percentiles[metric].astype(float).dropna()
            player_value = player[metric].values[0] if not player[metric].isna().all() else None
            try:
                percentile = stats.percentileofscore(stat, player_value) if player_value is not None else None
            except:
                percentile = None

            if len(stat) == 0:
                ax.set_facecolor("#fafafa")
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=14, color="gray")
                ax.set_title(metric, fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            elif len(stat.unique()) > 1:
                sns.kdeplot(stat, ax=ax, color=col, fill=True, legend=False, alpha=0.5, linewidth=2)
            else:
                ax.hist(stat, color=col, bins=5, alpha=0.5)

            if player_value is not None and not pd.isnull(player_value):
                ax.axvline(player_value, color='black', linewidth=2, zorder=10)
                ax.text(player_value, ax.get_ylim()[1]*0.9, f'{player_value:.2f}',
                        color='black', ha='center', va='top', fontsize=11, fontweight='bold', zorder=11)
                xmin, xmax = ax.get_xlim()
                rango = xmax - xmin
                if (player_value < xmin) or (player_value > xmax):
                    ax.set_xlim(min(xmin, player_value-0.1*rango), max(xmax, player_value+0.1*rango))
            titulo = f"{metric}"
            if percentile is not None and not np.isnan(percentile):
                titulo += f"\nPercentile {int(percentile)}"
            ax.set_title(titulo, fontsize=14)
            ax.set(xlabel=None)
            ax.set(ylabel=None)
            ax.set(yticks=[])
        else:
            ax.axis('off')

    fig2.text(0.01, 1.03, f'{jugador}', fontsize=26, fontweight='bold', ha='left', va='top')
    fig2.text(0.99, 1.03, f'Distribución KPI vs todos los porteros', fontsize=16, ha='right', va='top')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    st.pyplot(fig2)





    
# ====== TAB 4: GRÁFICOS JUGADORES ======
with tab4:
    st.header("Gráfico de comparación")
    # --- Parámetros interactivos ---
    min_matches = st.slider("Mínimo de partidos jugados", min_value=1, max_value=20, value=5)
    df = pd.read_excel('Goalkeeper_Stats_WyScout_2.xlsx')
    df["Dif CG-xCG"] = df["xCG"] - df["Conceded Goals"]

    # Solo jugadores con más de X partidos jugados
    nombres_validos = df["Nombre"].value_counts()
    nombres_validos = nombres_validos[nombres_validos >= min_matches].index
    df_filtrado = df[df["Nombre"].isin(nombres_validos)]

    # Limpia columnas innecesarias
    cols_a_quitar = [
        'Minutes Played','Conceded Goals', 'Shots Against','Exits','Total Actions','Total Actions Successful','Assists',
        'Duels','Duels Won','Aerial Duels','Aerial Duels Won','Losses','Losses Own Half','Recoveries Opposite Half',
        'Defensive Duels','Defensive Duels Won','Loose Ball Duels','Loose Ball Duels Won','Sliding Tackles',
        'Sliding Tackles Won','Clearances','Fouls','Yellow card','Red card','Yellow cards','Red cards','Through Passes',
        'Through Passes Accurate','xA','Passes To GK','Passes To GK Accurate'
    ]
    df_filtrado = df_filtrado.drop(cols_a_quitar, axis=1, errors='ignore')

    # Calcula promedio por jugador
    df_avg = df_filtrado.groupby("Nombre").mean(numeric_only=True).reset_index()
    # Elimina porteros no deseados (puedes editar esta lista)
    df_avg = df_avg[~df_avg["Nombre"].isin(["B. Kamara", "L. Carević", "K. Pirić"])]

    # --- Selectores interactivos ---
    # Solo columnas numéricas
    metricas_disponibles = [col for col in df_avg.columns if col != "Nombre" and pd.api.types.is_numeric_dtype(df_avg[col])]
    metric_value = st.selectbox("Selecciona la métrica a comparar", metricas_disponibles)
    jugadores_disponibles = df_avg["Nombre"].tolist()
    player1 = st.selectbox("Jugador 1", jugadores_disponibles, index=0)
    player2 = st.selectbox("Jugador 2", jugadores_disponibles, index=1 if len(jugadores_disponibles) > 1 else 0)

    # --- Ordena por la métrica seleccionada
    df_avg = df_avg.sort_values(by=metric_value, ascending=False)
    # Obtén los valores de cada jugador seleccionado
    try:
        p1_val = df_avg.loc[df_avg["Nombre"] == player1, metric_value].values[0]
        p2_val = df_avg.loc[df_avg["Nombre"] == player2, metric_value].values[0]
    except IndexError:
        st.error("Selecciona jugadores válidos.")
        st.stop()

    # --- Colores y fondo para Streamlit ---
    text_color = 'white'
    background = '#313332'

    # --- Gráfico ---
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.set_facecolor(background)
    ax.patch.set_facecolor(background)
    mpl.rcParams['xtick.color'] = text_color
    mpl.rcParams['ytick.color'] = text_color
    ax.grid(ls='dotted', lw=.5, color='lightgrey', axis='y', zorder=1)
    for x in ['top', 'bottom', 'left', 'right']:
        ax.spines[x].set_visible(False)
    # Swarmplot
    sns.swarmplot(x=metric_value, data=df_avg, color='white', zorder=1, ax=ax)
    offset = 0.13
    # Overlay jugadores seleccionados
    ax.scatter(p1_val, 0, c='red', edgecolor='white', s=200, zorder=2)
    ax.text(p1_val, 0 + offset, player1, color=text_color, ha='center', va='bottom', fontsize=11)
    ax.scatter(p2_val, 0, c='blue', edgecolor='white', s=200, zorder=2)
    ax.text(p2_val, 0 + offset, player2, color=text_color, ha='center', va='bottom', fontsize=11)
    ax.set_title(
        f"{metric_value}\n{player1} vs. {player2} (y resto porteros, min {min_matches} partidos)",
        color=text_color, fontsize=14, loc='center', pad=30
    )
    ax.set_xlabel(metric_value, color=text_color)
    # Ajusta el eje y para limpieza visual
    ax.set_ylim(-0.1, 0.25)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    st.pyplot(fig)

        # --- 1. Carga y preprocesado de datos ---
    df1 = pd.read_excel('Goalkeeper_Stats_WyScout_2.xlsx')
    df2 = pd.read_excel('WyScout Search results.xlsx')
    df1["Dif CG-xCG"] = df1["xCG"] - df1["Conceded Goals"]
    df2["Clean sheets ratio"] = df2["Clean sheets"] / df2["Matches played"]
    # Solo jugadores con más de N partidos jugados
    min_matches = st.slider("Mínimo de partidos jugados", min_value=1, max_value=20, value=5, key="slider2")
    nombres_validos = df1["Nombre"].value_counts()
    nombres_validos = nombres_validos[nombres_validos >= min_matches].index
    df_filtrado = df1[df1["Nombre"].isin(nombres_validos)]
    df_avg = df_filtrado.groupby("Nombre").mean(numeric_only=True).reset_index()
    df1 = df_avg.drop(['Level'], axis=1, errors="ignore")
    df_merged = pd.merge(
        df1, df2, left_on='Nombre', right_on='Player', how='inner', suffixes=('', '_df2')
    )
    cols_to_drop = [col for col in df_merged.columns if col.endswith('_df2')]
    df_merged = df_merged.drop(columns=cols_to_drop + ['Player'], errors="ignore")
    df_merged = df_merged.rename(columns={'Nombre': 'Player'})
    df_percentiles = df_merged

    # --- 2. KPIs y métricas ---
    key_stats = {
        'Key Stats': [
            'Passes Accurate %','Long Passes Accurate %','Through Passes Accurate %','Passes To Final Third Accurate %',
            'Losses Own Half %','Losses Opposite Half %','Aerial Duels Won %','Exits','Save Rate %','Saves With Reflexes %',
            'Prevented Goals','Clean Sheets Ratio'
        ]
    }

    # Calcula métricas derivadas si no existen
    derivadas = [
        ("Total Actions Successful %", ["Total Actions Successful", "Total Actions"], lambda x, y: 100*x/y),
        ("Passes Accurate %", ["Passes Accurate", "Passes"], lambda x, y: 100*x/y),
        ("Long Passes Accurate %", ["Long Passes Accurate", "Long Passes"], lambda x, y: 100*x/y),
        ("Through Passes Accurate %", ["Through Passes Accurate", "Through Passes"], lambda x, y: 100*x/y),
        ("Passes To Final Third Accurate %", ["Passes To Final Third Accurate", "Passes To Final Third"], lambda x, y: 100*x/y),
        ("Duels Won %", ["Duels Won", "Duels"], lambda x, y: 100*x/y),
        ("Defensive Duels Won %", ["Defensive Duels Won", "Defensive Duels"], lambda x, y: 100*x/y),
        ("Loose Ball Duels Won %", ["Loose Ball Duels Won", "Loose Ball Duels"], lambda x, y: 100*x/y),
        ("Aerial Duels Won %", ["Aerial Duels Won", "Aerial Duels"], lambda x, y: 100*x/y),
        ("Losses Own Half %", ["Losses Own Half", "Losses"], lambda x, y: 100*x/y),
        ("Losses Opposite Half %", ["Losses Own Half %"], lambda x: 100-x),
        ("Saves With Reflexes %", ["Saves With Reflexes", "Saves"], lambda x, y: 100*x/y),
    ]
    for name, cols, func in derivadas:
        if name not in df_percentiles.columns and all(c in df_percentiles.columns for c in cols):
            try:
                df_percentiles[name] = func(*[df_percentiles[c] for c in cols])
            except:
                df_percentiles[name] = np.nan

    # --- 3. Selección de métricas y jugadores ---
    metricas_disponibles = [col for grupo in key_stats.values() for col in grupo if col in df_percentiles.columns]
    st.write("Puedes modificar las métricas clave desde el código o añadir más opciones avanzadas.")
    jugadores_disponibles = df_percentiles["Player"].unique().tolist()
    jugadores_seleccionados = st.multiselect("Elige los jugadores a mostrar", jugadores_disponibles, default=jugadores_disponibles[:10])

    df_stats = df_percentiles[df_percentiles["Player"].isin(jugadores_seleccionados)][["Player"] + metricas_disponibles].dropna().reset_index(drop=True)

    # --- 4. Normalización 0-10 ---
    df_stats_scaled = df_stats.copy()
    for col in metricas_disponibles:
        min_val = df_stats_scaled[col].min()
        max_val = df_stats_scaled[col].max()
        if max_val > min_val:
            df_stats_scaled[col + "_score"] = 10 * (df_stats_scaled[col] - min_val) / (max_val - min_val)
        else:
            df_stats_scaled[col + "_score"] = 0

    score_cols = [col + "_score" for col in metricas_disponibles]

    # --- 5. Pesos y score final ---
    st.write("Pesos para cada KPI (entre 0.8 y 1 recomendado):")
    default_pesos = [0.9, 0.9, 0.8, 0.9, 0.9, 0.8, 0.9, 0.8, 1.0, 0.8, 1.0, 1.0][:len(score_cols)]
    pesos = []
    for i, col in enumerate(score_cols):
        pesos.append(st.number_input(f"{metricas_disponibles[i]}", min_value=0.1, max_value=2.0, value=float(default_pesos[i]), step=0.1, key=f"peso_{col}"))
    X_relative = df_stats_scaled[score_cols].values
    df_stats_scaled["player_score_10"] = (X_relative * pesos).sum(axis=1) / np.sum(pesos)

    # --- 6. Ranking y heatmap ---
    ranking_final = df_stats_scaled[["Player", "player_score_10"] + score_cols].sort_values("player_score_10", ascending=False).reset_index(drop=True)
    st.subheader("Ranking de porteros (Score 0-10)")
    st.dataframe(ranking_final[["Player", "player_score_10"]].style.background_gradient(cmap='YlGnBu'))

    # --- 7. Heatmap (top 10 o todos seleccionados) ---
    st.subheader("Heatmap de puntuaciones normalizadas")
    top10 = ranking_final.head(10).set_index("Player")
    scores_mtx = top10[score_cols]

    fig, ax = plt.subplots(figsize=(1+len(score_cols), max(8, 0.8*len(top10))))
    sns.heatmap(scores_mtx, annot=True, cmap="YlGnBu", cbar=True, linewidths=0.5, fmt=".1f", ax=ax)
    ax.set_title("Individual Score (0-10) - Best Goalkeepers", fontsize=14, weight="bold", color='black')
    ax.set_ylabel("Player", color='black', fontsize=12, weight='bold')
    ax.set_xlabel("KPI", color='black', fontsize=12, weight='bold')
    ax.set_yticklabels(top10.index, rotation=0, fontsize=12, weight='bold', color='black')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', color='black')
    plt.subplots_adjust(left=0.3, right=0.98, top=0.92, bottom=0.05)
    st.pyplot(fig)



