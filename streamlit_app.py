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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Radar comparativo defensivo", "Gráficos generales de equipos", "Percentiles", "Graficos jugadores", "Similitud entre jugadores"])

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
    
    # Cargar datos solo una vez (usa @st.cache_data si lo prefieres)
    df1 = pd.read_excel('Goalkeeper_Stats_WyScout_2.xlsx')
    df2 = pd.read_excel('WyScout Search results.xlsx')
    
    # Cálculos y limpieza igual que en tu notebook
    df1["Dif CG-xCG"] = df1["xCG"] - df1["Conceded Goals"]
    df2["Clean sheets ratio"] = df2["Clean sheets"] / df2["Matches played"]
    nombres_validos = df1["Nombre"].value_counts()
    nombres_validos = nombres_validos[nombres_validos >= 5].index
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
    df_percentiles["Total Actions Successful %"] = (df_percentiles["Total Actions Successful"]*100) / df_percentiles["Total Actions"]
    df_percentiles["Passes Accurate %"] = (df_percentiles["Passes Accurate"]*100) / df_percentiles["Passes"]
    df_percentiles["Through Passes Accurate %"] = (df_percentiles["Through Passes Accurate"]*100) / df_percentiles["Through Passes"]
    df_percentiles["Passes To Final Third Accurate %"] = (df_percentiles["Passes To Final Third Accurate"]*100) / df_percentiles["Passes To Final Third"]
    df_percentiles["Long Passes Accurate %"] = (df_percentiles["Long Passes Accurate"]*100) / df_percentiles["Long Passes"]
    df_percentiles["Duels Won %"] = (df_percentiles["Duels Won"]*100) / df_percentiles["Duels"]
    df_percentiles["Defensive Duels Won %"] = (df_percentiles["Defensive Duels Won"]*100) / df_percentiles["Defensive Duels"]
    df_percentiles["Loose Ball Duels Won %"] = (df_percentiles["Loose Ball Duels Won"]*100) / df_percentiles["Loose Ball Duels"]
    df_percentiles["Aerial Duels Won %"] = (df_percentiles["Aerial Duels Won"]*100) / df_percentiles["Aerial Duels"]
    df_percentiles["Losses Own Half %"] = (df_percentiles["Losses Own Half"]*100) / df_percentiles["Losses"]
    df_percentiles["Losses Opposite Half %"] = 100 - df_percentiles["Losses Own Half %"]
    df_percentiles["Saves With Reflexes %"] = (df_percentiles["Saves With Reflexes"]*100) / df_percentiles["Saves"]
    df_percentiles = df_percentiles.fillna(0)
    # ---- renombra columnas para evitar errores por nombre ----
    rename_dict = {
        'Yellow cards':'Yellow Cards',
        'Red cards':'Red Cards',
        'Minutes played':'Minutes Played',
        'Conceded goals per 90': 'Conceded Goals',
        'xG against per 90':'xG Against',
        'Prevented goals per 90':'Prevented Goals',
        'Shots against per 90':'Shots Against',
        'Save rate, %':'Save Rate %',
        'Exits per 90':'Exits',
        'Clean sheets ratio':'Clean Sheets Ratio'
    }
    df_percentiles = df_percentiles.rename(columns=rename_dict)
    
    # ---- Listado de jugadores ----
    player_list = sorted(df_percentiles["Player"].unique())
    player_name = st.selectbox("Selecciona jugador", player_list)
    
    # --- Métricas disponibles (personalízalo si quieres) ---
    stats_porteros = {
        'General': ['Total Actions Successful %', 'Fouls', 'Yellow Cards', 'Red Cards'],
        'Keeper': ['Saves','Saves With Reflexes %', 'Save Rate %', 'Conceded Goals', 'Prevented Goals', 'Exits', 'Clean Sheets Ratio'],
        'Passes': ['Received Passes','Passes Accurate %', 'Long Passes Accurate %', 'Through Passes Accurate %', 'Passes To Final Third Accurate %', 'Assists','xA', 'Losses Own Half %', 'Losses Opposite Half %'],
        'Defensive': ['Duels Won %','Defensive Duels Won %', 'Aerial Duels Won %', 'Loose Ball Duels Won %', 'Interceptions', 'Clearances']
    } 
    group_colors = {
        "General": "#1f77b4",       # azul
        "Keeper": "#2ca02c",       # verde
        "Passes": "#ff7f0e",         # naranja
        "Defensive": "#d62728"  # rojo
    }
    metrics_dict = stats_porteros  # Cambia aquí si quieres otro bloque
    metrics_list = []
    for group in metrics_dict.values():
        metrics_list.extend(group)
    percentile_cols = [f"{m} Percentil" for m in metrics_list]
    # Calcula percentiles solo una vez
    # Quita duplicados de columnas
    df_percentiles = df_percentiles.loc[:,~df_percentiles.columns.duplicated()]

    for metric in metrics_list:
        if metric in df_percentiles.columns and df_percentiles[metric].ndim == 1:
            stat = df_percentiles[metric].astype(float)
            df_percentiles[f"{metric} Percentil"] = stat.rank(pct=True) * 100
        
    df_player = df_percentiles[df_percentiles['Player'] == player_name]
    profile = pd.DataFrame({
        "Metric": metrics_list,
        "Percentile": [df_player[f"{m} Percentil"].values[0] if f"{m} Percentil" in df_player.columns else None for m in metrics_list]
    })
    # Asocia colores de grupo
    metric_to_group = {}
    for group, metrics in metrics_dict.items():
        for m in metrics:
            metric_to_group[m] = group
    profile['Grupo'] = profile['Metric'].map(metric_to_group)
    profile['ColorGrupo'] = profile['Grupo'].map(group_colors)
    def color_percentil(p):
        if pd.isna(p):
            return 'gray'
        elif p >= 70:
            return 'green'
        elif p >= 40:
            return 'yellow'
        elif p >= 20:
            return 'orange'
        else:
            return 'red'
    profile['Color'] = profile['Percentile'].apply(color_percentil)

    # --- Gráfico ---
    fig, ax = plt.subplots(figsize=(14, 7))
    profile_plot = profile[::-1]  # Invierte para que la primera quede arriba
    ax.hlines(
        y=profile_plot['Metric'], xmin=0, xmax=profile_plot['Percentile'],
        color=profile_plot['Color'], lw=4
    )
    ax.scatter(
        profile_plot['Percentile'], profile_plot['Metric'],
        color=profile_plot['Color'], s=120, zorder=3
    )
    # Etiquetas
    for idx, row in profile_plot.iterrows():
        if not pd.isna(row['Percentile']):
            ax.text(
                row['Percentile'] + 2, row['Metric'],
                f"{int(row['Percentile'])}",
                va='center', ha='left', fontsize=9, color='gray'
            )
    # Color en ylabels
    y_labels = ax.get_yticklabels()
    for label in y_labels:
        metric = label.get_text()
        color = profile_plot[profile_plot['Metric'] == metric]['ColorGrupo'].values[0]
        label.set_color(color)
        label.set_fontweight('bold')
    # Títulos de grupo
    grouped = profile_plot.groupby('Grupo')
    for group, data in grouped:
        color = data['ColorGrupo'].iloc[0]
        metric_central = data['Metric'].iloc[len(data)//2]
        ax.text(
            -35,  # Ajusta para mover a la izquierda
            metric_central,
            group,
            va='center',
            ha='right',
            fontsize=12,
            color=color,
            rotation=90,
            fontweight='bold'
        )
    ax.set_xlim(-2, 110)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Percentile Profile - {player_name}", fontsize=18, weight='bold', pad=20, x=0.33)
    ax.grid(axis='x', linestyle=':', alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    st.pyplot(fig)


    # ========== 2. MINI-GRAFICOS DISTRIBUCION KDE/HIST ==========

    n_stats = len(metrics_list)
    n_cols = 3
    n_rows = int(np.ceil(n_stats / n_cols))
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(13, n_rows*4))
    axes = np.array(axes).reshape(-1)
    col = "#8897f4"

    for i, ax in enumerate(axes):
        if i < n_stats:
            metric = metrics_list[i]
            stat = df_percentiles[metric].astype(float).dropna()
            player_value = df_player[metric].values[0] if not df_player[metric].isna().all() else None
            try:
                percentile = stats.percentileofscore(stat, player_value) if player_value is not None else None
            except:
                percentile = None

            if len(stat) == 0:
                ax.set_facecolor("#fafafa")
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=13, color="gray")
                ax.set_title(metric, fontsize=15)
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
            ax.set_title(titulo, fontsize=13)
            ax.set(xlabel=None)
            ax.set(ylabel=None)
            ax.set(yticks=[])
        else:
            ax.axis('off')

    fig2.text(0.01, 1.015, f'{player_name}', fontsize=20, fontweight='bold', ha='left', va='top')
    fig2.text(0.99, 1.015, f'Distribución KPI vs todos los porteros', fontsize=13, ha='right', va='top')
    plt.tight_layout(rect=[0.00, 0.00, 0.98, 0.96])
    st.pyplot(fig2)


# ====== TAB 4: GRÁFICOS JUGADORES ======
with tab4:
    st.header("Gráfico de comparación")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from soccerplots.radar_chart import Radar
    import seaborn as sns
    import streamlit as st

    # --- 1. PARÁMETROS INTERACTIVOS ---
    min_matches = st.slider("Mínimo de partidos jugados", min_value=1, max_value=20, value=5)
    df = pd.read_excel('Goalkeeper_Stats_WyScout_2.xlsx')
    df["Dif CG-xCG"] = df["xCG"] - df["Conceded Goals"]

    # Solo jugadores con más de X partidos jugados
    nombres_validos = df["Nombre"].value_counts()
    nombres_validos = nombres_validos[nombres_validos >= min_matches].index
    df_filtrado = df[df["Nombre"].isin(nombres_validos)]

    # Limpia columnas innecesarias
    cols_a_quitar = [ ... ]  # tu lista de columnas a quitar
    df_filtrado = df_filtrado.drop(cols_a_quitar, axis=1, errors='ignore')

    # Calcula promedio por jugador
    df_avg = df_filtrado.groupby("Nombre").mean(numeric_only=True).reset_index()
    df_avg = df_avg[~df_avg["Nombre"].isin(["B. Kamara", "L. Carević", "K. Pirić"])]

    # --- Selectores interactivos ---
    metricas_disponibles = [col for col in df_avg.columns if col != "Nombre" and pd.api.types.is_numeric_dtype(df_avg[col])]
    metric_value = st.selectbox("Selecciona la métrica a comparar", metricas_disponibles)
    jugadores_disponibles = df_avg["Nombre"].tolist()
    player1 = st.selectbox("Jugador 1", jugadores_disponibles, index=0)
    player2 = st.selectbox("Jugador 2", jugadores_disponibles, index=1 if len(jugadores_disponibles) > 1 else 0)

    # --- 2. SWARMPLOT (Gráfico barra comparativo simple) ---
    try:
        p1_val = df_avg.loc[df_avg["Nombre"] == player1, metric_value].values[0]
        p2_val = df_avg.loc[df_avg["Nombre"] == player2, metric_value].values[0]
    except IndexError:
        st.error("Selecciona jugadores válidos.")
        st.stop()

    fig, ax = plt.subplots(figsize=(10, 5))
    background = '#313332'
    text_color = 'white'
    fig.set_facecolor(background)
    ax.patch.set_facecolor(background)
    import matplotlib as mpl
    mpl.rcParams['xtick.color'] = text_color
    mpl.rcParams['ytick.color'] = text_color
    ax.grid(ls='dotted', lw=.5, color='lightgrey', axis='y', zorder=1)
    for x in ['top', 'bottom', 'left', 'right']:
        ax.spines[x].set_visible(False)
    sns.swarmplot(x=metric_value, data=df_avg, color='white', zorder=1, ax=ax)
    offset = 0.13
    ax.scatter(p1_val, 0, c='red', edgecolor='white', s=200, zorder=2)
    ax.text(p1_val, 0 + offset, player1, color=text_color, ha='center', va='bottom', fontsize=11)
    ax.scatter(p2_val, 0, c='blue', edgecolor='white', s=200, zorder=2)
    ax.text(p2_val, 0 + offset, player2, color=text_color, ha='center', va='bottom', fontsize=11)
    ax.set_title(
        f"{metric_value}\n{player1} vs. {player2} (y resto porteros, min {min_matches} partidos)",
        color=text_color, fontsize=14, loc='center', pad=30
    )
    ax.set_xlabel(metric_value, color=text_color)
    ax.set_ylim(-0.1, 0.25)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    st.pyplot(fig)

    # --- 3. HEATMAP DE PUNTUACIONES NORMALIZADAS ---
    metricas_heatmap = metricas_disponibles.copy()
    df_stats = df_avg[["Nombre"] + metricas_heatmap].dropna().reset_index(drop=True)
    df_stats_scaled = df_stats.copy()
    for col in metricas_heatmap:
        min_val = df_stats_scaled[col].min()
        max_val = df_stats_scaled[col].max()
        if max_val > min_val:
            df_stats_scaled[col + "_score"] = 10 * (df_stats_scaled[col] - min_val) / (max_val - min_val)
        else:
            df_stats_scaled[col + "_score"] = 0
    score_cols = [col + "_score" for col in metricas_heatmap]
    df_stats_scaled["player_score_10"] = df_stats_scaled[score_cols].mean(axis=1)
    ranking_final = df_stats_scaled[["Nombre", "player_score_10"] + score_cols].sort_values("player_score_10", ascending=False).reset_index(drop=True)
    top10 = ranking_final.head(10).set_index("Nombre")
    scores_mtx = top10[score_cols]
    fig_heat, ax_heat = plt.subplots(figsize=(1+len(score_cols), max(8, 0.8*len(top10))))
    sns.heatmap(scores_mtx, annot=True, cmap="YlGnBu", cbar=True, linewidths=0.5, fmt=".1f", ax=ax_heat)
    ax_heat.set_title("Individual Score (0-10) - Best Goalkeepers", fontsize=14, weight="bold", color='black')
    ax_heat.set_ylabel("Player", color='black', fontsize=12, weight='bold')
    ax_heat.set_xlabel("KPI", color='black', fontsize=12, weight='bold')
    ax_heat.set_yticklabels(top10.index, rotation=0, fontsize=12, weight='bold', color='black')
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha='right', color='black')
    plt.subplots_adjust(left=0.3, right=0.98, top=0.92, bottom=0.05)
    st.pyplot(fig_heat)

    # --- 4. RADAR PARA COMPARAR AMBOS JUGADORES EN VARIAS MÉTRICAS ---
    from soccerplots.radar_chart import Radar

    # Radar sólo con las métricas seleccionadas arriba
    radar_metrics = st.multiselect("Métricas a comparar en el radar", metricas_disponibles, default=metricas_disponibles[:6])
    params = [m for m in radar_metrics if m in df_avg.columns and pd.api.types.is_numeric_dtype(df_avg[m])]
    if len(params) < 3:
        st.warning("El radar necesita al menos 3 métricas seleccionadas.")
    else:
        ranges = []
        for m in params:
            serie = df_avg[m].astype(float)
            mini = serie.min()
            maxi = serie.max()
            if pd.isnull(mini) or pd.isnull(maxi) or mini == maxi:
                mini, maxi = 0, 1
            ranges.append((mini, maxi))
        a_row = df_avg[df_avg['Nombre'] == player1]
        b_row = df_avg[df_avg['Nombre'] == player2]
        a_values = [float(a_row[m].values[0]) if not a_row.empty and not pd.isnull(a_row[m].values[0]) else 0 for m in params]
        b_values = [float(b_row[m].values[0]) if not b_row.empty and not pd.isnull(b_row[m].values[0]) else 0 for m in params]
        values = [a_values, b_values]
        title = dict(
            title_name=player1, title_color='blue',
            subtitle_name='', subtitle_color='blue',
            title_name_2=player2, title_color_2='red',
            subtitle_name_2='', subtitle_color_2='red',
            title_fontsize=18, subtitle_fontsize=15
        )
        endnote = '@futboldata_pafos'
        radar = Radar()
        fig2, ax2 = radar.plot_radar(
            ranges=ranges,
            params=params,
            values=values,
            radar_color=['blue', 'red'],
            alphas=[.75, .6],
            title=title,
            endnote=endnote,
            compare=True
        )
        st.pyplot(fig2)




# ====== TAB 5: SIMILITUD ======
with tab5:
    st.header("Similitud entre jugadores")


    import streamlit as st
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.spatial.distance import cdist
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    # =============== CARGA Y PREPROCESADO DE DATOS ===============
   
    df1 = pd.read_excel('Goalkeeper_Stats_WyScout_2.xlsx')
    df2 = pd.read_excel('WyScout Search results.xlsx')

    df1["Dif CG-xCG"] = df1["xCG"] - df1["Conceded Goals"]
    df2["Clean sheets ratio"] = df2["Clean sheets"] / df2["Matches played"]

    nombres_validos = df1["Nombre"].value_counts()
    nombres_validos = nombres_validos[nombres_validos >= 5].index
    df_filtrado = df1[df1["Nombre"].isin(nombres_validos)]
    df_avg = df_filtrado.groupby("Nombre").mean(numeric_only=True).reset_index()
    df1 = df_avg.drop(['Level'], axis=1, errors="ignore")

    df_merged = pd.merge(
        df1, df2, left_on='Nombre', right_on='Player', how='inner', 
        suffixes=('', '_df2')
    )
    cols_to_drop = [col for col in df_merged.columns if col.endswith('_df2')]
    df_merged = df_merged.drop(columns=cols_to_drop + ['Player'], errors="ignore")
    df_merged = df_merged.rename(columns={'Nombre': 'Player'})

    df_percentiles = df_merged
    df_percentiles["Total Actions Successful %"] = (df_percentiles["Total Actions Successful"]*100) / df_percentiles["Total Actions"]
    df_percentiles["Passes Accurate %"] = (df_percentiles["Passes Accurate"]*100) / df_percentiles["Passes"]
    df_percentiles["Through Passes Accurate %"] = (df_percentiles["Through Passes Accurate"]*100) / df_percentiles["Through Passes"]
    df_percentiles["Passes To Final Third Accurate %"] = (df_percentiles["Passes To Final Third Accurate"]*100) / df_percentiles["Passes To Final Third"]
    df_percentiles["Long Passes Accurate %"] = (df_percentiles["Long Passes Accurate"]*100) / df_percentiles["Long Passes"]
    df_percentiles["Duels Won %"] = (df_percentiles["Duels Won"]*100) / df_percentiles["Duels"]
    df_percentiles["Defensive Duels Won %"] = (df_percentiles["Defensive Duels Won"]*100) / df_percentiles["Defensive Duels"]
    df_percentiles["Loose Ball Duels Won %"] = (df_percentiles["Loose Ball Duels Won"]*100) / df_percentiles["Loose Ball Duels"]
    df_percentiles["Aerial Duels Won %"] = (df_percentiles["Aerial Duels Won"]*100) / df_percentiles["Aerial Duels"]
    df_percentiles["Losses Own Half %"] = (df_percentiles["Losses Own Half"]*100) / df_percentiles["Losses"]
    df_percentiles["Losses Opposite Half %"] = 100 - df_percentiles["Losses Own Half %"]
    df_percentiles["Saves With Reflexes %"] = (df_percentiles["Saves With Reflexes"]*100) / df_percentiles["Saves"]

    df_percentiles = df_percentiles[~df_percentiles["Player"].isin(["B. Kamara", "L. Carević", "K. Pirić"])]

    df_percentiles = df_percentiles.drop([
        'Total Actions', 'Total Actions Successful', 'Passes', 'Passes Accurate', 'Long Passes', 'Long Passes Accurate', 
        'Duels', 'Duels Won', 'Aerial Duels Won', 'Aerial Duels','Losses Own Half','Losses','Recoveries','Recoveries Opposite Half',
        'Yellow card','Red card','Defensive Duels','Defensive Duels Won','Loose Ball Duels','Loose Ball Duels Won',
        'Sliding Tackles','Sliding Tackles Won','Through Passes','Through Passes Accurate','Passes To Final Third',
        'Passes To Final Third Accurate','xCG','Saves With Reflexes','Passes To GK','Passes To GK Accurate','Goal Kicks',
        'Short Goal Kicks','Long Goal Kicks','Matches played','Clean sheets','Aerial duels per 90','Exits','Prevented goals',
        'Shots Against','Shots against','xG against','Conceded goals','Conceded Goals','Minutes Played','Level','Country','Position','Dif CG-xCG'
    ], axis=1, errors='ignore')

    df_percentiles = df_percentiles.rename(columns={
        'Yellow cards':'Yellow Cards',
        'Red cards':'Red Cards',
        'Minutes played':'Minutes Played',
        'Conceded goals per 90': 'Conceded Goals',
        'xG against per 90':'xG Against',
        'Prevented goals per 90':'Prevented Goals',
        'Shots against per 90':'Shots Against',
        'Save rate, %':'Save Rate %',
        'Exits per 90':'Exits',
        'Clean sheets ratio':'Clean Sheets Ratio'
    })

    # =============== SELECCIÓN DE MÉTRICAS Y JUGADORES ===============
    key_stats = {
        'Key Stats': ['Minutes Played','Total Actions Successful %','Passes Accurate %','Long Passes Accurate %','Through Passes Accurate %','Passes To Final Third Accurate %','Losses Own Half %','Losses Opposite Half %','Aerial Duels Won %','Exits','Save Rate %','Saves With Reflexes %','Prevented Goals','Clean Sheets Ratio']
    }

    metrics_dict = key_stats  # Puedes cambiar por stats_defensores, stats_medios, etc.

    metrics_list = []
    for group in metrics_dict.values():
        metrics_list.extend(group)

    # Solo jugadores en el dataset final
    nombres = df_percentiles["Player"].drop_duplicates().sort_values().tolist()
    jugador_ref = st.selectbox("Selecciona jugador de referencia", nombres, index=0)

    df_filtrado = df_percentiles[df_percentiles['Player'].isin(nombres)]
    df = df_filtrado

    # =============== CÁLCULO DE SIMILITUD ===============

    df_stats = df[["Player"] + metrics_list].dropna().reset_index(drop=True)
    X = df_stats[metrics_list].astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if jugador_ref not in df_stats["Player"].values:
        st.error(f"'{jugador_ref}' no está en el DataFrame filtrado.")
        st.stop()

    idx_ref = df_stats[df_stats["Player"] == jugador_ref].index[0]
    ref_vector = X_scaled[idx_ref].reshape(1, -1)

    distancias = cdist(ref_vector, X_scaled, metric='euclidean').flatten()
    df_stats["distancia"] = distancias
    df_resultado = df_stats.sort_values("distancia").reset_index(drop=True)
    df_resultado_filtrado = df_resultado[df_resultado["Player"] != jugador_ref].copy()
    df_resultado_filtrado["similitud"] = 1 / (1 + df_resultado_filtrado["distancia"])
    top_similares = df_resultado_filtrado[["Player", "similitud"]].head(5)
    top_similares = top_similares.iloc[::-1]  # Más similar arriba

    # =============== GRÁFICO ===============
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top_similares["Player"], top_similares["similitud"], color='royalblue')
    ax.set_xlabel("0 (Nada similar) — 1 (Idéntico)")
    ax.set_title(f"Jugadores más similares a {jugador_ref}", fontsize=15, weight='bold')
    ax.set_xlim(0, 1.05)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.3f}', va='center', fontsize=11)

    metricas_texto = "\n".join(metrics_list)
    ax.text(1.06, 0.5, "KPIs:\n" + metricas_texto,
            fontsize=10, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
            va='center', ha='left', transform=ax.transAxes)

    tick_labels = ax.get_yticklabels() + ax.get_xticklabels()
    for label in tick_labels:
        label.set_color('black')


    plt.tight_layout()
    st.pyplot(fig)

