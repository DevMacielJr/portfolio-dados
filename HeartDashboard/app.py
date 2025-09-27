import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Configuração da Página
st.set_page_config(page_title="Dashboard Cardiológico", layout="wide")

# ======================================== CABEÇALHO PERSONALIZADO + PARTICLES ===============================================

# Controle manual do tema na sidebar
st.sidebar.markdown("---")
tema = st.sidebar.selectbox(
    "Tema do Dashboard",
    options=["Automático", "Claro", "Escuro"],
    index=0,
    help="Escolha o tema manualmente ou deixe automático para seguir o sistema"
)

# CSS para forçar tema claro
tema_claro = """
<style>
    html, body, [class*="css"] {
        background-color: #f4f6f8 !important;
        color: #333 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #ecf0f1 !important;
    }
    .stButton > button {
        background-color: #3498DB !important;
        color: white !important;
    }
    .stButton > button:hover {
        background-color: #2980B9 !important;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff !important;
        color: #333 !important;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin: 5px;
    }
</style>
"""

# CSS para forçar tema escuro
tema_escuro = """
<style>
    html, body, [class*="css"] {
        background-color: #0e1117 !important;
        color: #FAFAFA !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0e1117 !important;
    }
    .stButton > button {
        background-color: #1f77b4 !important;
        color: #FAFAFA !important;
    }
    .stButton > button:hover {
        background-color: #155d8a !important;
    }
    div[data-testid="metric-container"] {
        background-color: #1e1e1e !important;
        color: #FAFAFA !important;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
</style>
"""

if tema == "Claro":
    st.markdown(tema_claro, unsafe_allow_html=True)
elif tema == "Escuro":
    st.markdown(tema_escuro, unsafe_allow_html=True)

# PARTICLE BACKGROUND E CABEÇALHO CENTRALIZADO
st.markdown("""
<div id="particles-js"></div>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap');

#particles-js {
    position: fixed;
    width: 100%;
    height: 100%;
    z-index: -1;
    top: 0;
    left: 0;
}

.header {
    position: relative;
    text-align: center;
    margin: 40px auto 20px auto;
    font-family: 'Montserrat', sans-serif;
}

.header h1 {
    font-size: 58px;
    font-weight: 900;
    color: #c62828;
    letter-spacing: 1px;
    margin-bottom: 10px;
}

@media (max-width: 768px) {
    .header h1 {
        font-size: 38px;
    }
}
</style>

<div class="header">
    <h1><i class="fa-solid fa-heart-pulse"></i> Heart Disease UCI</h1>
</div>

<script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
<script>
particlesJS('particles-js',
{
  "particles": {
    "number": {
      "value": 60
    },
    "color": {
      "value": "#ffffff"
    },
    "shape": {
      "type": "circle"
    },
    "opacity": {
      "value": 0.2
    },
    "size": {
      "value": 3
    },
    "line_linked": {
      "enable": true,
      "distance": 150,
      "color": "#ffffff",
      "opacity": 0.2,
      "width": 1
    },
    "move": {
      "enable": true,
      "speed": 2
    }
  },
  "interactivity": {
    "events": {
      "onhover": {
        "enable": true,
        "mode": "grab"
      },
      "onclick": {
        "enable": true,
        "mode": "push"
      }
    }
  },
  "retina_detect": true
});
</script>
""", unsafe_allow_html=True)

# Espaço após cabeçalho
st.markdown("""<div style='margin-top: 30px;'></div>""", unsafe_allow_html=True)

# ========================================================== BASE DE DADOS ================================================================

# Carregamento dos dados
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

df = load_data()

# Sidebar: filtros expandidos
st.sidebar.header("Filtros de Pacientes")

# Sexo
sexo = st.sidebar.radio("Sexo", ["Todos", "Homem", "Mulher"])

# Faixa etária
idade_min, idade_max = st.sidebar.slider(
    "Faixa Etária",
    int(df.age.min()),
    int(df.age.max()),
    (30, 60)
)

# Faixa pressão arterial em repouso (trestbps)
pressao_min, pressao_max = st.sidebar.slider(
    "Pressão Arterial em Repouso (trestbps)",
    int(df.trestbps.min()),
    int(df.trestbps.max()),
    (df.trestbps.min(), df.trestbps.max())
)

# Faixa colesterol (chol)
colesterol_min, colesterol_max = st.sidebar.slider(
    "Colesterol (chol)",
    int(df.chol.min()),
    int(df.chol.max()),
    (df.chol.min(), df.chol.max())
)

# Faixa frequência cardíaca máxima (thalach)
freqcard_max_min, freqcard_max_max = st.sidebar.slider(
    "Frequência Cardíaca Máxima (thalach)",
    int(df.thalach.min()),
    int(df.thalach.max()),
    (df.thalach.min(), df.thalach.max())
)

# Aplicar filtros no dataframe
df_filtrado = df.copy()

if sexo != "Todos":
    valor = 1 if sexo == "Homem" else 0
    df_filtrado = df_filtrado[df_filtrado["sex"] == valor]

df_filtrado = df_filtrado[
    (df_filtrado["age"] >= idade_min) & (df_filtrado["age"] <= idade_max) &
    (df_filtrado["trestbps"] >= pressao_min) & (df_filtrado["trestbps"] <= pressao_max) &
    (df_filtrado["chol"] >= colesterol_min) & (df_filtrado["chol"] <= colesterol_max) &
    (df_filtrado["thalach"] >= freqcard_max_min) & (df_filtrado["thalach"] <= freqcard_max_max)
]

# KPIs expandidos
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Pacientes Filtrados", df_filtrado.shape[0])

if df_filtrado.shape[0] > 0:
    col2.metric("Com Doença (%)", f"{df_filtrado['target'].mean() * 100:.1f}%")
    col3.metric("Sem Doença", df_filtrado['target'].value_counts().get(0, 0))
    col4.metric("Idade Média", f"{df_filtrado['age'].mean():.1f} anos")
    col5.metric("Colesterol Médio", f"{df_filtrado['chol'].mean():.0f} mg/dL")
else:
    col2.metric("Com Doença (%)", "N/A")
    col3.metric("Sem Doença", "N/A")
    col4.metric("Idade Média", "N/A")
    col5.metric("Colesterol Médio", "N/A")

st.divider()

# Abas para organizar visualizações
st.markdown("""
<style>
/* Container das abas */
div[role="tablist"] {
    border-bottom: 2px solid #3498DB;
    margin-bottom: 20px;
}

/* Estilo das abas */
div[role="tab"] {
    color: #555555;
    font-weight: 600;
    padding: 10px 20px;
    margin-right: 15px;
    border-radius: 0;
    border: none;
    background-color: transparent;
    transition: all 0.3s ease;
    cursor: pointer;
    font-size: 16px;
}

/* Aba ativa: underline azul e texto azul */
div[role="tab"][aria-selected="true"] {
    color: #3498DB;
    border-bottom: 3px solid #3498DB;
    font-weight: 700;
}

/* Hover: aumenta e muda cor */
div[role="tab"]:hover {
    color: #1D5BBB;
    font-size: 17.5px;
    font-weight: 700;
    transition: all 0.2s ease;
}
</style>
""", unsafe_allow_html=True)

# Definição das abas, só texto simples
aba1, aba2, aba3, aba4, aba5 = st.tabs([
    "Visão Geral", "Gráficos", "Animações", "Modelo Preditivo", "Sobre o Autor"
])

# Aba 1 - Visão Geral
with aba1:
    st.markdown("""
    ## Sobre a Base de Dados: Heart Disease UCI

    A base de dados Heart Disease UCI é um conjunto de dados amplamente utilizado para estudos e projetos relacionados à previsão de doenças cardíacas. Ela contém informações clínicas e demográficas de pacientes, coletadas de diversos institutos médicos, com o objetivo de auxiliar na identificação de fatores de risco associados a problemas cardiovasculares.

    ### • Para que serve?
    Essa base é frequentemente utilizada por pesquisadores, cientistas de dados e estudantes para:

    - Desenvolver modelos de machine learning que prevejam a presença de doenças cardíacas.
    - Identificar padrões e correlações entre variáveis como idade, colesterol, pressão arterial e outros indicadores de saúde.
    - Apoiar estudos na área de medicina preventiva, ajudando a entender quais fatores mais contribuem para doenças cardiovasculares.

    ### • Como funciona?
    O dataset contém 14 atributos (colunas) que incluem:

    - **Dados clínicos:** pressão arterial (trestbps), colesterol (chol), frequência cardíaca máxima atingida (thalach), entre outros.
    - **Dados demográficos:** idade (age), sexo (sex).
    - **Resultado diagnóstico:** presença ou ausência de doença cardíaca (target – onde 0 indica ausência e 1 indica presença).

    Os dados são estruturados de forma tabular, permitindo análises estatísticas e treinamento de algoritmos de classificação.

    ### • Origem e Aplicações
    Originalmente disponibilizada pelo UCI Machine Learning Repository, essa base é um recurso valioso para:

    - Aprendizado de máquina: Testar algoritmos como Regressão Logística, Random Forest e Redes Neurais.
    - Visualização de dados: Criar gráficos para entender a distribuição dos fatores de risco.
    - Educação: Usada em cursos de ciência de dados e bioinformática para exemplos práticos.

    Se você está interessado em saúde, análise de dados ou machine learning, essa base oferece uma ótima oportunidade para explorar como a tecnologia pode auxiliar na detecção precoce de doenças.

 **Fonte:** [Kaggle - Heart Disease UCI Dataset](https://www.kaggle.com/datasets/mragpavank/heart-diseaseuci?resource=download)

    """, unsafe_allow_html=True)
# CSS para estilizar o botão de download
st.markdown("""
<style>
/* Seleciona o botão de download */
div.stDownloadButton > button {
    background-color: #3498DB !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    transition: background-color 0.3s ease;
}

/* Efeito hover */
div.stDownloadButton > button:hover {
    background-color: #2980B9 !important;
    color: #fff !important;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# Exportar dados filtrados
st.subheader("Exportar Dados Filtrados:")
csv = df_filtrado.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Baixar Dados Filtrados (.csv)",
    data=csv,
    file_name="dados_filtrados.csv",
    mime="text/csv",
    help="Clique para baixar os dados filtrados no formato CSV"
)

# Aba 2 - Gráficos Estáticos
with aba2:
    st.subheader("Gráficos Interativos")

 # Explicação geral
    st.markdown("""
    Esta seção apresenta gráficos que ajudam a entender melhor a distribuição dos dados e as relações entre variáveis clínicas e demográficas.
    """)

    # Gráfico de Pizza - Distribuição do Diagnóstico
    df_pie = df_filtrado.copy()
    df_pie['diagnóstico'] = df_pie['target'].map({0: 'Sem Doença', 1: 'Com Doença'})
    fig_pie = px.pie(
        df_pie,
        names='diagnóstico',
        color='diagnóstico',
        color_discrete_map={'Sem Doença': "#3498DB", 'Com Doença': "#E74C3C"},
        hole=0.4,
        title="Distribuição Relativa dos Diagnósticos"
    )
    fig_pie.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_pie, use_container_width=True)
    st.caption("Proporção de pacientes com e sem doença cardíaca na amostra filtrada.")

    st.markdown("---")

    # Layout de 3 colunas para outros gráficos
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(" · Dispersão: Idade vs Colesterol")
        fig_scatter = px.scatter(
            df_filtrado,
            x='age',
            y='chol',
            color='target',
            color_discrete_map={0: '#2ECC71', 1: '#E74C3C'},
            labels={'chol': 'Colesterol (mg/dL)', 'age': 'Idade (anos)'},
            title="Relação entre Idade e Colesterol"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Pacientes coloridos por presença (vermelho) ou ausência (verde) de doença cardíaca.")

    with col2:
        st.subheader(" · Histograma de Variável")
        col_sel = st.selectbox(
            "Escolha uma variável numérica para visualizar a distribuição:",
            df_filtrado.select_dtypes(include='number').columns,
            key="hist_var"
        )
        fig_hist = px.histogram(
            df_filtrado,
            x=col_sel,
            color='target',
            barmode='overlay',
            nbins=20,
            color_discrete_map={0: '#2980B9', 1: '#C0392B'},
            title=f"Distribuição de {col_sel}"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.caption(f"Comparação da distribuição de '{col_sel}' entre pacientes com e sem doença cardíaca.")

    with col3:
        st.subheader(" · Boxplot por Diagnóstico")
        col_box = st.selectbox(
            "Escolha uma variável numérica para comparar por diagnóstico:",
            df_filtrado.select_dtypes(include='number').columns,
            key="box_var"
        )
        fig_box = px.box(
            df_filtrado,
            x='target',
            y=col_box,
            color='target',
            color_discrete_map={0: '#2980B9', 1: '#C0392B'},
            labels={'target': 'Diagnóstico', 'y': col_box},
            title=f"Distribuição de {col_box} por Diagnóstico"
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption(f"Visualização da variação e outliers da variável '{col_box}' para cada grupo diagnóstico.")

    st.markdown("---")

    # Gráfico de barras para variável categórica (sexo)
    st.subheader(" · Contagem por Sexo e Diagnóstico")
    if 'sex' in df_filtrado.columns:
        df_sex = df_filtrado.copy()
        df_sex['sexo'] = df_sex['sex'].map({0: 'Mulher', 1: 'Homem'})
        fig_bar = px.bar(
            df_sex,
            x='sexo',
            color='target',
            color_discrete_map={0: '#2980B9', 1: '#C0392B'},
            barmode='group',
            labels={'target': 'Diagnóstico', 'sexo': 'Sexo'},
            title="Número de Pacientes por Sexo e Diagnóstico"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Distribuição do diagnóstico cardíaco entre homens e mulheres.")

    st.markdown("---")

    # Heatmap de Correlação
    st.subheader(" · Correlação entre Variáveis")
    st.markdown(
        "Mapa de calor mostrando a correlação entre as variáveis numéricas do conjunto filtrado. "
        "Valores próximos de 1 ou -1 indicam forte correlação positiva ou negativa, respectivamente."
    )
    fig_corr, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        df_filtrado.corr(),
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.75}
    )
    st.pyplot(fig_corr)

# Aba 3 - Animação Interativa
with aba3:
    st.subheader("Animações Interativas")

    if df_filtrado.shape[0] < 15:
        st.warning("⚠️ Dados insuficientes para exibir animações.")
    else:
        # Animação 1: Idade x Colesterol animado por Frequência Cardíaca (thalach)
        if 'thalach' in df_filtrado.columns:
            st.markdown("** ‣ Idade vs Colesterol animado por Frequência Cardíaca**")
            st.markdown("""
            ·  **Idade** e **Colesterol** são fatores importantes para a saúde cardíaca.  
              ↳  A animação mostra como esses fatores mudam conforme a **Frequência Cardíaca Máxima** que o paciente atingiu.  
            """)
            fig_anim1 = px.scatter(df_filtrado, x='age', y='chol',
                                   animation_frame='thalach',
                                   color='target',
                                   color_discrete_map={0: 'lightblue', 1: 'red'},
                                   labels={'chol': 'Colesterol', 'age': 'Idade', 'thalach': 'Freq. Cardíaca'},
                                   title="Idade vs Colesterol com Frequência Cardíaca",
                                   range_x=[df_filtrado.age.min(), df_filtrado.age.max()],
                                   range_y=[df_filtrado.chol.min(), df_filtrado.chol.max()])
            st.plotly_chart(fig_anim1, use_container_width=True)
        else:
            st.info("ℹ️ Coluna 'thalach' não encontrada para animação 1.")

        st.markdown("---")

        # Animação 2: Idade x Frequência Cardíaca animado por Colesterol
        if 'chol' in df_filtrado.columns:
            st.markdown("**‣  Animação 2: Idade vs Frequência Cardíaca animado por Colesterol**")
            st.markdown("""
             ·  Veja a relação entre a **Idade** e a **Frequência Cardíaca Máxima** atingida.  
                ↳  A animação mostra como o nível de **Colesterol** influencia essa relação.  
            """)
            fig_anim2 = px.scatter(df_filtrado, x='age', y='thalach',
                                   animation_frame='chol',
                                   color='target',
                                   color_discrete_map={0: 'lightgreen', 1: 'darkred'},
                                   labels={'thalach': 'Freq. Cardíaca', 'age': 'Idade', 'chol': 'Colesterol'},
                                   title="Idade vs Frequência Cardíaca com Colesterol",
                                   range_x=[df_filtrado.age.min(), df_filtrado.age.max()],
                                   range_y=[df_filtrado.thalach.min(), df_filtrado.thalach.max()])
            st.plotly_chart(fig_anim2, use_container_width=True)
        else:
            st.info("ℹ️ Coluna 'chol' não encontrada para animação 2.")

        st.markdown("---")

        # Animação 3: Contagem de casos por faixa etária animado por diagnóstico
        st.markdown("**‣  Animação 3: Contagem por Faixa Etária e Diagnóstico**")
        st.markdown("""
         ·  Este gráfico mostra o número de pacientes em cada **faixa etária** (ex: 30-39 anos, 40-49 anos, etc.).  
            ↳  A animação destaca quantos pacientes foram diagnosticados com ou sem doença cardíaca em cada faixa.  
        Isso ajuda a entender como o risco muda conforme envelhecemos.
        """)
        # Criar faixa etária categórica
        bins = [29, 39, 49, 59, 69, 79]
        labels = ['30-39', '40-49', '50-59', '60-69', '70-79']
        df_anim = df_filtrado.copy()
        df_anim['faixa_etaria'] = pd.cut(df_anim['age'], bins=bins, labels=labels, right=True)
        # Contar por faixa etária e target
        df_anim_count = df_anim.groupby(['faixa_etaria', 'target']).size().reset_index(name='contagem')
        # Gerar gráfico de barras animado pela faixa etária
        fig_anim3 = px.bar(df_anim_count, x='target', y='contagem', color='target',
                           animation_frame='faixa_etaria',
                           labels={'target': 'Diagnóstico', 'contagem': 'Número de Pacientes'},
                           color_discrete_map={0: '#2980B9', 1: '#C0392B'},
                           category_orders={"target": [0,1]},
                           title="Número de Pacientes por Diagnóstico e Faixa Etária")
        fig_anim3.update_layout(xaxis=dict(tickmode='array', tickvals=[0,1], ticktext=['Sem Doença', 'Com Doença']))
        st.plotly_chart(fig_anim3, use_container_width=True)

# Aba 4 - Modelo Preditivo SVM
with aba4:
    st.subheader("Modelo Preditivo - SVM")

    if df_filtrado["target"].nunique() == 2 and df_filtrado.shape[0] > 20:
        X = pd.get_dummies(df_filtrado.drop('target', axis=1),
                           columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
        y = df_filtrado['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = SVC(kernel='linear', random_state=0, probability=True)  # probability=True para ROC
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Métricas em linha
        colM1, colM2, colM3, colM4 = st.columns(4)
        colM1.metric("Acurácia", f"{acc:.2%}")
        colM2.metric("Precisão", f"{precision:.2%}")
        colM3.metric("Recall", f"{recall:.2%}")
        colM4.metric("F1-Score", f"{f1:.2%}")

        # Gráficos lado a lado: matriz de confusão + ROC
        colG1, colG2 = st.columns(2)

        with colG1:
            fig_cm, ax = plt.subplots(figsize=(3, 2))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Sem Doença", "Com Doença"],
                        yticklabels=["Sem Doença", "Com Doença"], ax=ax)
            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")
            st.pyplot(fig_cm)

        with colG2:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax = plt.subplots(figsize=(3, 2))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Falso Positivo')
            ax.set_ylabel('Verdadeiro Positivo')
            ax.set_title('Curva ROC')
            ax.legend(loc="lower right")
            st.pyplot(fig_roc)

        # Importância dos coeficientes
        st.markdown("**Importância dos Atributos (coeficientes do modelo):**")
        coef_df = pd.DataFrame({
            "Atributo": X.columns,
            "Coeficiente": model.coef_[0]
        }).sort_values(by="Coeficiente", key=abs, ascending=False)
        st.dataframe(coef_df)

    else:
        st.warning("⚠️ Dados insuficientes ou apenas uma classe presente para treinar o modelo.")

# Aba 5 - Sobre o autor
with aba5:
    st.subheader("Sobre o Autor")

    # CSS para foto redonda e elegante
    st.markdown("""
    <style>
    .foto-perfil {
        border-radius: 50%;
        width: 300px;
        height: 300px;
        object-fit: cover;
        border: 4px solid #c62828;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        # Substitua o link abaixo pela URL da sua foto no GitHub
        st.markdown("""
        <img src="https://github.com/DevMacielJr.png" class="foto-perfil">
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        ### Edson Maciel
        Estudante de Ciências & Tecnologias e Engenheiro da Computação com foco em ciências de dados, análises de saúde e Machine Learning.  
        Apaixonado por transformar dados em insights que podem salvar vidas.

        -  Formação: Engenharia da Computação, Ciências e Tecnologia  
        -  Ferramentas Utilizadas: Python, Pandas, Streamlit, Scikit-learn, Plotly  
        -  Contato: [edson.maciel.017@ufrn.edu.br](mailto:edson.maciel.017@ufrn.edu.br)  
        -  [LinkedIn](https://linkedin.com/in/edsonmaciel017) | [GitHub](https://github.com/DevMacielJr)

        **Objetivo deste projeto:**  
        Criar uma ferramenta acessível para análise e previsão de doenças cardíacas, facilitando a compreensão de dados clínicos por profissionais e entusiastas.
        """)

    st.markdown("---")
    st.caption("© 2025 Edson Maciel. Todos os direitos reservados.")
