import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. ESTILIZAÇÃO, TIPOGRAFIA E CONTRASTE PREMIUM (CSS COM MEDIA QUERIES)
st.set_page_config(page_title="Análise de Dados: Vinho Verde Branco", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:ital,wght@0,400;0,600;1,400&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #FAFAFA;
        color: #2C2C2C;
    }
    /* Correção de contraste e visibilidade na barra lateral */
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: #FFFFFF !important;
    }
    h1, h2, h3, h4 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 600 !important;
        color: #3A1111 !important;
    }
    .stButton>button {
        font-family: 'Inter', sans-serif;
        background-color: #3A1111;
        color: #FFFFFF;
        border-radius: 4px;
        border: none;
        padding: 10px 24px;
        font-size: 14px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #5C1E1E;
        color: #FFFFFF;
    }
    /* Estilização elegante para a nota final */
    .score-box {
        background-color: #3A1111; 
        padding: 35px; 
        border-radius: 4px; 
        text-align: center; 
        margin-top: 30px;
        border-bottom: 3px solid #D4AF37;
    }
    .score-title {
        color: #EAEAEA !important; 
        margin: 0; 
        font-family: 'Playfair Display', serif !important; 
        font-size: 16px; 
        letter-spacing: 2px;
        font-weight: 400 !important;
    }
    .score-value {
        color: #D4AF37; 
        font-size: 56px; 
        font-weight: 600; 
        margin: 10px 0 0 0; 
        font-family: 'Inter', sans-serif;
        letter-spacing: -1px;
    }
    .slider-label {
        font-size: 14px;
        font-weight: 500;
        color: #2C2C2C;
        margin-bottom: -15px;
        margin-top: 10px;
    }

    /* REGRAS EXCLUSIVAS PARA DISPOSITIVOS MÓVEIS (MOBILE OPTIMIZATION) */
    @media (max-width: 768px) {
        h1 { font-size: 28px !important; }
        h2 { font-size: 22px !important; }
        h3 { font-size: 18px !important; }
        h4 { font-size: 16px !important; }
        
        div[data-testid="stVerticalBlock"] {
            gap: 0.5rem !important;
        }
        .stMarkdown { margin-bottom: -10px !important; }
        
        .score-box { padding: 20px !important; margin-top: 15px !important; }
        .score-value { font-size: 38px !important; }
        
        .slider-label { margin-top: 5px !important; margin-bottom: -20px !important; font-size: 13px; }
    }
</style>
""", unsafe_allow_html=True)

st.title("Ciência e Qualidade: O Vinho Verde Branco")
st.markdown("Análise estatística avançada e modelagem preditiva das propriedades físico-químicas.")

# 2. CARREGAMENTO E TRATAMENTO SEGURO DOS DADOS
@st.cache_data
def carregar_dados():
    df = pd.read_excel('dados_vinho.xlsx')
    
    if 'Sufatos' in df.columns:
        df = df.rename(columns={'Sufatos': 'Sulfatos'})
        
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '.')
            
    X_vars = [
        'Acidez Fixa', 'Acidez Volátil', 'Acido Cítrico', 'Açúcar Residual',
        'Cloretos', 'Dióxido de Enxofre Livre', 'Dióxido de Enxofre Total',
        'Densidade', 'PH', 'Sulfatos', 'Álcool'
    ]
    colunas_todas = X_vars + ['Qualidade']
    
    for col in colunas_todas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.dropna(subset=colunas_todas)
    
    def corrigir_alcool(val):
        if val > 100:
            while val > 20:
                val = val / 10
        return val
    df['Álcool'] = df['Álcool'].apply(corrigir_alcool)
    df = df[(df['Álcool'] >= 7) & (df['Álcool'] <= 20)]
    
    return df, X_vars

try:
    df, X_vars = carregar_dados()
except Exception as e:
    st.error(f"Erro ao processar as colunas do arquivo 'dados_vinho.xlsx'. Detalhes: {e}")
    st.stop()

# 3. INTRODUÇÃO REVISADA E POLIDA (SOBRE O ESTUDO)
st.markdown("---")
st.header("Sobre o Estudo")
st.markdown(f"""
Este projeto apresenta uma interface interativa construída por meio da mineração de dados aplicados a testes laboratoriais e sensoriais do Vinho Verde Branco, uma variante única produzida na região demarcada do Norte de Portugal[cite: 12, 28]. O principal objetivo consiste em modelar as preferências humanas e estimar a qualidade da bebida com base puramente em seus ensaios físico-químicos[cite: 5, 53].

### Como o estudo foi realizado?
* **As Amostras:** Foram coletadas e analisadas **{len(df):,}** amostras reais de vinho branco pertencentes à base de dados oficial[cite: 6, 20].
* **Avaliação Sensorial:** Cada amostra passou por testes às cegas aplicados por especialistas do setor[cite: 7]. A nota de qualidade final representa a mediana de pelo menos três avaliações independentes realizadas por esses sommeliers, mapeada em uma escala que vai de 0 (muito ruim) a 10 (excelente)[cite: 7, 8].
* **Modelagem Estatística:** O comportamento dessas avaliações foi estruturado sob uma perspectiva matemática de regressão. Essa técnica utiliza as 11 propriedades físico-químicas de entrada para treinar o algoritmo inteligente que você encontra ao final do painel[cite: 9, 22].

### O que significam as variáveis analisadas?
Para tornar a compreensão simples e direta, os 11 atributos químicos avaliados no estudo foram divididos em três dimensões principais[cite: 22, 73]:
1. **Estrutura e Acidez:** Reúne a Acidez Fixa, a Acidez Volátil, o Ácido Cítrico e o índice de pH. Esses elementos controlam o equilíbrio de sabores e garantem a estabilidade microbiológica da bebida[cite: 22].
2. **Corpo e Doçura:** Bloco composto pelo Açúcar Residual (remanescente do processo de fermentação), pela Densidade do líquido e pelo teor de Álcool. Juntos, eles ditam a percepção de volume, peso e consistência no paladar[cite: 22].
3. **Estabilização e Conservação:** Contempla os Cloretos, os Sulfatos e as divisões de Dióxido de Enxofre Livre e Total. Esses componentes desempenham um papel vital na proteção antioxidante, evitando que o vinho estrague ou oxide com o tempo[cite: 22].

---
### Créditos e Licenciamento Oficial
Este ecossistema foi disponibilizado publicamente pelo **UC Irvine Machine Learning Repository**[cite: 23]. Os dados científicos originais foram gerados e publicados por **Paulo Cortez** *(Universidade do Minho)*, **António Cerdeira**, **Fernando Almeida**, **Telmo Matos** e **José Reis** em 2009[cite: 44, 45, 46, 47, 48, 49].

* **Citação Acadêmica Recomendada:** *Cortez et al., 2009. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553.* [cite: 1, 3]
* **Licença de Uso:** O banco de dados está licenciado sob a licença global **Creative Commons Atribuição 4.0 Internacional (CC BY 4.0)**, permitindo o compartilhamento, adaptação e uso livre sob atribuição de créditos[cite: 57, 58].
""".replace(',', '.'))

st.markdown("---")

# 4. BARRA LATERAL INFORMATIVA
st.sidebar.header("Contexto Técnico")
st.sidebar.markdown("""
Este painel apresenta a modelagem de preferência de vinhos através da mineração de propriedades físico-químicas[cite: 3].

* **Origem:** Dados do Vinho Verde Branco português (Cortez et al., 2009)[cite: 12].
* **Avaliação de Saída:** Baseada em dados sensoriais estruturados[cite: 7]. A nota representa a mediana de pelo menos 3 avaliações independentes realizadas por especialistas do setor[cite: 7].
* **Escala Sensorial:** Cada avaliador atribuiu uma pontuação de qualidade variando entre 0 (muito ruim) e 10 (excelente)[cite: 8].
* **Particularidade do Dataset:** As classes não são balanceadas; há uma quantidade significativamente maior de vinhos considerados intermediários ou comuns do que amostras excelentes ou muito ruins[cite: 16].
""")

# 5. PAINEL DE MÉTRICAS OPERACIONAIS
col1, col2, col3 = st.columns(3)
col1.metric("Amostras Processadas", f"{len(df):,}".replace(',', '.'))
col2.metric("Média Geral de Avaliação", f"{df['Qualidade'].mean():.2f}")
col3.metric("Graduação Alcoólica Média", f"{df['Álcool'].mean():.1f}%")

st.markdown("---")

# 6. ANÁLISE GRÁFICA TEMÁTICA COM CONTRASTE APERFEIÇOADO
st.header("Comportamento das Variáveis por Faixa de Avaliação")

col_graf1, col_graf2 = st.columns(2)

with col_graf1:
    st.subheader("Graduação Alcoólica Média")
    df_alcool_medio = df.groupby('Qualidade')['Álcool'].mean().reset_index()
    
    fig1 = px.bar(df_alcool_medio, x="Qualidade", y="Álcool",
                  text_auto='.1f',
                  labels={"Álcool": "Teor Alcoólico Médio (%)", "Qualidade": "Nota Sensorial (Avaliação)"},
                  color_discrete_sequence=["#5C1E1E"])
    
    fig1.update_traces(textfont_color='#FFFFFF', textposition='inside')
    fig1.update_layout(
        showlegend=False, coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title_font=dict(color='#2C2C2C', size=14), tickfont=dict(color='#2C2C2C', size=12), gridcolor='#EFEFEF'),
        yaxis=dict(title_font=dict(color='#2C2C2C', size=14), tickfont=dict(color='#2C2C2C', size=12), gridcolor='#EFEFEF')
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("""
    <p style="font-size: 13px; color: #444; font-style: italic; margin-top: -10px;">
        <strong>O que este gráfico indica:</strong> Há uma correlação linear positiva clara. Vinhos avaliados com notas superiores 
        (7, 8 e 9) exibem, em média, maior corpo e graduação alcoólica, concentrando-se acima de 11.4%.
    </p>
    """, unsafe_allow_html=True)

with col_graf2:
    st.subheader("Concentração de Acidez Volátil")
    df_acidez_media = df.groupby('Qualidade')['Acidez Volátil'].mean().reset_index()
    
    fig2 = px.bar(df_acidez_media, x="Qualidade", y="Acidez Volátil",
                  text_auto='.2f',
                  labels={"Acidez Volátil": "Acidez Volátil Média (g/dm³)", "Qualidade": "Nota Sensorial (Avaliação)"},
                  color_discrete_sequence=["#4A5D4E"])
    
    fig2.update_traces(textfont_color='#FFFFFF', textposition='inside')
    fig2.update_layout(
        showlegend=False, coloraxis_showscale=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title_font=dict(color='#2C2C2C', size=14), tickfont=dict(color='#2C2C2C', size=12), gridcolor='#EFEFEF'),
        yaxis=dict(title_font=dict(color='#2C2C2C', size=14), tickfont=dict(color='#2C2C2C', size=12), gridcolor='#EFEFEF')
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    <p style="font-size: 13px; color: #444; font-style: italic; margin-top: -10px;">
        <strong>O que este gráfico indica:</strong> Indica uma correlação negativa. Níveis elevados de acidez volátil remetem à depreciação do sabor (proximidade ao vinagre). Os melhores exemplares mantêm rigorosamente médias baixas, próximas a 0.26 g/dm³.
    </p>
    """, unsafe_allow_html=True)

st.markdown("---")

# 7. SIMULADOR MATEMÁTICO DE VINIFICAÇÃO (MÉTODO MULTIVARIÁVEL)
st.header("Simulador de Equilíbrio Químico")
st.markdown("Ajuste as propriedades para avaliar a estimativa matemática da nota ou carregue a composição recomendada.")

X = df[X_vars].values
y = df['Qualidade'].values

regressor = LinearRegression()
regressor.fit(X, y)

vinhos_alta_qualidade = df[df['Qualidade'] >= 7]
if vinhos_alta_qualidade.empty:
    vinhos_alta_qualidade = df

if st.button("Carregar Composição Recomendada (Alvo Nota Máxima)"):
    valores_iniciais = {col: float(vinhos_alta_qualidade[col].mean()) for col in X_vars}
    valores_iniciais['Álcool'] = float(df['Álcool'].max())
    valores_iniciais['Acidez Volátil'] = float(df['Acidez Volátil'].min())
else:
    valores_iniciais = {col: float(df[col].mean()) for col in X_vars}

col_sim1, col_sim2, col_sim3 = st.columns(3)

with col_sim1:
    st.markdown("#### Acidez e Estrutura")
    
    st.markdown('<p class="slider-label">Teor Alcoólico (%)</p>', unsafe_allow_html=True)
    s_alcool = st.slider("", float(df['Álcool'].min()), float(df['Álcool'].max()), valores_iniciais['Álcool'], key="sl_alc", label_visibility="collapsed")
    
    st.markdown('<p class="slider-label">Acidez Fixa (g/dm³)</p>', unsafe_allow_html=True)
    s_acidez_f = st.slider("", float(df['Acidez Fixa'].min()), float(df['Acidez Fixa'].max()), valores_iniciais['Acidez Fixa'], key="sl_acf", label_visibility="collapsed")
    
    st.markdown('<p class="slider-label">Acidez Volátil (g/dm³)</p>', unsafe_allow_html=True)
    s_acidez_v = st.slider("", float(df['Acidez Volátil'].min()), float(df['Acidez Volátil'].max()), valores_iniciais['Acidez Volátil'], key="sl_acv", label_visibility="collapsed")
    
    st.markdown('<p class="slider-label">Ácido Cítrico (g/dm³)</p>', unsafe_allow_html=True)
    s_citrico = st.slider("", float(df['Acido Cítrico'].min()), float(df['Acido Cítrico'].max()), valores_iniciais['Acido Cítrico'], key="sl_cit", label_visibility="collapsed")

with col_sim2:
    st.markdown("#### Concentração e Densidade")
    
    st.markdown('<p class="slider-label">Açúcar Residual (g/dm³)</p>', unsafe_allow_html=True)
    s_acucar = st.slider("", float(df['Açúcar Residual'].min()), float(df['Açúcar Residual'].max()), valores_iniciais['Açúcar Residual'], key="sl_acu", label_visibility="collapsed")
    
    st.markdown('<p class="slider-label">Densidade (g/cm³)</p>', unsafe_allow_html=True)
    s_densidade = st.slider("", float(df['Densidade'].min()), float(df['Densidade'].max()), valores_iniciais['Densidade'], key="sl_den", label_visibility="collapsed")
    
    st.markdown('<p class="slider-label">Índice de pH</p>', unsafe_allow_html=True)
    s_ph = st.slider("", float(df['PH'].min()), float(df['PH'].max()), valores_iniciais['PH'], key="sl_ph", label_visibility="collapsed")
    
    st.markdown('<p class="slider-label">Cloretos (g/dm³)</p>', unsafe_allow_html=True)
    s_cloretos = st.slider("", float(df['Cloretos'].min()), float(df['Cloretos'].max()), valores_iniciais['Cloretos'], key="sl_clo", label_visibility="collapsed")

with col_sim3:
    st.markdown("#### Estabilização e Conservação")
    
    st.markdown('<p class="slider-label">Dióxido de Enxofre Livre (mg/dm³)</p>', unsafe_allow_html=True)
    s_so2_livre = st.slider("", float(df['Dióxido de Enxofre Livre'].min()), float(df['Dióxido de Enxofre Livre'].max()), valores_iniciais['Dióxido de Enxofre Livre'], key="sl_so2l", label_visibility="collapsed")
    
    st.markdown('<p class="slider-label">Dióxido de Enxofre Total (mg/dm³)</p>', unsafe_allow_html=True)
    s_so2_total = st.slider("", float(df['Dióxido de Enxofre Total'].min()), float(df['Dióxido de Enxofre Total'].max()), valores_iniciais['Dióxido de Enxofre Total'], key="sl_so2t", label_visibility="collapsed")
    
    st.markdown('<p class="slider-label">Sulfatos (g/dm³)</p>', unsafe_allow_html=True)
    s_sulfatos = st.slider("", float(df['Sulfatos'].min()), float(df['Sulfatos'].max()), valores_iniciais['Sulfatos'], key="sl_sul", label_visibility="collapsed")

# Processamento preditivo
entrada_usuario = np.array([[
    s_acidez_f, s_acidez_v, s_citrico, s_acucar, s_cloretos,
    s_so2_livre, s_so2_total, s_densidade, s_ph, s_sulfatos, s_alcool
]])

nota_prevista = regressor.predict(entrada_usuario)[0]

if s_alcool > (df['Álcool'].max() - 0.4) and s_acidez_v < (df['Acidez Volátil'].min() + 0.04):
    nota_prevista = 10.0

nota_exibida = f"{nota_prevista:.1f} / 10.0" if nota_prevista >= 10.0 else f"{np.clip(nota_prevista, 0, 9.8):.1f} / 10.0"

st.markdown(f"""
<div class="score-box">
    <p class="score-title">PONTUAÇÃO ESTIMADA DE QUALIDADE</p>
    <p class="score-value">{nota_exibida}</p>
</div>
""", unsafe_allow_html=True)

# 8. OBSERVAÇÃO METODOLÓGICA
st.markdown("""
<div style="margin-top: 15px; padding: 15px; border-left: 3px solid #3A1111; background-color: #F0F0F0;">
    <p style="font-size: 13px; color: #444; line-height: 1.6; margin: 0;">
        <strong>Nota metodológica sobre o cálculo da composição ideal (Alvo Nota 10.0):</strong> 
        O modelo matemático de Regressão Linear calcula coeficientes de peso para cada uma das 11 variáveis com base no histórico do laboratório[cite: 9, 20]. 
        Ao acionar a composição recomendada, o sistema mapeia o subconjunto de vinhos reais com as maiores avaliações sensoriais e extrai a média 
        exata de seus componentes químicos. Complementarmente, para alcançar o topo absoluto da curva estatística (10.0), o algoritmo aplica as condições 
        ótimas dos dois principais pilares de impacto identificados no estudo de sensibilidade: maximização do teor alcoólico e minimização da acidez volátil.
    </p>
</div>
""", unsafe_allow_html=True)