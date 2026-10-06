import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title='TV Sales Dashboard', page_icon='📺', layout='wide')

# ===== السطور المطلوبة في التاسك =====
st.header('Pair programming')
st.subheader('Q1. Vibe Coding Practice with TV_Sales.csv')
st.markdown('#### Dashboard Goal: Help a TV retail manager understand what drives sales (price, competitor price, advertising, shelf location, customer profile), compare markets, and simulate "what-if" decisions. ####')

PATH = 'C:/Users/user/streamlit/Data/TVsales/TV_Sales.csv'
COLORS = ['#22d3ee', '#a78bfa', '#f472b6', '#fbbf24', '#34d399']
SHELF_COLORS = {'Bad': '#f87171', 'Medium': '#fbbf24', 'Good': '#34d399'}
SHELF_ORDER = ['Bad', 'Medium', 'Good']
NUM = ['CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age', 'Education']

px.defaults.template = 'plotly_dark'
px.defaults.color_discrete_sequence = COLORS

# ===== Style =====
st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 8% 0%,#1d2150 0%,#0b0d17 55%);}
section[data-testid="stSidebar"]{background:#0f1224;border-right:1px solid rgba(255,255,255,.08);}
[data-testid="stMetric"]{background:linear-gradient(135deg,rgba(34,211,238,.10),rgba(167,139,250,.10));
  border:1px solid rgba(255,255,255,.12);border-radius:18px;padding:16px 18px;}
[data-testid="stMetricValue"]{font-size:1.9rem;font-weight:800;}
.hero{padding:26px 30px;border-radius:22px;margin:8px 0 18px 0;
  background:linear-gradient(120deg,#22d3ee33,#a78bfa33,#f472b633);border:1px solid rgba(255,255,255,.15);}
.hero h1{margin:0;font-size:2.3rem;background:linear-gradient(90deg,#22d3ee,#a78bfa,#f472b6);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero p{margin:6px 0 0 0;opacity:.8;}
.card{padding:16px 18px;border-radius:16px;background:rgba(255,255,255,.05);
  border:1px solid rgba(255,255,255,.1);height:100%;}
.card b{color:#22d3ee;}
.stTabs [data-baseweb="tab"]{font-size:1.02rem;padding:10px 16px;}
</style>
""", unsafe_allow_html=True)


# ===== Data =====
@st.cache_data
def load(src):
    df = pd.read_csv(src)
    df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')])
    df['PriceGap'] = df['Price'] - df['CompPrice']
    df['AgeGroup'] = pd.cut(df['Age'], [24, 35, 45, 55, 65, 80],
                            labels=['25-35', '36-45', '46-55', '56-65', '66-80'])
    df['IncomeGroup'] = pd.cut(df['Income'], [0, 40, 70, 100, 130],
                               labels=['Low <40', 'Mid 40-70', 'Upper 70-100', 'High 100+'])
    df['PriceBand'] = pd.cut(df['Price'], [0, 90, 110, 130, 150, 200],
                             labels=['<90', '90-110', '110-130', '130-150', '150+'])
    df['AdLevel'] = pd.cut(df['Advertising'], [-1, 0, 5, 10, 15, 30],
                           labels=['None', '1-5', '6-10', '11-15', '16+'])
    df['SalesTier'] = pd.qcut(df['Sales'], 3, labels=['Low', 'Medium', 'High'])
    df['Market'] = np.where(df['US'] == 'Yes', 'US', 'Non-US')
    df['Area'] = np.where(df['Urban'] == 'Yes', 'Urban', 'Rural')
    return df


def design(d):
    X = d[NUM].astype(float).copy()
    X['Shelf_Good'] = (d['ShelveLoc'] == 'Good').astype(float)
    X['Shelf_Medium'] = (d['ShelveLoc'] == 'Medium').astype(float)
    X['Urban'] = (d['Urban'] == 'Yes').astype(float)
    X['US'] = (d['US'] == 'Yes').astype(float)
    return X


@st.cache_data
def fit_model(d):
    X = design(d)
    y = d['Sales'].values
    A = np.c_[np.ones(len(X)), X.values]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ beta
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    std_coef = beta[1:] * X.std().values / y.std()
    return beta, list(X.columns), r2, std_coef


def style(fig, h=380):
    fig.update_layout(height=h, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=10, r=10, t=50, b=10), font=dict(size=13),
                      legend=dict(orientation='h', y=-0.18))
    return fig


def show(fig, h=380):
    try:
        st.plotly_chart(style(fig, h), width='stretch')
    except TypeError:  # older Streamlit versions
        st.plotly_chart(style(fig, h), use_container_width=True)


def table(d, **kw):
    try:
        st.dataframe(d, width='stretch', **kw)
    except TypeError:
        st.dataframe(d, use_container_width=True, **kw)


def add_trend(fig, x, y, name='Trend'):
    if len(x) > 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode='lines', name=name,
                                 line=dict(color='white', dash='dash', width=2)))


# ===== Source =====
src = None
if Path(PATH).exists():
    src = PATH
elif Path('TV_Sales.csv').exists():
    src = 'TV_Sales.csv'
up = st.sidebar.file_uploader('📂 Upload TV_Sales.csv (optional)', type='csv')
if up is not None:
    src = up
if src is None:
    st.warning('Upload TV_Sales.csv from the sidebar.')
    st.stop()

full = load(src)

# ===== Sidebar filters =====
st.sidebar.title('🎛️ Filters')
shelf = st.sidebar.multiselect('Shelf location', SHELF_ORDER, default=SHELF_ORDER)
market = st.sidebar.multiselect('Market', ['US', 'Non-US'], default=['US', 'Non-US'])
area = st.sidebar.multiselect('Area', ['Urban', 'Rural'], default=['Urban', 'Rural'])


def rng(label, col):
    lo, hi = int(full[col].min()), int(full[col].max())
    return st.sidebar.slider(label, lo, hi, (lo, hi))


price_r = rng('Price ($)', 'Price')
age_r = rng('Customer age', 'Age')
inc_r = rng('Income (K$)', 'Income')
ad_r = rng('Advertising budget', 'Advertising')

df = full[
    full['ShelveLoc'].isin(shelf) & full['Market'].isin(market) & full['Area'].isin(area)
    & full['Price'].between(*price_r) & full['Age'].between(*age_r)
    & full['Income'].between(*inc_r) & full['Advertising'].between(*ad_r)
].copy()

st.sidebar.caption(f'Showing **{len(df)}** of {len(full)} stores')
if df.empty:
    st.warning('No data matches the filters. Loosen them from the sidebar.')
    st.stop()

# ===== Hero + KPIs =====
st.markdown("""
<div class="hero"><h1>📺 TV Sales Intelligence Dashboard</h1>
<p>What drives TV sales across 400 store locations? Explore price, competition, advertising, shelf placement and customers.</p></div>
""", unsafe_allow_html=True)

best_shelf = df.groupby('ShelveLoc', observed=True)['Sales'].mean().idxmax()
k = st.columns(6)
k[0].metric('🏬 Stores', f'{len(df):,}')
k[1].metric('📦 Total Sales', f'{df.Sales.sum():,.0f}K')
k[2].metric('📊 Avg Sales', f'{df.Sales.mean():.2f}K',
            f'{df.Sales.mean() - full.Sales.mean():+.2f} vs all')
k[3].metric('💲 Avg Price', f'${df.Price.mean():.0f}', f'{df.PriceGap.mean():+.1f} vs competitor')
k[4].metric('📣 Avg Advertising', f'{df.Advertising.mean():.1f}')
k[5].metric('🏆 Best Shelf', best_shelf)

# ===== Auto insights =====
st.subheader('💡 Auto Insights')
c_price = df['Price'].corr(df['Sales'])
c_ad = df['Advertising'].corr(df['Sales'])
sh = df.groupby('ShelveLoc', observed=True)['Sales'].mean()
mk = df.groupby('Market')['Sales'].mean()
ag = df.groupby('AgeGroup', observed=True)['Sales'].mean()
noad = df[df.Advertising == 0].Sales.mean()
hiad = df[df.Advertising > 10].Sales.mean()


def card(title, body):
    return f'<div class="card"><b>{title}</b><br>{body}</div>'


i = st.columns(4)
i[0].markdown(card('Shelf matters',
    f'"{sh.idxmax()}" shelf avg <b>{sh.max():.1f}K</b> vs "{sh.idxmin()}" <b>{sh.min():.1f}K</b>.'),
    unsafe_allow_html=True)
i[1].markdown(card('Price sensitivity',
    f'Price vs Sales correlation = <b>{c_price:.2f}</b> ' + ('(higher price → lower sales).' if c_price < 0 else '.')),
    unsafe_allow_html=True)
ad_txt = (f'Stores with ad budget &gt;10 sell <b>{hiad:.1f}K</b> vs <b>{noad:.1f}K</b> with none.'
          if not (np.isnan(noad) or np.isnan(hiad)) else f'Advertising correlation = <b>{c_ad:.2f}</b>.')
i[2].markdown(card('Advertising', ad_txt), unsafe_allow_html=True)
i[3].markdown(card('Best customer age',
    f'Age group <b>{ag.idxmax()}</b> has the highest avg sales (<b>{ag.max():.1f}K</b>).'),
    unsafe_allow_html=True)

st.write('')

# ===== Tabs =====
t1, t2, t3, t4, t5, t6 = st.tabs(['🏠 Overview', '💰 Price & Competition', '📣 Advertising',
                                  '👥 Customers', '🧠 Drivers & What-If', '🔎 Data'])

# ---- Overview
with t1:
    a, b = st.columns(2)
    with a:
        f = px.histogram(df, x='Sales', nbins=30, marginal='box', color='SalesTier',
                         category_orders={'SalesTier': ['Low', 'Medium', 'High']},
                         color_discrete_map={'Low': '#f87171', 'Medium': '#fbbf24', 'High': '#34d399'},
                         title='Sales distribution (thousand units)')
        show(f, 420)
    with b:
        f = px.box(df, x='ShelveLoc', y='Sales', color='ShelveLoc', points='all',
                   category_orders={'ShelveLoc': SHELF_ORDER}, color_discrete_map=SHELF_COLORS,
                   title='Sales by shelf location')
        f.update_layout(showlegend=False)
        show(f, 420)
    a, b = st.columns(2)
    with a:
        f = px.sunburst(df, path=['Market', 'Area', 'ShelveLoc'], values='Sales',
                        title='Where do sales come from? Market → Area → Shelf')
        show(f, 450)
    with b:
        g = df.groupby(['Market', 'Area'], as_index=False).agg(AvgSales=('Sales', 'mean'), Stores=('Sales', 'size'))
        g['Segment'] = g['Market'] + ' / ' + g['Area']
        f = px.bar(g, x='Segment', y='AvgSales', color='Market', text=g['AvgSales'].round(2),
                   title='Average sales per segment')
        f.update_traces(textposition='outside')
        show(f, 450)

# ---- Price
with t2:
    a, b = st.columns(2)
    with a:
        f = px.scatter(df, x='Price', y='Sales', color='ShelveLoc', size='Advertising', opacity=.75,
                       category_orders={'ShelveLoc': SHELF_ORDER}, color_discrete_map=SHELF_COLORS,
                       title='Price vs Sales (bubble size = advertising)',
                       hover_data=['CompPrice', 'Income', 'Age'])
        add_trend(f, df['Price'], df['Sales'])
        show(f, 440)
    with b:
        f = px.scatter(df, x='PriceGap', y='Sales', color='Market', opacity=.75,
                       title='Price gap vs competitor (Price − CompPrice)')
        add_trend(f, df['PriceGap'], df['Sales'])
        f.add_vline(x=0, line_dash='dot', line_color='gray')
        show(f, 440)
    a, b = st.columns(2)
    with a:
        g = df.groupby('PriceBand', observed=True)['Sales'].mean().reset_index()
        f = px.bar(g, x='PriceBand', y='Sales', color='Sales', color_continuous_scale='Tealgrn',
                   text=g['Sales'].round(2), title='Avg sales by price band')
        f.update_traces(textposition='outside')
        show(f)
    with b:
        p = df.pivot_table(index='PriceBand', columns='ShelveLoc', values='Sales',
                           aggfunc='mean', observed=True)
        p = p.reindex(columns=[c for c in SHELF_ORDER if c in p.columns])
        f = px.imshow(p, text_auto='.1f', color_continuous_scale='Viridis', aspect='auto',
                      title='Avg sales: Price band × Shelf location')
        show(f)

# ---- Advertising
with t3:
    a, b = st.columns(2)
    with a:
        g = df.groupby('AdLevel', observed=True)['Sales'].mean().reset_index()
        f = px.bar(g, x='AdLevel', y='Sales', color='Sales', color_continuous_scale='Sunsetdark',
                   text=g['Sales'].round(2), title='Avg sales by advertising level')
        f.update_traces(textposition='outside')
        show(f, 420)
    with b:
        f = px.scatter(df, x='Advertising', y='Sales', color='Market', size='Population', opacity=.7,
                       title='Advertising vs Sales (bubble = population)')
        add_trend(f, df['Advertising'], df['Sales'])
        show(f, 420)
    g = df.groupby(['AdLevel', 'Market'], observed=True)['Sales'].mean().reset_index()
    f = px.bar(g, x='AdLevel', y='Sales', color='Market', barmode='group',
               title='Does advertising pay off equally in US and Non-US?')
    show(f, 380)

# ---- Customers
with t4:
    a, b = st.columns(2)
    with a:
        g = df.groupby('AgeGroup', observed=True)['Sales'].mean().reset_index()
        f = px.bar(g, x='AgeGroup', y='Sales', color='Sales', color_continuous_scale='Purp',
                   text=g['Sales'].round(2), title='Avg sales by customer age group')
        f.update_traces(textposition='outside')
        show(f)
    with b:
        p = df.pivot_table(index='IncomeGroup', columns='AgeGroup', values='Sales',
                           aggfunc='mean', observed=True)
        f = px.imshow(p, text_auto='.1f', color_continuous_scale='Plasma', aspect='auto',
                      title='Avg sales: Income group × Age group')
        show(f)
    a, b = st.columns(2)
    with a:
        f = px.scatter(df, x='Income', y='Sales', color='ShelveLoc', opacity=.7,
                       category_orders={'ShelveLoc': SHELF_ORDER}, color_discrete_map=SHELF_COLORS,
                       title='Income vs Sales')
        add_trend(f, df['Income'], df['Sales'])
        show(f)
    with b:
        f = px.violin(df, x='Education', y='Sales', color='Area', box=True,
                      title='Sales by education level & area')
        show(f)

# ---- Drivers & What-if
with t5:
    beta, cols, r2, std_coef = fit_model(full)
    a, b = st.columns(2)
    with a:
        corr = full[['Sales'] + NUM].corr()
        f = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                      title='Correlation matrix')
        show(f, 480)
    with b:
        d = pd.DataFrame({'Feature': cols, 'Impact': std_coef}).sort_values('Impact')
        f = px.bar(d, x='Impact', y='Feature', orientation='h', color='Impact',
                   color_continuous_scale='RdYlGn', title=f'Key drivers of sales (linear model, R² = {r2:.2f})')
        show(f, 480)
    st.caption('Impact = standardized regression coefficient: bigger bar = stronger effect on Sales, holding the rest constant.')

    st.subheader('🎮 What-If Simulator')
    s = st.columns(4)
    p_price = s[0].slider('Our price ($)', 24, 191, int(full.Price.median()))
    p_comp = s[1].slider('Competitor price ($)', 77, 175, int(full.CompPrice.median()))
    p_ad = s[2].slider('Advertising', 0, 29, int(full.Advertising.median()))
    p_shelf = s[3].selectbox('Shelf location', SHELF_ORDER, index=1)
    s = st.columns(4)
    p_inc = s[0].slider('Income (K$)', 21, 120, int(full.Income.median()))
    p_age = s[1].slider('Age', 25, 80, int(full.Age.median()))
    p_pop = s[2].slider('Population (K)', 10, 509, int(full.Population.median()))
    p_us = s[3].selectbox('Market', ['US', 'Non-US'])
    row = pd.DataFrame([{'CompPrice': p_comp, 'Income': p_inc, 'Advertising': p_ad, 'Population': p_pop,
                         'Price': p_price, 'Age': p_age, 'Education': int(full.Education.median()),
                         'ShelveLoc': p_shelf, 'Urban': 'Yes', 'US': 'Yes' if p_us == 'US' else 'No'}])
    pred = max(0.0, float(beta[0] + design(row).values[0] @ beta[1:]))
    a, b = st.columns([1, 1])
    with a:
        g = go.Figure(go.Indicator(
            mode='gauge+number+delta', value=pred, number={'suffix': 'K units'},
            delta={'reference': full.Sales.mean(), 'valueformat': '.2f'},
            title={'text': 'Predicted sales'},
            gauge={'axis': {'range': [0, 16]}, 'bar': {'color': '#22d3ee'},
                   'steps': [{'range': [0, 5], 'color': '#7f1d1d'},
                             {'range': [5, 9], 'color': '#78350f'},
                             {'range': [9, 16], 'color': '#14532d'}],
                   'threshold': {'line': {'color': 'white', 'width': 3}, 'value': full.Sales.mean()}}))
        show(g, 340)
    with b:
        st.markdown('**Rule-of-thumb effects (model)**')
        st.write(f'• +$10 price → **{beta[1 + cols.index("Price")] * 10:+.2f}K** units')
        st.write(f'• +5 advertising → **{beta[1 + cols.index("Advertising")] * 5:+.2f}K** units')
        st.write(f'• Good vs Bad shelf → **{beta[1 + cols.index("Shelf_Good")]:+.2f}K** units')
        st.write(f'• +$10 competitor price → **{beta[1 + cols.index("CompPrice")] * 10:+.2f}K** units')
        st.info('Simple linear model for exploration, not a production forecast.')

    st.subheader('🧵 Multi-factor view')
    f = px.parallel_coordinates(df, dimensions=['Price', 'CompPrice', 'Advertising', 'Income', 'Age', 'Sales'],
                                color='Sales', color_continuous_scale='Turbo')
    show(f, 420)

# ---- Data
with t6:
    table(df.drop(columns=['Market', 'Area']), height=420)
    st.download_button('⬇️ Download filtered data (CSV)', df.to_csv(index=False).encode('utf-8'),
                       'tv_sales_filtered.csv', 'text/csv')
    with st.expander('Summary statistics'):
        table(df[['Sales'] + NUM].describe().T)
