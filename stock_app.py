import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title="Stock Analysis Tool", page_icon="📈", layout="wide")

st.title("Stock Movement Predictor")
st.write("""
A machine learning tool that analyzes historical stock data to identify 
potential big moves. Uses technical indicators + ensemble models.

*Not financial advice. Educational project only.*
""")


# --- sidebar inputs ---
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Ticker", value="AAPL").upper()
start_date = st.sidebar.date_input("From", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("To", value=pd.to_datetime("2024-01-01"))
threshold = st.sidebar.slider("Confidence threshold", 0.3, 0.7, 0.5, 0.05)
move_pct = st.sidebar.slider("Big move threshold (%)", 1.0, 10.0, 2.0, 0.5)
run = st.sidebar.button("Run Analysis", type="primary", use_container_width=True)


def rsi(series, period=14):
    """standard RSI calculation"""
    d = series.diff()
    up = d.where(d > 0, 0).rolling(period).mean()
    down = (-d.where(d < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + up / (down + 1e-10)))


def add_features(df):
    """
    builds all the technical indicators we need.
    tried a bunch of different combos — these worked best in testing.
    """
    # returns over different periods
    for n in [1, 2, 3, 5, 10]:
        df[f'ret_{n}d'] = df['Close'].pct_change(n)

    # moving averages
    for n in [5, 10, 20, 50]:
        df[f'ma_{n}'] = df['Close'].rolling(n).mean()
        df[f'above_ma{n}'] = (df['Close'] > df[f'ma_{n}']).astype(int)

    # ma crossovers
    df['ma5_x_ma20'] = (df['ma_5'] > df['ma_20']).astype(int)
    df['ma10_x_ma50'] = (df['ma_10'] > df['ma_50']).astype(int)

    # volatility
    df['vol_5'] = df['ret_1d'].rolling(5).std()
    df['vol_10'] = df['ret_1d'].rolling(10).std()
    df['vol_20'] = df['ret_1d'].rolling(20).std()
    df['vol_ratio'] = df['vol_5'] / (df['vol_20'] + 1e-10)

    # rsi
    df['rsi_14'] = rsi(df['Close'], 14)
    df['rsi_7'] = rsi(df['Close'], 7)

    # volume stuff
    df['vol_ma10'] = df['Volume'].rolling(10).mean()
    df['vol_spike'] = (df['Volume'] > df['vol_ma10'] * 1.5).astype(int)

    # misc
    df['weekday'] = df.index.dayofweek
    df['up_day'] = (df['ret_1d'] > 0).astype(int)
    df['consec_up'] = df['up_day'].rolling(5).sum()
    df['dist_ma20'] = (df['Close'] - df['ma_20']) / df['ma_20']

    return df


# features we actually feed into the model
# dropped a few that weren't helping (above_ma5, above_ma10 were useless)
FEATURES = [
    'ret_1d', 'ret_2d', 'ret_3d', 'ret_5d', 'ret_10d',
    'above_ma5', 'above_ma10', 'above_ma20', 'above_ma50',
    'ma5_x_ma20', 'ma10_x_ma50',
    'vol_5', 'vol_10', 'vol_ratio',
    'rsi_14', 'rsi_7',
    'vol_spike', 'weekday', 'consec_up', 'dist_ma20'
]


def train_models(X_tr, y_tr):
    """trains RF + GB and returns both"""
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=4,
        min_samples_split=30, min_samples_leaf=15,
        class_weight='balanced', random_state=42, n_jobs=-1
    )

    gb = GradientBoostingClassifier(
        n_estimators=300, max_depth=3,
        learning_rate=0.005, subsample=0.7,
        min_samples_split=20, min_samples_leaf=10,
        random_state=42
    )

    rf.fit(X_tr, y_tr)
    gb.fit(X_tr, y_tr)
    return rf, gb


def ensemble_predict(rf, gb, X):
    """average probabilities from both models"""
    p1 = rf.predict_proba(X)[:, 1]
    p2 = gb.predict_proba(X)[:, 1]
    return (p1 + p2) / 2


if run:
    # grab data
    with st.spinner(f"Pulling {ticker} data..."):
        raw = yf.download(ticker, start=start_date, end=end_date)
        if len(raw) == 0:
            st.error(f"Couldn't find data for {ticker}")
            st.stop()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

    st.success(f"Got {len(raw)} trading days for {ticker}")

    # quick overview
    st.header(f"{ticker} at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    last = raw['Close'].iloc[-1]
    change = (raw['Close'].iloc[-1] / raw['Close'].iloc[0] - 1) * 100
    c1.metric("Last Price", f"${last:.2f}")
    c2.metric("Total Return", f"{change:.1f}%")
    c3.metric("52w High", f"${raw['Close'].tail(252).max():.2f}")
    c4.metric("52w Low", f"${raw['Close'].tail(252).min():.2f}")

    # build features
    with st.spinner("Crunching numbers..."):
        df = add_features(raw.copy())

        cutoff = move_pct / 100
        df['fwd_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        df['big_up'] = (df['fwd_5d'] > cutoff).astype(int)
        df['big_down'] = (df['fwd_5d'] < -cutoff).astype(int)
        df['crash'] = (df['fwd_5d'] < -0.05).astype(int)
        df.dropna(inplace=True)

        # 70/15/15 split
        n = len(df)
        t1, t2 = int(n * 0.7), int(n * 0.85)

        X_tr = df[FEATURES][:t1]
        X_val = df[FEATURES][t1:t2]
        X_te = df[FEATURES][t2:]

        sc = StandardScaler()
        X_tr_s = pd.DataFrame(sc.fit_transform(X_tr), columns=FEATURES, index=X_tr.index)
        X_val_s = pd.DataFrame(sc.transform(X_val), columns=FEATURES, index=X_val.index)
        X_te_s = pd.DataFrame(sc.transform(X_te), columns=FEATURES, index=X_te.index)

        # train for each target
        models = {}
        for name, col in [('Big Up', 'big_up'), ('Big Down', 'big_down'), ('Crash', 'crash')]:
            y_tr = df[col][:t1]
            y_te = df[col][t2:]

            if y_tr.sum() < 10:
                models[name] = None
                continue

            rf, gb = train_models(X_tr_s, y_tr)
            proba = ensemble_predict(rf, gb, X_te_s)
            pred = (proba > threshold).astype(int)

            models[name] = {
                'y_test': y_te,
                'proba': proba,
                'pred': pred,
                'baseline': y_te.mean(),
                'rf': rf, 'gb': gb,
                'acc': accuracy_score(y_te, pred),
                'prec': precision_score(y_te, pred, zero_division=0),
                'n_trades': pred.sum()
            }

    # --- charts ---
    st.header("Charts")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=('Price', 'Crash Probability', 'RSI')
    )

    # price + moving averages
    fig.add_trace(go.Scatter(
        x=df.index[t2:], y=df['Close'][t2:],
        name='Price', line=dict(color='#2196F3', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index[t2:], y=df['ma_20'][t2:],
        name='MA20', line=dict(color='orange', width=1, dash='dash')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index[t2:], y=df['ma_50'][t2:],
        name='MA50', line=dict(color='red', width=1, dash='dash')
    ), row=1, col=1)

    # crash prob
    if models.get('Crash') is not None:
        cp = models['Crash']['proba']
        fig.add_trace(go.Scatter(
            x=df.index[t2:], y=cp, name='Crash prob',
            fill='tozeroy', line=dict(color='red', width=1),
            fillcolor='rgba(255,0,0,0.15)'
        ), row=2, col=1)
        fig.add_hline(y=0.3, line_dash="dash", line_color="orange", row=2, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=2, col=1)

    # rsi
    fig.add_trace(go.Scatter(
        x=df.index[t2:], y=df['rsi_14'][t2:],
        name='RSI', line=dict(color='purple', width=1)
    ), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

    fig.update_layout(height=800, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

    # --- results ---
    st.header("Model Performance")

    for name in ['Big Up', 'Big Down', 'Crash']:
        m = models.get(name)
        if m is None:
            st.warning(f"{name}: not enough samples to train")
            continue

        with st.expander(f"{name}", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Baseline", f"{m['baseline']*100:.1f}%")
            c2.metric("Precision", f"{m['prec']*100:.1f}%",
                      delta=f"{(m['prec']-m['baseline'])*100:+.1f}%")
            c3.metric("Trades", m['n_trades'])

            edge = m['prec'] - m['baseline']
            if edge > 0 and m['n_trades'] >= 5:
                c4.metric("Edge", f"+{edge*100:.1f}%")
            else:
                c4.metric("Edge", "None")

    # --- feature importance ---
    st.header("Feature Importance")

    if models.get('Big Down') is not None:
        imp = pd.Series(
            models['Big Down']['rf'].feature_importances_,
            index=FEATURES
        ).sort_values(ascending=True)

        fig2 = go.Figure(go.Bar(
            x=imp.values, y=imp.index,
            orientation='h', marker_color='#2196F3'
        ))
        fig2.update_layout(height=500, template='plotly_white',
                          title="What drives big down moves?")
        st.plotly_chart(fig2, use_container_width=True)

    # --- live signal ---
    st.header("Current Signal")

    recent = yf.download(ticker, period="3mo")
    if isinstance(recent.columns, pd.MultiIndex):
        recent.columns = recent.columns.get_level_values(0)

    recent = add_features(recent)
    recent.dropna(inplace=True)

    if len(recent) > 0:
        last_row = recent[FEATURES].iloc[[-1]]
        last_scaled = pd.DataFrame(
            sc.transform(last_row), columns=FEATURES, index=last_row.index
        )

        st.subheader(f"As of {recent.index[-1].date()}")
        c1, c2, c3 = st.columns(3)

        signals = []
        for i, (name, _) in enumerate([('Big Up', 'big_up'),
                                         ('Big Down', 'big_down'),
                                         ('Crash', 'crash')]):
            m = models.get(name)
            if m is None:
                continue
            prob = ensemble_predict(m['rf'], m['gb'], last_scaled)[0]
            signals.append((name, prob))

            col = [c1, c2, c3][i]
            with col:
                status = "ALERT" if prob > threshold else "Normal"
                st.metric(f"{name}", f"{prob*100:.1f}%", delta=status)
                st.progress(min(prob, 1.0))

        # overall vibe
        st.markdown("---")
        crash_p = next((p for n, p in signals if n == 'Crash'), 0)
        down_p = next((p for n, p in signals if n == 'Big Down'), 0)
        up_p = next((p for n, p in signals if n == 'Big Up'), 0)

        if crash_p > 0.5:
            st.error("High crash risk detected")
        elif down_p > 0.5:
            st.warning("Elevated downside risk")
        elif up_p > 0.5:
            st.success("Potential upside detected")
        else:
            st.info("No strong signals right now")

        # current indicators
        st.subheader("Technical Snapshot")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RSI (14)", f"{recent['rsi_14'].iloc[-1]:.1f}")
        c2.metric("Vol Ratio", f"{recent['vol_ratio'].iloc[-1]:.2f}")
        c3.metric("Dist from MA20", f"{recent['dist_ma20'].iloc[-1]*100:.1f}%")
        c4.metric("Up days (last 5)", f"{int(recent['consec_up'].iloc[-1])}/5")

    st.markdown("---")
    st.caption(
        "Educational project. Not financial advice. "
        "Past performance doesn't guarantee future results."
    )

else:
    st.info("Pick a ticker and hit Run Analysis to start")
    st.markdown("""
    **Some tickers to try:** AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA, META, JPM
    """)
