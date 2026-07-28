import streamlit as st
import pickle
import numpy as np

# -----------------------------
# Load model and artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("train_model3.pkl", "rb"))
    encoders = pickle.load(open("encoders3.pkl", "rb"))
    player_maps = pickle.load(open("player_maps.pkl", "rb"))
    return model, encoders, player_maps

model, encoders, player_maps = load_artifacts()

le_batteam = encoders["batting_team_encoder"]
le_bowlteam = encoders["bowling_team_encoder"]
phase_sr_map = encoders["batsman_phase_sr"]          # (batsman, phase) -> SR
phase_econ_map = encoders["bowler_phase_econ"]        # (bowler, phase) -> econ
phase_sr_counts = encoders["batsman_phase_counts"]    # (batsman, phase) -> ball count
phase_econ_counts = encoders["bowler_phase_counts"]   # (bowler, phase) -> ball count
career_sr_map = encoders["batsman_career_sr"]         # batsman -> SR
career_econ_map = encoders["bowler_career_econ"]      # bowler -> econ
global_sr = encoders["global_sr"]
global_econ = encoders["global_econ"]
MIN_PHASE_SAMPLES = encoders.get("min_phase_samples", 20)

team_batsmen_map = player_maps["team_batsmen_map"]
team_bowlers_map = player_maps["team_bowlers_map"]


def phase_of(over: int) -> str:
    if over <= 6:
        return "powerplay"
    elif over <= 15:
        return "middle"
    else:
        return "death"


def get_phase_sr(batsman, phase):
    """Returns (value, is_reliable, ball_count). Never silently blends —
    caller is responsible for telling the user when is_reliable is False."""
    count = phase_sr_counts.get((batsman, phase), 0)
    if count >= MIN_PHASE_SAMPLES:
        return phase_sr_map[(batsman, phase)], True, count
    return career_sr_map.get(batsman, global_sr), False, count


def get_phase_econ(bowler, phase):
    count = phase_econ_counts.get((bowler, phase), 0)
    if count >= MIN_PHASE_SAMPLES:
        return phase_econ_map[(bowler, phase)], True, count
    return career_econ_map.get(bowler, global_econ), False, count


def get_career_sr(batsman):
    return career_sr_map.get(batsman, global_sr)


def get_career_econ(bowler):
    return career_econ_map.get(bowler, global_econ)


# -----------------------------
# Real IPL team colors (jersey-accurate where possible)
# -----------------------------
TEAM_COLORS = {
    "Chennai Super Kings": "#F9CD05",
    "Mumbai Indians": "#004BA0",
    "Royal Challengers Bangalore": "#EC1C24",
    "Kolkata Knight Riders": "#3A225D",
    "Delhi Daredevils": "#17479E",
    "Sunrisers Hyderabad": "#FF822A",
    "Rajasthan Royals": "#EA1E8C",
    "Kings XI Punjab": "#ED1B24",
    "Deccan Chargers": "#000080",
    "Gujarat Lions": "#EB1B23",
    "Pune Warriors": "#5B84B1",
    "Rising Pune Supergiant": "#C2A15E",
    "Rising Pune Supergiants": "#C2A15E",
    "Kochi Tuskers Kerala": "#F26522",
}
DEFAULT_ACCENT = "#3AF23A"  # LED-green fallback

st.set_page_config(page_title="IPL Runs Predictor", page_icon="🏏", layout="wide")

# -----------------------------
# Session state for team selection (needed before we know accent color)
# -----------------------------
if "batting_team" not in st.session_state:
    st.session_state.batting_team = sorted(le_batteam.classes_)[0]

accent = TEAM_COLORS.get(st.session_state.batting_team, DEFAULT_ACCENT)

# -----------------------------
# Broadcast-style CSS
# -----------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=JetBrains+Mono:wght@500;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Oswald', sans-serif;
}}

.stApp {{
    background: radial-gradient(circle at 50% -20%, #14181f 0%, #0a0c10 60%, #05060a 100%);
}}

.scorecard-strip {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #10131a;
    border: 1px solid #262b36;
    border-left: 6px solid {accent};
    border-radius: 6px;
    padding: 18px 28px;
    margin-bottom: 22px;
}}
.scorecard-team {{
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #f2f2f2;
    text-transform: uppercase;
}}
.scorecard-vs {{
    font-family: 'JetBrains Mono', monospace;
    color: #6b7280;
    font-size: 1rem;
    padding: 0 18px;
}}
.scorecard-meta {{
    font-family: 'JetBrains Mono', monospace;
    color: #9ca3af;
    font-size: 0.95rem;
    text-align: right;
}}

/* Probability board — replaces the old single LED digit */
.prob-board {{
    background: #050705;
    border: 2px solid #1b1f1b;
    border-radius: 8px;
    padding: 22px 26px;
    box-shadow: inset 0 0 25px rgba(0,0,0,0.6);
}}
.prob-row {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 14px;
}}
.prob-row:last-child {{
    margin-bottom: 0;
}}
.prob-outcome {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: #e5e7eb;
    width: 38px;
    text-align: center;
}}
.prob-track {{
    flex: 1;
    background: #14171c;
    border-radius: 4px;
    height: 26px;
    position: relative;
    overflow: hidden;
}}
.prob-fill {{
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}}
.prob-pct {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    font-weight: 700;
    color: #f2f2f2;
    width: 56px;
    text-align: right;
}}
.prob-top-label {{
    font-family: 'JetBrains Mono', monospace;
    color: #6b7280;
    font-size: 0.78rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 16px;
}}

/* Stat chips */
.stat-chip {{
    background: #10131a;
    border: 1px solid #262b36;
    border-radius: 6px;
    padding: 14px 18px;
    text-align: center;
}}
.stat-chip-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #f2f2f2;
}}
.stat-chip-label {{
    font-size: 0.78rem;
    color: #6b7280;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 4px;
}}

.stButton>button {{
    background-color: {accent} !important;
    color: #05060a !important;
    font-weight: 700;
    border: none;
    letter-spacing: 1px;
    text-transform: uppercase;
}}

section[data-testid="stSidebar"] {{
    background: #0a0c10;
    border-right: 1px solid #1b1f28;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar — inputs
# -----------------------------
with st.sidebar:
    st.markdown("### Match Setup")

    inning = st.selectbox("Inning", [1, 2])
    over = st.slider("Over", min_value=1, max_value=20, value=10)
    ball = st.slider("Ball", min_value=1, max_value=6, value=3)

    st.markdown("---")
    batting_team = st.selectbox(
        "Batting Team", sorted(le_batteam.classes_),
        key="batting_team",
    )
    bowling_team = st.selectbox("Bowling Team", sorted(le_bowlteam.classes_))

    st.markdown("---")
    batsman = None
    bowler = None
    if batting_team == bowling_team:
        st.warning("Pick two different teams.")
    else:
        batsman_options = sorted(team_batsmen_map.get(batting_team, []))
        bowler_options = sorted(team_bowlers_map.get(bowling_team, []))
        batsman = st.selectbox("Batsman", batsman_options, key=f"batsman_{batting_team}")
        bowler = st.selectbox("Bowler", bowler_options, key=f"bowler_{bowling_team}")

    st.markdown("---")
    predict_clicked = st.button("Predict Ball Outcome", use_container_width=True)

# -----------------------------
# Scorecard strip
# -----------------------------
st.markdown(
    '<div class="scorecard-strip">'
    f'<div class="scorecard-team">{batting_team}</div>'
    '<div class="scorecard-vs">VS</div>'
    f'<div class="scorecard-team">{bowling_team}</div>'
    f'<div class="scorecard-meta">INN {inning} &nbsp;|&nbsp; OVER {over}.{ball}</div>'
    '</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Result area
# -----------------------------
if predict_clicked and batting_team != bowling_team and batsman and bowler:
    phase = phase_of(over)

    batting_encoded = le_batteam.transform([batting_team])[0]
    bowling_encoded = le_bowlteam.transform([bowling_team])[0]

    p_sr, sr_reliable, sr_count = get_phase_sr(batsman, phase)
    p_econ, econ_reliable, econ_count = get_phase_econ(bowler, phase)
    c_sr = get_career_sr(batsman)
    c_econ = get_career_econ(bowler)

    if not sr_reliable or not econ_reliable:
        notice_bits = []
        if not sr_reliable:
            notice_bits.append(
                f"**{batsman}** has only **{sr_count} ball(s)** on record in the {phase} phase "
                f"(need {MIN_PHASE_SAMPLES}+) — showing career figures for this phase instead of a "
                f"guessed/blended number."
            )
        if not econ_reliable:
            notice_bits.append(
                f"**{bowler}** has only **{econ_count} ball(s)** on record in the {phase} phase "
                f"(need {MIN_PHASE_SAMPLES}+) — showing career figures for this phase instead of a "
                f"guessed/blended number."
            )
        st.warning("Insufficient phase-specific data:  \n" + "  \n".join(notice_bits))

    features = np.array([[
        inning, over, ball, batting_encoded, bowling_encoded,
        p_sr, p_econ, c_sr, c_econ,
    ]])

    proba = model.predict_proba(features)[0]
    classes = model.classes_  # e.g. [0, 1, 2, 3, 4, 6]

    # sort outcomes in natural cricket order, not by probability
    order = [0, 1, 2, 3, 4, 6]
    outcome_probs = {c: p for c, p in zip(classes, proba)}

    board_col, chip_col1, chip_col2 = st.columns([1.4, 1, 1])

    with board_col:
        rows = []
        for outcome in order:
            if outcome not in outcome_probs:
                continue
            pct = outcome_probs[outcome] * 100
            rows.append(
                '<div class="prob-row">'
                f'<div class="prob-outcome">{outcome}</div>'
                '<div class="prob-track">'
                f'<div class="prob-fill" style="width:{pct}%; background:{accent};"></div>'
                '</div>'
                f'<div class="prob-pct">{pct:.1f}%</div>'
                '</div>'
            )
        rows_html = "".join(rows)
        board_html = (
            '<div class="prob-board">'
            '<div class="prob-top-label">Outcome Probability — This Delivery</div>'
            f'{rows_html}'
            '</div>'
        )
        st.markdown(board_html, unsafe_allow_html=True)

    sr_source = f"{phase} · {sr_count} balls" if sr_reliable else f"career (only {sr_count} in {phase})"
    econ_source = f"{phase} · {econ_count} balls" if econ_reliable else f"career (only {econ_count} in {phase})"

    with chip_col1:
        st.markdown(
            '<div class="stat-chip">'
            f'<div class="stat-chip-value">{p_sr:.1f}</div>'
            f'<div class="stat-chip-label">{batsman} — SR ({sr_source})</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with chip_col2:
        st.markdown(
            '<div class="stat-chip">'
            f'<div class="stat-chip-value">{p_econ:.2f}</div>'
            f'<div class="stat-chip-label">{bowler} — Economy ({econ_source})</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "Probabilities come from a Random Forest Classifier trained on phase-specific "
        "(powerplay / middle / death) strike rate and economy, plus career averages as a fallback "
        "signal. A single ball has huge inherent randomness — even real broadcast analytics "
        "shows odds, not a guaranteed outcome."
    )
else:
    st.info("Set the match situation in the sidebar, then hit **Predict Ball Outcome**.")

st.divider()
st.caption(
    "Model: Random Forest Classifier · Features: match context + phase-specific batsman SR "
    "+ phase-specific bowler economy + career SR/economy fallback"
)
