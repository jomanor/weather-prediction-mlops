import os

import streamlit as st


def apply_unified_theme():
    """Apply coherent high-end dark glassmorphism design system across all Streamlit pages."""

    # Optional Security Passcode Lock for Private Netlify Deployment
    passcode = os.getenv("DASHBOARD_PASSCODE")
    if passcode:
        if "authenticated" not in st.session_state:
            st.session_state["authenticated"] = False

        if not st.session_state["authenticated"]:
            st.markdown(
                """
            <style>
                @import url('https://fonts.googleapis.com/css2?'
                            'family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
                html, body, [class*="css"] {
                    font-family: 'Plus Jakarta Sans', sans-serif !important;
                }
                .stApp { background: #020617 !important; color: #f8fafc !important; }
            </style>
            """,
                unsafe_allow_html=True,
            )
            st.title("🔒 Private Access Lock")
            st.markdown(
                "This deployment is private. "
                "Enter your security passcode to access telemetry analytics."
            )
            entered = st.text_input("Enter Passcode", type="password")
            if st.button("Unlock Dashboard", type="primary"):
                if entered == passcode:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Incorrect passcode.")
            st.stop()

    st.markdown(
        """
    <style>
        @import url('https://fonts.googleapis.com/css2?'
                    'family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&'
                    'family=JetBrains+Mono:wght@400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }

        /* App Deep Dark Background */
        .stApp {
            background: radial-gradient(
                circle at 15% 15%, #0f172a 0%, #090d16 50%, #020617 100%
            ) !important;
            color: #f8fafc !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.85) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }

        /* Coherent Page Title Banner */
        .page-header {
            background: linear-gradient(
                135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%
            );
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px 30px;
            margin-bottom: 24px;
            box-shadow: 0 12px 24px -10px rgba(0, 0, 0, 0.5);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .page-title {
            font-size: 2rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #ffffff 0%, #38bdf8 50%, #818cf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 !important;
            letter-spacing: -0.02em;
        }

        .page-subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
            font-weight: 400;
            margin-top: 4px;
        }

        /* Coherent Metric Cards */
        .metric-card {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            border-color: rgba(56, 189, 248, 0.4);
        }
        .metric-label {
            font-size: 0.8rem;
            color: #94a3b8;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-val {
            font-size: 2.2rem;
            font-weight: 800;
            font-family: 'JetBrains Mono', monospace;
            margin: 8px 0;
            color: #f8fafc;
        }
        .val-cyan { color: #00f2fe !important; }
        .val-pink { color: #ff007f !important; }
        .val-green { color: #00f5d4 !important; }

        /* Status Badge Pill */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34d399;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.825rem;
            font-weight: 600;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            background-color: #34d399;
            border-radius: 50%;
            box-shadow: 0 0 12px #34d399;
        }

        /* Tag Badges */
        .tag {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            margin-right: 6px;
        }
        .tag-cyan {
            background: rgba(56, 189, 248, 0.15);
            color: #38bdf8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .tag-purple {
            background: rgba(168, 85, 247, 0.15);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.3);
        }
        .tag-emerald {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
