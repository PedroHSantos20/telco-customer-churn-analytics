import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import joblib


# ============================================================
# 1. LOAD MODEL FROM HUGGING FACE
# ============================================================

MODEL_REPO = "fnap/telco-churn-random-forest"
MODEL_FILENAME = "random_forest_model.joblib"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
model = joblib.load(model_path)

feature_order = list(model.feature_names_in_)


# ============================================================
# 2. FEATURE ENGINEERING HELPERS
# ============================================================

def contract_to_values(contract_type: str):
    """Return Contract_encoded and Contract_Value."""
    if contract_type == "Month-to-month":
        return 0, 1
    elif contract_type == "One year":
        return 1, 2
    elif contract_type == "Two year":
        return 2, 3
    return 0, 1


def payment_to_code(method: str):
    mapping = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3,
    }
    return mapping.get(method, 0)


def compute_cltv(monthly: float, tenure: int):
    """
    Estimativa simples de CLTV para efeitos de scoring.
    (Não é CLTV real, é apenas para alimentar o modelo na demo.)
    """
    return float(monthly * max(1, tenure) * 1.8)


def compute_engagement(
    satisfaction: float,
    total_services: int,
    tenure: int,
    monthly: float,
    contract_type: str,
    payment_method: str,
    has_streaming: bool,
):
    """
    Cálculo heurístico mais realista do Engagement_Score ∈ [0,1] + breakdown.

    Intuição:
      - clientes com mais satisfação, tenure, serviços e gasto mensal
        tendem a estar mais envolvidos;
      - contratos de maior duração aumentam o compromisso;
      - electronic/mailed check normalmente são “menos digitais”;
      - streaming é um sinal de uso de conteúdos.

    Retorna:
      score (float), components (lista de (nome, valor_normalizado, contribuição))
    """

    # 1) Satisfação (1–10) → [0,1]
    sat_score = float(np.clip(satisfaction / 10.0, 0.0, 1.0))

    # 2) Tenure (meses) → [0,1], saturando aos 36 meses
    tenure_score = float(np.clip(tenure / 36.0, 0.0, 1.0))

    # 3) Total de serviços (0–6) → [0,1]
    svc_score = float(np.clip(total_services / 6.0, 0.0, 1.0))

    # 4) Monthly charge (assumir 10–130 €) → [0,1]
    monthly_norm = (monthly - 10.0) / (130.0 - 10.0)
    spend_score = float(np.clip(monthly_norm, 0.0, 1.0))

    # 5) Bónus de streaming
    streaming_bonus = 0.08 if has_streaming else 0.0

    # 6) Estabilidade do contrato
    if contract_type == "Month-to-month":
        contract_bonus = 0.0
    elif contract_type == "One year":
        contract_bonus = 0.06
    elif contract_type == "Two year":
        contract_bonus = 0.10
    else:
        contract_bonus = 0.0

    # 7) Fricção do método de pagamento
    pm_penalty = 0.0
    if payment_method == "Electronic check":
        pm_penalty = -0.05
    elif payment_method == "Mailed check":
        pm_penalty = -0.03
    elif payment_method in ["Bank transfer (automatic)", "Credit card (automatic)"]:
        pm_penalty = 0.0

    # Pesos base
    w_sat = 0.35
    w_tenure = 0.25
    w_svc = 0.20
    w_spend = 0.15

    sat_contrib = w_sat * sat_score
    tenure_contrib = w_tenure * tenure_score
    svc_contrib = w_svc * svc_score
    spend_contrib = w_spend * spend_score
    streaming_contrib = streaming_bonus
    contract_contrib = contract_bonus
    pm_contrib = pm_penalty

    raw = (
        sat_contrib
        + tenure_contrib
        + svc_contrib
        + spend_contrib
        + streaming_contrib
        + contract_contrib
        + pm_contrib
    )

    score = float(np.clip(raw, 0.0, 1.0))

    components = [
        ("Satisfaction (1–10)", sat_score, sat_contrib),
        ("Tenure (0–72m)", tenure_score, tenure_contrib),
        ("Total services (0–6)", svc_score, svc_contrib),
        ("Monthly charge (10–130€)", spend_score, spend_contrib),
        ("Streaming usage", 1.0 if has_streaming else 0.0, streaming_contrib),
        ("Contract stability", 1.0, contract_contrib),
        ("Payment fricção", 1.0, pm_contrib),
    ]

    return score, components


def derive_features(
    age: int,
    tenure: int,
    monthly: float,
    satisfaction: int,
    n_services: int,
    contract_type: str,
    payment_method: str,
    has_streaming: bool,
):
    """
    Constrói o vetor de features completo, alinhado com as colunas esperadas pelo modelo.
    Retorna:
      X (DataFrame na ordem do modelo),
      engineered (dict com features derivadas),
      engagement_components (para o mini-waterfall textual)
    """
    longevity_ratio = tenure / age if age > 0 else 0.0
    total_services = int(n_services + (1 if has_streaming else 0))

    contract_encoded, contract_value = contract_to_values(contract_type)
    payment_code = payment_to_code(payment_method)

    engagement_score, engagement_components = compute_engagement(
        satisfaction=satisfaction,
        total_services=total_services,
        tenure=tenure,
        monthly=monthly,
        contract_type=contract_type,
        payment_method=payment_method,
        has_streaming=has_streaming,
    )

    cltv = compute_cltv(monthly, tenure)

    features = {
        "Age": age,
        "Tenure in Months": tenure,
        "Monthly Charge": monthly,
        "Satisfaction Score": satisfaction,
        "Total_Services": total_services,
        "Longevity_Ratio": longevity_ratio,
        "Contract_encoded": contract_encoded,
        "Contract_Value": contract_value,
        "Payment_code": payment_code,
        "Engagement_Score": engagement_score,
        "CLTV": cltv,
    }

    full_vector = {col: features.get(col, 0) for col in feature_order}
    X = pd.DataFrame([full_vector])[feature_order]

    engineered = {
        "Total_Services": total_services,
        "Longevity_Ratio": longevity_ratio,
        "Engagement_Score": engagement_score,
        "CLTV": cltv,
    }

    return X, engineered, engagement_components


# ============================================================
# 3. PLOTTING HELPERS
# ============================================================

def plot_gauge(proba: float):
    fig, ax = plt.subplots(figsize=(4, 2.2))
    ax.axis("off")

    ax.barh(0, 100, height=0.3, color="#e0e0e0")

    if proba < 0.30:
        color = "#2ecc71"
    elif proba < 0.60:
        color = "#f39c12"
    else:
        color = "#e74c3c"

    ax.barh(0, proba * 100, height=0.3, color=color)

    ax.text(
        50,
        0.15,
        f"{proba*100:.1f}% churn probability",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def plot_risk_bars(proba: float):
    fig, ax = plt.subplots(figsize=(4, 2))

    ax.barh(["Risk Level"], [proba * 100], color="#3498db")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Churn Probability (%)")

    ax.axvline(30, color="green", linestyle="--", alpha=0.6)
    ax.axvline(60, color="orange", linestyle="--", alpha=0.6)

    ax.text(15, 0.15, "Low", color="green", fontsize=10, fontweight="bold")
    ax.text(45, 0.15, "Medium", color="orange", fontsize=10, fontweight="bold")
    ax.text(80, 0.15, "High", color="red", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig


# ============================================================
# 4. RISK CATEGORY & ACTIONS
# ============================================================

def classify_risk(proba: float) -> str:
    if proba < 0.30:
        return "Low Risk"
    elif proba < 0.60:
        return "Medium Risk"
    return "High Risk"


def suggest_actions(risk_category: str):
    if risk_category == "Low Risk":
        return [
            "Explorar campanhas de upsell mantendo o foco em satisfação.",
            "Reforçar programas de fidelização e benefícios exclusivos.",
            "Monitorizar periodicamente a satisfação e a qualidade de serviço.",
        ]

    if risk_category == "Medium Risk":
        return [
            "Rever o pacote atual para otimizar custos e valor percebido.",
            "Contactar o cliente para recolher feedback e resolver irritantes.",
            "Oferecer upgrade técnico (velocidade / suporte) condicionado à permanência.",
        ]

    if risk_category == "High Risk":
        return [
            "Oferecer um benefício de retenção imediato (desconto, upgrade ou oferta).",
            "Propor migração para contrato anual com condições mais atrativas.",
            "Rever incidentes recentes (faturação, falhas, suporte) e atuar rapidamente.",
        ]

    return ["Sem recomendações disponíveis."]


# ============================================================
# 5. SCORING FUNCTION
# ============================================================

def score_customer(
    age: int,
    tenure: int,
    monthly: float,
    satisfaction: int,
    n_services: int,
    contract_type: str,
    payment_method: str,
    has_streaming: bool,
    threshold: float,
):
    X, engineered, eng_components = derive_features(
        age=age,
        tenure=tenure,
        monthly=monthly,
        satisfaction=satisfaction,
        n_services=n_services,
        contract_type=contract_type,
        payment_method=payment_method,
        has_streaming=has_streaming,
    )

    proba = float(model.predict_proba(X)[0, 1])
    label = int(proba >= threshold)
    risk = classify_risk(proba)
    actions = suggest_actions(risk)

    # Construir mini-waterfall textual para Engagement
    wf_lines = []
    for name, norm_val, contrib in eng_components:
        sign = "+" if contrib >= 0 else "−"
        wf_lines.append(
            f"- **{name}** → valor normalizado = `{norm_val:.2f}`, "
            f"contribuição = `{sign}{abs(contrib):.2f}`"
        )
    wf_text = "\n".join(wf_lines)

    md = f"""
### Customer Risk Assessment

- **Churn Probability:** `{proba*100:.1f}%`
- **Risk Category:** **{risk}**
- **Decision Threshold:** `{threshold:.2f}`
- **Predicted Label:** `{"Churn" if label == 1 else "No Churn"}`

---

### Key Engineered Features

- **Total Services:** `{engineered['Total_Services']}`
- **Longevity Ratio (Tenure/Age):** `{engineered['Longevity_Ratio']:.2f}`
- **Engagement Score (heurístico):** `{engineered['Engagement_Score']:.2f}`
- **Estimated CLTV:** `€{engineered['CLTV']:.0f}`

---

### Engagement Score – Explicação

*(Todos os valores de contribuição estão em unidades de score, antes de truncar para [0,1].)*

{wf_text}

---

### ✅ Recommended Actions

1. {actions[0]}
2. {actions[1]}
3. {actions[2]}
"""

    gauge_fig = plot_gauge(proba)
    bars_fig = plot_risk_bars(proba)

    input_view = pd.DataFrame(
        [
            {
                "Age": age,
                "Tenure (Months)": tenure,
                "Monthly Charge (€)": monthly,
                "Satisfaction (1–10)": satisfaction,
                "Number of Services": n_services,
                "Has Streaming": has_streaming,
                "Contract Type": contract_type,
                "Payment Method": payment_method,
            }
        ]
    )

    return md, gauge_fig, bars_fig, input_view


# ============================================================
# 6. PRESET SCENARIOS
# ============================================================

def preset_high_risk():
    """
    Cliente típico de alto risco:
    - Idade mais elevada, tenure curto, custo alto, baixa satisfação,
      muitos serviços, contrato M2M, electronic check, streaming.
    """
    return (
        65,                     # age
        2,                      # tenure
        95.0,                   # monthly
        4,                      # satisfaction
        4,                      # n_services
        "Month-to-month",       # contract_type
        "Electronic check",     # payment_method
        True,                   # has_streaming
        0.5,                    # threshold
    )


def preset_medium_risk():
    """
    Cliente borderline:
    - Meia-idade, tenure médio, custo moderado-alto, satisfação 6,
      alguns serviços, M2M, mailed check, streaming.
    """
    return (
        45,                     # age
        14,                     # tenure
        75.0,                   # monthly
        6,                      # satisfaction
        3,                      # n_services
        "Month-to-month",       # contract_type
        "Mailed check",         # payment_method
        True,                   # has_streaming
        0.5,                    # threshold
    )


def preset_low_risk():
    """
    Cliente de baixo risco:
    - Adulto, tenure longo, custo moderado, satisfação alta,
      poucos serviços, contrato 2 anos, débito direto, streaming.
    """
    return (
        38,                         # age
        48,                         # tenure
        55.0,                       # monthly
        8,                          # satisfaction
        2,                          # n_services
        "Two year",                 # contract_type
        "Bank transfer (automatic)",# payment_method
        True,                       # has_streaming
        0.5,                        # threshold
    )


# ============================================================
# 7. GRADIO UI
# ============================================================

with gr.Blocks(title="Telco Churn – RF Scoring Demo") as demo:
    gr.Markdown(
        """
# Telco Customer Churn – Random Forest Demo

Introduza os dados de um cliente ou escolha um cenário pré-definido.  
O modelo estima o risco de churn, calcula features derivadas e sugere ações de retenção.
        """
    )

    with gr.Row():
        with gr.Column():
            age_in = gr.Slider(18, 90, value=40, step=1, label="Age")
            tenure_in = gr.Slider(0, 72, value=12, step=1, label="Tenure (Months)")
            monthly_in = gr.Slider(10, 130, value=70, step=1, label="Monthly Charge (€)")
            satisfaction_in = gr.Slider(1, 10, value=7, step=1, label="Satisfaction Score (1–10)")

        with gr.Column():
            n_services_in = gr.Slider(0, 5, value=2, step=1, label="Number of Services")
            contract_in = gr.Radio(
                ["Month-to-month", "One year", "Two year"],
                value="Month-to-month",
                label="Contract Type",
            )
            payment_in = gr.Radio(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                value="Electronic check",
                label="Payment Method",
            )
            streaming_in = gr.Checkbox(value=True, label="Has Streaming?")
            threshold_in = gr.Slider(
                0.1,
                0.9,
                value=0.5,
                step=0.05,
                label="Decision Threshold",
            )

    gr.Markdown("### Preset Scenarios")

    with gr.Row():
        high_btn = gr.Button("🔴 High-risk example")
        med_btn = gr.Button("🟠 Medium-risk example")
        low_btn = gr.Button("🟢 Low-risk example")

    def _apply_preset(preset_name: str):
        if preset_name == "high":
            return preset_high_risk()
        if preset_name == "medium":
            return preset_medium_risk()
        return preset_low_risk()

    high_btn.click(
        fn=lambda: _apply_preset("high"),
        inputs=[],
        outputs=[
            age_in,
            tenure_in,
            monthly_in,
            satisfaction_in,
            n_services_in,
            contract_in,
            payment_in,
            streaming_in,
            threshold_in,
        ],
    )

    med_btn.click(
        fn=lambda: _apply_preset("medium"),
        inputs=[],
        outputs=[
            age_in,
            tenure_in,
            monthly_in,
            satisfaction_in,
            n_services_in,
            contract_in,
            payment_in,
            streaming_in,
            threshold_in,
        ],
    )

    low_btn.click(
        fn=lambda: _apply_preset("low"),
        inputs=[],
        outputs=[
            age_in,
            tenure_in,
            monthly_in,
            satisfaction_in,
            n_services_in,
            contract_in,
            payment_in,
            streaming_in,
            threshold_in,
        ],
    )

    gr.Markdown("### Score New Customer")

    btn = gr.Button("Score Customer", variant="primary")

    out_md = gr.Markdown(label="Risk Assessment")
    out_gauge = gr.Plot(label="Risk Gauge")
    out_bars = gr.Plot(label="Risk Zone Position")
    out_table = gr.Dataframe(label="Customer Input Summary")

    btn.click(
        fn=score_customer,
        inputs=[
            age_in,
            tenure_in,
            monthly_in,
            satisfaction_in,
            n_services_in,
            contract_in,
            payment_in,
            streaming_in,
            threshold_in,
        ],
        outputs=[out_md, out_gauge, out_bars, out_table],
    )

if __name__ == "__main__":
    demo.launch()
