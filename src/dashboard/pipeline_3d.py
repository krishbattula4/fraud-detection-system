"""AI Fraud Risk Intelligence Network Visualizer component.

Renders an interactive spatial network topology representing:
Transaction Ingestion -> Preprocessing -> XGBoost / IForest / LOF Signals -> Hybrid Risk Engine -> Final Risk Decision
"""
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from typing import Dict, Any, Optional

def render_risk_network_pipeline(metrics: Optional[Dict[str, Any]] = None, current_risk_score: Optional[float] = None, theme_mode: str = "dark"):
    """Render Interactive Fraud Risk Network Pipeline Topology."""
    is_light = theme_mode == "light"
    bg_canvas = "#F8FAFC" if is_light else "#090D16"
    text_color = "#0F172A" if is_light else "#F8FAFC"
    border_color = "rgba(0,0,0,0.1)" if is_light else "rgba(255,255,255,0.1)"

    st.subheader("🌐 AI Fraud Risk Intelligence Network")
    st.caption("Interactive spatial topology of real-time multi-model risk analysis & decision flow.")

    weights = {"xgboost": 0.8, "isolation_forest": 0.1, "lof": 0.1}
    if metrics and "hybrid_configuration" in metrics:
        weights = metrics["hybrid_configuration"].get("weights", weights)

    score_val = current_risk_score if current_risk_score is not None else 18.5
    decision_color = "#10B981" if score_val < 40 else ("#F59E0B" if score_val < 70 else "#EF4444")
    decision_label = "LOW RISK (PASS)" if score_val < 40 else ("MEDIUM RISK (REVIEW)" if score_val < 70 else "HIGH RISK (BLOCK)")

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                overflow: hidden;
                background-color: {bg_canvas};
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            #container {{
                width: 100%;
                height: 380px;
                position: relative;
            }}
            canvas {{
                width: 100%;
                height: 100%;
            }}
            .tooltip {{
                position: absolute;
                padding: 8px 12px;
                background: {"rgba(255, 255, 255, 0.95)" if is_light else "rgba(15, 23, 42, 0.9)"};
                border: 1px solid {"rgba(79, 70, 229, 0.3)" if is_light else "rgba(99, 102, 241, 0.4)"};
                border-radius: 6px;
                color: {text_color};
                font-size: 11px;
                pointer-events: none;
                display: none;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                z-index: 10;
            }}
            .legend {{
                position: absolute;
                bottom: 10px;
                left: 10px;
                background: {"rgba(241, 245, 249, 0.9)" if is_light else "rgba(11, 15, 25, 0.8)"};
                border: 1px solid {border_color};
                border-radius: 6px;
                padding: 6px 12px;
                color: {"#475569" if is_light else "#94A3B8"};
                font-size: 11px;
                display: flex;
                gap: 12px;
            }}
            .dot {{
                width: 8px;
                height: 8px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 4px;
            }}
        </style>
    </head>
    <body>
        <div id="container">
            <canvas id="stage"></canvas>
            <div id="tooltip" class="tooltip"></div>
            <div class="legend">
                <span><span class="dot" style="background:#3B82F6;"></span> Ingestion</span>
                <span><span class="dot" style="background:#6366F1;"></span> XGBoost ({weights.get('xgboost', 0.8)*100:.0f}%)</span>
                <span><span class="dot" style="background:#F59E0B;"></span> IForest ({weights.get('isolation_forest', 0.1)*100:.0f}%)</span>
                <span><span class="dot" style="background:#F97316;"></span> LOF ({weights.get('lof', 0.1)*100:.0f}%)</span>
                <span><span class="dot" style="background:{decision_color};"></span> Decision ({score_val:.1f})</span>
            </div>
        </div>

        <script>
            const canvas = document.getElementById('stage');
            const ctx = canvas.getContext('2d');
            const container = document.getElementById('container');
            const tooltip = document.getElementById('tooltip');

            function resize() {{
                canvas.width = container.clientWidth * window.devicePixelRatio;
                canvas.height = container.clientHeight * window.devicePixelRatio;
            }}
            resize();
            window.addEventListener('resize', resize);

            const nodes = [
                {{ id: 'tx', name: 'Transaction Ingestion', type: 'Input', x: -220, y: 0, z: 0, color: '#3B82F6', radius: 14, info: 'Payload: Time, Amount, V1-V28' }},
                {{ id: 'prep', name: 'Robust Scaler & Preprocessor', type: 'Transform', x: -110, y: 0, z: 20, color: '#06B6D4', radius: 12, info: 'Fitted on 70% Train split' }},
                {{ id: 'xgb', name: 'XGBoost Calibrated Model', type: 'Supervised', x: 0, y: -70, z: -30, color: '#6366F1', radius: 16, info: 'Weight: {weights.get('xgboost', 0.8):.2f} | Pos-Weight: 518.2x' }},
                {{ id: 'if', name: 'Isolation Forest Anomaly', type: 'Unsupervised', x: 0, y: 0, z: 50, color: '#F59E0B', radius: 14, info: 'Weight: {weights.get('isolation_forest', 0.1):.2f} | Quantile Scaled' }},
                {{ id: 'lof', name: 'Local Outlier Factor (LOF)', type: 'Unsupervised', x: 0, y: 70, z: -20, color: '#F97316', radius: 14, info: 'Weight: {weights.get('lof', 0.1):.2f} | Novelty Detection' }},
                {{ id: 'hybrid', name: 'Hybrid Risk Engine', type: 'Ensemble', x: 120, y: 0, z: 0, color: '#EC4899', radius: 18, info: 'Weighted Risk Aggregator (0-100)' }},
                {{ id: 'dec', name: 'Final Decision: {decision_label}', type: 'Action', x: 230, y: 0, z: 0, color: '{decision_color}', radius: 20, info: 'Score: {score_val:.1f} / 100' }}
            ];

            const links = [
                {{ from: 0, to: 1 }},
                {{ from: 1, to: 2 }},
                {{ from: 1, to: 3 }},
                {{ from: 1, to: 4 }},
                {{ from: 2, to: 5 }},
                {{ from: 3, to: 5 }},
                {{ from: 4, to: 5 }},
                {{ from: 5, to: 6 }}
            ];

            let angleY = 0.005;
            let rotation = 0;
            let particles = [];

            for(let i=0; i<16; i++) {{
                particles.push({{
                    linkIdx: Math.floor(Math.random() * links.length),
                    progress: Math.random(),
                    speed: 0.008 + Math.random() * 0.012
                }});
            }}

            function project(x, y, z, rot) {{
                const rad = rot;
                const rx = x * Math.cos(rad) - z * Math.sin(rad);
                const rz = x * Math.sin(rad) + z * Math.cos(rad);
                const perspective = 500 / (500 + rz);
                const px = canvas.width / 2 + rx * perspective;
                const py = canvas.height / 2 + y * perspective;
                return {{ x: px, y: py, scale: perspective, zIndex: rz }};
            }}

            function draw() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                rotation += angleY;

                const projectedNodes = nodes.map(n => {{
                    const p = project(n.x, n.y, n.z, rotation);
                    return {{ ...n, px: p.x, py: p.y, scale: p.scale, zIndex: p.zIndex }};
                }});

                links.forEach(l => {{
                    const n1 = projectedNodes[l.from];
                    const n2 = projectedNodes[l.to];
                    ctx.beginPath();
                    ctx.moveTo(n1.px, n1.py);
                    ctx.lineTo(n2.px, n2.py);
                    ctx.strokeStyle = '{"rgba(79, 70, 229, 0.3)" if is_light else "rgba(99, 102, 241, 0.25)"}';
                    ctx.lineWidth = 2 * Math.min(n1.scale, n2.scale);
                    ctx.stroke();
                }});

                particles.forEach(pt => {{
                    pt.progress += pt.speed;
                    if (pt.progress > 1) pt.progress = 0;

                    const l = links[pt.linkIdx];
                    const n1 = projectedNodes[l.from];
                    const n2 = projectedNodes[l.to];

                    const cx = n1.px + (n2.px - n1.px) * pt.progress;
                    const cy = n1.py + (n2.py - n1.py) * pt.progress;

                    ctx.beginPath();
                    ctx.arc(cx, cy, 3 * n1.scale, 0, Math.PI * 2);
                    ctx.fillStyle = '#60A5FA';
                    ctx.fill();
                }});

                const sortedNodes = [...projectedNodes].sort((a, b) => b.zIndex - a.zIndex);

                sortedNodes.forEach(n => {{
                    const r = n.radius * n.scale * (window.devicePixelRatio > 1 ? 1.4 : 1.0);
                    
                    ctx.beginPath();
                    ctx.arc(n.px, n.py, r * 1.5, 0, Math.PI * 2);
                    ctx.fillStyle = n.color + '22';
                    ctx.fill();

                    ctx.beginPath();
                    ctx.arc(n.px, n.py, r, 0, Math.PI * 2);
                    ctx.fillStyle = n.color;
                    ctx.fill();

                    ctx.strokeStyle = '{"#0F172A" if is_light else "#FFFFFF"}';
                    ctx.lineWidth = 1.5 * n.scale;
                    ctx.stroke();

                    ctx.font = `${{11 * n.scale}}px Inter, sans-serif`;
                    ctx.fillStyle = '{"#334155" if is_light else "#E2E8F0"}';
                    ctx.textAlign = 'center';
                    ctx.fillText(n.name.split(' ')[0], n.px, n.py + r + 14 * n.scale);
                }});

                requestAnimationFrame(draw);
            }}

            draw();

            canvas.addEventListener('mousemove', (e) => {{
                const rect = canvas.getBoundingClientRect();
                const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
                const my = (e.clientY - rect.top) * (canvas.height / rect.height);

                let hovered = false;
                const projectedNodes = nodes.map(n => project(n.x, n.y, n.z, rotation));

                projectedNodes.forEach((n, idx) => {{
                    const dx = mx - n.x;
                    const dy = my - n.y;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist < nodes[idx].radius * 2 * n.scale) {{
                        hovered = true;
                        tooltip.style.display = 'block';
                        tooltip.style.left = (e.clientX - rect.left + 15) + 'px';
                        tooltip.style.top = (e.clientY - rect.top - 20) + 'px';
                        tooltip.innerHTML = `<strong>${{nodes[idx].name}}</strong><br/><span style="color:{"#64748B" if is_light else "#94A3B8"};">${{nodes[idx].info}}</span>`;
                    }}
                }});

                if (!hovered) tooltip.style.display = 'none';
            }});
        </script>
    </body>
    </html>
    """

    try:
        components.html(html_code, height=400)
    except Exception:
        render_plotly_network_fallback(weights, score_val, decision_color, theme_mode)

def render_plotly_network_fallback(weights: Dict[str, float], score: float, color: str, theme_mode: str = "dark"):
    """Render Plotly Scatter network graph fallback."""
    is_light = theme_mode == "light"
    bg_plotly = "rgba(248, 250, 252, 1.0)" if is_light else "rgba(9, 13, 22, 1.0)"

    x = [-2, -1, 0, 0, 0, 1, 2]
    y = [0, 0, 1.5, 0, -1.5, 0, 0]
    z = [0, 0.5, 0, -0.5, 0, 0, 0]
    names = [
        "Transaction Ingestion",
        "Robust Scaler Preprocessor",
        f"XGBoost Calibrated (w={weights.get('xgboost', 0.8):.2f})",
        f"Isolation Forest Anomaly (w={weights.get('isolation_forest', 0.1):.2f})",
        f"Local Outlier Factor (w={weights.get('lof', 0.1):.2f})",
        "Hybrid Risk Engine",
        f"Final Decision (Score: {score:.1f})",
    ]
    node_colors = ["#3B82F6", "#06B6D4", "#6366F1", "#F59E0B", "#F97316", "#EC4899", color]

    fig = go.Figure()

    edges = [(0, 1), (1, 2), (1, 3), (1, 4), (2, 5), (3, 5), (4, 5), (5, 6)]
    for e in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[x[e[0]], x[e[1]]],
                y=[y[e[0]], y[e[1]]],
                z=[z[e[0]], z[e[1]]],
                mode="lines",
                line=dict(color="rgba(79, 70, 229, 0.5)" if is_light else "rgba(99, 102, 241, 0.5)", width=4),
                hoverinfo="none",
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers+text",
            text=[n.split(" ")[0] for n in names],
            hovertext=names,
            marker=dict(size=14, color=node_colors, symbol="circle", opacity=0.9),
            showlegend=False,
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor=bg_plotly,
        ),
        paper_bgcolor=bg_plotly,
        margin=dict(l=0, r=0, b=0, t=0),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

# Alias for backward compatibility
render_3d_pipeline = render_risk_network_pipeline
