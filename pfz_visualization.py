"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

import plotly.graph_objects as go

import pfz_constants as C
import pfz_orchard as O


def _layout(fig, height=360, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.22), **kw)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def build_timeline(a, show=("gleich", "folgend", "anti", "voraus")):
    """Pfad 0: Preisindex des Marktes (rechte Achse) und Pflanzmenge der gewählten Regeln (linke Achse), je Jahr."""
    years = list(range(1, len(a.timeline["price"]) + 1))
    fig = go.Figure()
    for r in show:
        fig.add_trace(go.Scatter(x=years, y=a.timeline["planted"][r], name=C.RULE_NAME[r], mode="lines",
                                 line=dict(color=C.COLORS[r], width=3 if r == "gleich" else 1.8)))
    fig.add_trace(go.Scatter(x=years, y=a.timeline["price"], name="Preisindex des Marktes (rechts)", yaxis="y2", mode="lines",
                             line=dict(color=C.COLOR_PRICE, width=1.5, dash="dot")))
    pmax = O.max_planting(a.farm)
    fig.update_layout(yaxis=dict(title="gepflanzte Bäume je Jahr", range=[0, pmax * 1.05]),
                      yaxis2=dict(title="Preisindex", overlaying="y", side="right", range=[0, max(2.5, float(a.timeline["price"].max()) * 1.05)],
                                  showgrid=False, fixedrange=True, dtick=0.5, tickformat=".1f"),
                      xaxis=dict(title="Jahr"))
    return _layout(fig, 380)


def build_rule_bars(a):
    """Jahresüberschuss je Regel gegen gleichmäßiges Pflanzen in % (Fehlerbalken: zwei Standardfehler)."""
    rules = [r for r in C.RULES if r != "gleich"]
    vals = [a.stats[r].delta_pct for r in rules]
    fig = go.Figure(go.Bar(x=[C.RULE_NAME[r] for r in rules], y=vals, marker_color=[C.COLORS[r] for r in rules],
                           error_y=dict(type="data", array=[2 * a.stats[r].delta_se_pct for r in rules], visible=True),
                           text=[f"{v:+.1f} %".replace(".", ",") for v in vals], textposition="outside"))
    lo = min(-3.0, min(vals) * 1.25)
    hi = max(3.0, max(vals) * 1.25)
    fig.update_layout(yaxis=dict(title="gegen gleichmäßig [%]", range=[lo, hi], zeroline=True), xaxis=dict(tickangle=-20))
    return _layout(fig, 360)


def build_sweep(rows, show=("folgend", "anti", "voraus", "nach", "lueck", "perfekt")):
    """Experiment: Abstand zu gleichmäßigem Pflanzen in % über die Marktreaktion."""
    fig = go.Figure()
    for r in show:
        fig.add_trace(go.Scatter(x=[x["phi"] for x in rows], y=[x["delta"][r] for x in rows], mode="lines+markers", name=C.RULE_NAME[r],
                                 line=dict(color=C.COLORS[r], width=2.4 if r == "folgend" else 1.6),
                                 customdata=[x["spread"] for x in rows], hovertemplate="Marktreaktion %{x}, Preisstreuung %{customdata:.2f}: %{y:+.1f} %<extra></extra>"))
    fig.add_hline(y=0, line_color="#888", line_width=1)
    fig.update_layout(xaxis=dict(title="Marktreaktion (Preisflexibilität mal Pflanzreaktion)", tickvals=[x["phi"] for x in rows]),
                      yaxis=dict(title="gegen gleichmäßig [%]", range=[-70, 8]))
    return _layout(fig, 380)


def build_exact(ex):
    """Pfad 0: Preisindex der Verkaufsjahre und der Plan des LP gegen gleichmäßig und dem Preis folgen."""
    p0 = ex.path0
    years = list(range(1, len(p0["optimum"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=p0["optimum"], name="Optimum (LP)", mode="lines", line=dict(color="#222222", width=2.4)))
    for r in ("gleich", "folgend"):
        fig.add_trace(go.Scatter(x=years, y=p0["plans"][r], name=C.RULE_NAME[r], mode="lines", line=dict(color=C.COLORS[r], width=1.6)))
    fig.add_trace(go.Scatter(x=years, y=p0["price"][: len(years)], name="Preisindex (rechts)", yaxis="y2", mode="lines",
                             line=dict(color=C.COLOR_PRICE, width=1.4, dash="dot")))
    top = max(float(p0["optimum"].max()), float(p0["plans"]["folgend"].max()), 1.0) * 1.05
    fig.update_layout(yaxis=dict(title="gepflanzte Bäume je Jahr", range=[0, top]),
                      yaxis2=dict(title="Preisindex", overlaying="y", side="right", showgrid=False, fixedrange=True, dtick=0.5, tickformat=".1f",
                                  range=[0, max(2.5, float(p0["price"][: len(years)].max()) * 1.05)]),
                      xaxis=dict(title="Jahr"))
    return _layout(fig, 380)


def build_levers(sa):
    """Hebel der Saison in % des Gewinns ohne Markt bei Richtpreis."""
    levers = list(reversed(sa.levers))
    fig = go.Figure(go.Bar(y=[l.name for l in levers], x=[l.pct for l in levers], orientation="h",
                           marker_color=[C.COLOR_GOOD if l.pct >= 0 else C.COLOR_BAD for l in levers],
                           error_x=dict(type="data", array=[2 * l.se / sa.base * 100 for l in levers], visible=True),
                           text=[f"{l.pct:+.1f} %".replace(".", ",") for l in levers], textposition="outside", cliponaxis=False))
    hi = max(3.0, max(l.pct for l in levers) * 1.3)
    fig.update_layout(xaxis=dict(title="in % des Gewinns ohne Markt", range=[min(-1.0, min(l.pct for l in levers) * 1.3), hi]),
                      yaxis=dict(automargin=True))
    return _layout(fig, 330)
