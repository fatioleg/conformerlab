"""ConformerMPhysChem GUI — carregar, visualizar, gerar, refinar e explorar confôrmeros."""

from __future__ import annotations

import base64
import io
import sys
import tempfile
import threading
from pathlib import Path

import pandas as pd
import plotly.express as px
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from conformerlab.analysis.align import align_conformers
from conformerlab.core.errors import InvalidMoleculeFileError, InvalidSmilesError
from conformerlab.core.molecule import mol_from_input
from conformerlab.core.types import (
    EnsembleResult,
    GenerationSettings,
    MoleculeInput,
    RefinementSettings,
)
from conformerlab.refine.mlip import list_methods, refine_with_mlip


# ---------------------------------------------------------------------------
# Helpers — molécula de entrada
# ---------------------------------------------------------------------------

def _mol_2d_png(mol: Chem.Mol, width: int = 500, height: int = 300) -> bytes:
    mol2d = Chem.Mol(mol)
    mol2d.RemoveAllConformers()
    AllChem.Compute2DCoords(mol2d)
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.DrawMolecule(mol2d)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _embed_3d(mol: Chem.Mol) -> Chem.Mol | None:
    mol3d = Chem.Mol(mol)
    mol3d.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol3d, params) == -1:
        return None
    AllChem.MMFFOptimizeMolecule(mol3d)
    return mol3d


def _render_3d_single(mol: Chem.Mol, conf_id: int = 0, height: int = 400) -> None:
    """Renderiza um único confôrmero pelo conf_id — fundo branco, CPK ball+stick."""
    conf_mol = Chem.Mol(mol, False, conf_id)
    sdf = Chem.MolToMolBlock(conf_mol)
    view = py3Dmol.view(width="100%", height=height)
    view.addModel(sdf, "sdf")
    view.setStyle({}, {
        "stick": {"radius": 0.12, "colorscheme": "Jmol"},
        "sphere": {"scale": 0.22, "colorscheme": "Jmol"},
    })
    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=height + 10, scrolling=False)  # type: ignore[attr-defined]



def _render_3d_overlay(
    mol: Chem.Mol, records, height: int = 520, key_suffix: str = ""
) -> None:
    """Sobreposição estilo Rowan: selecionado em CPK sólido, demais como fantasmas."""
    sorted_recs = sorted(records, key=lambda r: r.delta_e_kcal or 0.0)
    ref_conf_id = sorted_recs[0].conf_id  # menor ΔE como referencial
    mol_aligned = align_conformers(mol, ref_conf_id=ref_conf_id)

    sel_idx = st.selectbox(
        "Confôrmero em destaque",
        options=range(len(sorted_recs)),
        format_func=lambda i: (
            f"#{sorted_recs[i].conf_id}  "
            f"ΔE = {sorted_recs[i].delta_e_kcal:.2f} kcal/mol  "
            f"Peso = {sorted_recs[i].boltzmann_weight * 100:.2f}%"
        ),
        key=f"overlay_highlight{key_suffix}",
    )
    highlight_id = sorted_recs[sel_idx].conf_id

    view = py3Dmol.view(width="100%", height=height)
    for i, rec in enumerate(sorted_recs):
        conf_mol = Chem.Mol(mol_aligned, False, rec.conf_id)
        sdf = Chem.MolToMolBlock(conf_mol)
        view.addModel(sdf, "sdf")
        if rec.conf_id == highlight_id:
            view.setStyle({"model": i}, {
                "stick": {"radius": 0.12, "colorscheme": "Jmol"},
                "sphere": {"scale": 0.22, "colorscheme": "Jmol"},
            })
        else:
            view.setStyle({"model": i}, {
                "stick": {"radius": 0.12, "colorscheme": "Jmol", "opacity": 0.80},
            })

    view.setBackgroundColor("white")
    view.zoomTo()
    components.html(view._make_html(), height=height + 10, scrolling=False)  # type: ignore[attr-defined]


def _ensemble_to_sdf_bytes(mol: Chem.Mol, records) -> bytes:
    buf = io.StringIO()
    writer = Chem.SDWriter(buf)
    for rec in records:
        conf_mol = Chem.Mol(mol, False, rec.conf_id)
        conf_mol.SetProp("_Name", f"conf_{rec.conf_id}")
        conf_mol.SetProp("delta_e_kcal", f"{rec.delta_e_kcal:.4f}" if rec.delta_e_kcal is not None else "")
        conf_mol.SetProp("boltzmann_weight", f"{rec.boltzmann_weight:.4f}" if rec.boltzmann_weight is not None else "")
        conf_mol.SetProp("rmsd_to_min_ang", f"{rec.rmsd_to_min_ang:.3f}" if rec.rmsd_to_min_ang is not None else "")
        writer.write(conf_mol)
    writer.close()
    return buf.getvalue().encode()


def _render_3d_input(mol: Chem.Mol, height: int = 420) -> None:
    mol3d = mol if mol.GetNumConformers() > 0 else _embed_3d(mol)
    if mol3d is None:
        st.warning("Não foi possível gerar coordenadas 3D.")
        return
    _render_3d_single(mol3d, conf_id=0, height=height)


def _load_molecule(
    mode: str, smiles_input: str | None, uploaded_file, charge: int
) -> tuple[MoleculeInput | None, str | None]:
    try:
        if mode == "SMILES":
            if not smiles_input or not smiles_input.strip():
                return None, None
            return MoleculeInput(smiles=smiles_input.strip()), None
        if uploaded_file is None:
            return None, None
        suffix = ".sdf" if mode == "SDF" else ".xyz"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        if mode == "SDF":
            return MoleculeInput(sdf_path=tmp_path), None
        return MoleculeInput(xyz_path=tmp_path, charge=charge), None
    except (InvalidSmilesError, InvalidMoleculeFileError, ValueError) as exc:
        return None, str(exc)


def _desc_table(mol: Chem.Mol, mol_input: MoleculeInput) -> None:
    formula = rdMolDescriptors.CalcMolFormula(mol)
    mw = round(Descriptors.MolWt(mol), 2)
    n_heavy = mol.GetNumHeavyAtoms()
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    fonte = (
        f"`{mol_input.smiles}`" if mol_input.smiles
        else "SDF" if mol_input.sdf_path
        else f"XYZ (carga {mol_input.charge})"
    )
    st.markdown(f"""
| Propriedade | Valor |
|---|---|
| Fórmula | **{formula}** |
| Massa molar | **{mw} g/mol** |
| Átomos pesados | {n_heavy} |
| H explícitos | {mol.GetNumAtoms() - n_heavy} |
| Ligações rotacionáveis | {n_rot} |
| Fonte | {fonte} |
""")


# ---------------------------------------------------------------------------
# MLIP refinement — background thread
# ---------------------------------------------------------------------------

def _run_refinement(ensemble: EnsembleResult, settings: RefinementSettings) -> None:
    """Runs in a daemon thread; writes progress and live trajectory into session_state."""
    def _progress_cb(done: int, total: int) -> None:
        st.session_state["refine_progress"] = (done, total)

    def _step_cb(conf_id: int, step: int, energy_kcal: float, max_force: float) -> None:
        st.session_state.setdefault("refine_live_trajectories", {}).setdefault(
            conf_id, []
        ).append(energy_kcal)
        st.session_state.setdefault("refine_live_forces", {}).setdefault(
            conf_id, []
        ).append(max_force)

    try:
        from conformerlab.analysis.pipeline import analyze  # local import avoids circular
        refined = refine_with_mlip(
            ensemble, settings,
            progress_callback=_progress_cb,
            step_callback=_step_cb,
        )
        refined = analyze(refined)
        st.session_state["ensemble_refined"] = refined
        st.session_state["refine_error"] = None
    except Exception as exc:  # noqa: BLE001
        st.session_state["refine_error"] = str(exc)
    finally:
        st.session_state["refine_running"] = False


# Paleta qualitativa com alto contraste mútuo, legível em fundo claro e escuro
# (evita dois azuis quase idênticos da paleta padrão do Plotly).
_CONF_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
]
_GRID = "rgba(128,128,128,0.22)"   # cinza neutro semi-transparente: ok nos 2 temas
_FRAME = "rgba(128,128,128,0.55)"  # moldura (box) do gráfico, visível nos 2 temas
_FMAX_COLOR = "#E45756"


def _style_chart(fig, title: str | None = None, *, height: int = 300):
    """Estilo profissional adaptável ao tema (claro/escuro).

    Não fixa cor de fonte — deixa o tema do Streamlit definir (passe
    ``theme="streamlit"`` em ``st.plotly_chart``). Fundos transparentes deixam
    o fundo do app aparecer; a grade usa cinza neutro que serve aos dois temas.
    """
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t": 48 if title else 12, "b": 10, "l": 10, "r": 10},
        title=(
            {"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 15}}
            if title else None
        ),
        legend={
            "orientation": "v", "yanchor": "top", "y": 1,
            "xanchor": "left", "x": 1.01, "bgcolor": "rgba(0,0,0,0)",
        },
        hovermode="x unified",
    )
    # moldura completa (box) — eixos espelhados nas 4 bordas, estilo científico
    fig.update_xaxes(
        showgrid=False, showline=True, linewidth=1, linecolor=_FRAME,
        mirror=True, ticks="outside", ticklen=4, tickcolor=_FRAME, zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor=_GRID, showline=True, linewidth=1,
        linecolor=_FRAME, mirror=True, ticks="outside", ticklen=4,
        tickcolor=_FRAME, zeroline=False,
    )
    return fig


@st.fragment(run_every=1)
def _progress_fragment() -> None:
    """Auto-refreshing panel above tabs: progress bar + live convergence chart."""
    running = st.session_state.get("refine_running", False)
    live: dict = st.session_state.get("refine_live_trajectories", {})
    error = st.session_state.get("refine_error")

    # Quando o refinamento (em thread) termina, força um rerun do app inteiro para
    # que a tabela e o overlay apareçam sozinhos — sem o usuário mexer em nada.
    if not running and st.session_state.get("refine_pending_rerun"):
        st.session_state["refine_pending_rerun"] = False
        st.rerun(scope="app")

    if not running and not live and not error:
        return

    if running:
        done, total = st.session_state.get("refine_progress", (0, 1))
        st.progress(done / max(total, 1), text=f"⚡ Refinando confôrmeros… {done}/{total}")
    elif error:
        col1, col2 = st.columns([10, 1])
        with col1:
            st.error(f"Erro no refinamento MLIP: {error}")
        with col2:
            if st.button("✕", key="dismiss_refine_err"):
                del st.session_state["refine_error"]
                st.rerun()

    if live:
        live_forces: dict = st.session_state.get("refine_live_forces", {})
        fmax_val: float = st.session_state.get("refine_fmax", 0.05)

        e_rows = [
            {"Confôrmero": f"#{cid}", "Passo": s + 1, "Energia (kcal/mol)": e}
            for cid, energies in live.items()
            for s, e in enumerate(energies)
        ]
        f_rows = [
            {"Confôrmero": f"#{cid}", "Passo": s + 1, "Força máx. (eV/Å)": f}
            for cid, forces in live_forces.items()
            for s, f in enumerate(forces)
        ]

        col_e, col_f = st.columns(2)
        with col_e:
            fig_e = px.line(
                pd.DataFrame(e_rows),
                x="Passo", y="Energia (kcal/mol)", color="Confôrmero",
                color_discrete_sequence=_CONF_COLORS,
            )
            _style_chart(fig_e, "Energia — tempo real", height=300)
            st.plotly_chart(
                fig_e, use_container_width=True, theme="streamlit",
                key="live_e_chart",
            )

        with col_f:
            if f_rows:
                fig_f = px.line(
                    pd.DataFrame(f_rows),
                    x="Passo", y="Força máx. (eV/Å)", color="Confôrmero",
                    color_discrete_sequence=_CONF_COLORS,
                )
                _style_chart(fig_f, "Força máx. — tempo real", height=300)
                fig_f.add_hline(
                    y=fmax_val, line_dash="dash", line_color=_FMAX_COLOR,
                    line_width=2, annotation_text=f"fmax = {fmax_val:.3f}",
                    annotation_position="top right",
                    annotation_font_color=_FMAX_COLOR,
                )
                st.plotly_chart(
                    fig_f, use_container_width=True, theme="streamlit",
                    key="live_f_chart",
                )


# ---------------------------------------------------------------------------
# Helpers — ensemble
# ---------------------------------------------------------------------------

def _ensemble_to_df(ensemble: EnsembleResult) -> pd.DataFrame:
    rows = []
    for r in ensemble.records:
        rows.append({
            "conf_id": r.conf_id,
            "E (kcal/mol)": round(r.energy_kcal, 4),
            "ΔE (kcal/mol)": round(r.delta_e_kcal, 4) if r.delta_e_kcal is not None else None,
            "Peso Boltzmann": (
                f"{r.boltzmann_weight * 100:.2f}%"
                if r.boltzmann_weight is not None else None
            ),
            "RMSD (Å)": round(r.rmsd_to_min_ang, 3) if r.rmsd_to_min_ang is not None else None,
            "Rg (Å)": round(r.radius_of_gyration_ang, 3) if r.radius_of_gyration_ang is not None else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_icon_path = Path(__file__).parent / "icon.png"
_icon = Image.open(_icon_path)
st.set_page_config(page_title="ConformerMPhysChem", page_icon=_icon, layout="wide")

# Render 3D da molécula (overlay de confôrmeros) recortado para preencher o tile.
# Fonte em 768px exibida em 64px → nitidez em telas HiDPI; o tile escuro com
# gradiente dá contraste ao render claro tanto no tema claro quanto no escuro.
_icon_b64 = base64.b64encode(_icon_path.read_bytes()).decode()
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:2px">
      <div style="width:74px;height:74px;flex-shrink:0;border-radius:16px;
                  display:flex;align-items:center;justify-content:center;
                  overflow:hidden;
                  background:linear-gradient(135deg,#1b1f2e 0%,#2c3350 100%);
                  box-shadow:0 1px 4px rgba(0,0,0,0.25)">
        <img src="data:image/png;base64,{_icon_b64}"
             style="width:66px;height:66px;display:block;
                    image-rendering:auto"/>
      </div>
      <div style="line-height:1.05">
        <div style="font-size:1.9rem;font-weight:700">ConformerMPhysChem</div>
        <div style="font-size:0.9rem;opacity:0.65;margin-top:2px">
          Geração e análise de confôrmeros — interface local
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

_progress_fragment()  # auto-refreshes every second during MLIP refinement

tab_load, tab_generate, tab_results, tab_refine = st.tabs([
    "📂 Carregar", "⚙️ Gerar", "📊 ConformerGen", "⚡ Refinar MLIP",
])

# ── Aba 1: Carregar ───────────────────────────────────────────────────────
with tab_load:
    input_mode = st.radio("Formato de entrada", ["SMILES", "SDF", "XYZ"], horizontal=True)

    charge = 0
    smiles_val = None
    uploaded = None

    if input_mode == "SMILES":
        smiles_val = st.text_input(
            "SMILES",
            value="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
            placeholder="ex: CC(=O)Oc1ccccc1C(=O)O",
        )
    elif input_mode == "SDF":
        uploaded = st.file_uploader("Arquivo SDF", type=["sdf", "mol"])
    else:
        uploaded = st.file_uploader("Arquivo XYZ", type=["xyz"])
        charge = st.slider("Carga total", min_value=-5, max_value=5, value=0)

    if st.button("Carregar molécula", type="primary"):
        mol_input, err = _load_molecule(input_mode, smiles_val, uploaded, charge)
        if err:
            st.session_state.pop("mol_input", None)
            st.error(f"Erro: {err}")
        elif mol_input is None:
            st.warning("Preencha o campo acima antes de carregar.")
        else:
            try:
                rdkit_mol = mol_from_input(mol_input, add_hs=True)
                st.session_state["mol_input"] = mol_input
                st.session_state["rdkit_mol"] = rdkit_mol
                st.session_state.pop("ensemble", None)
                st.success("Molécula carregada. Acesse a aba **Gerar**.")
            except (InvalidSmilesError, InvalidMoleculeFileError) as exc:
                st.session_state.pop("mol_input", None)
                st.error(str(exc))

    if "mol_input" in st.session_state:
        mol_input: MoleculeInput = st.session_state["mol_input"]
        rdkit_mol: Chem.Mol = st.session_state["rdkit_mol"]

        st.divider()
        desc_col, img_col = st.columns([1, 2])
        with desc_col:
            st.markdown("**Descritores**")
            _desc_table(rdkit_mol, mol_input)
        with img_col:
            sub2d, sub3d = st.tabs(["2D", "3D"])
            with sub2d:
                st.image(_mol_2d_png(rdkit_mol), use_container_width=True)
            with sub3d:
                _render_3d_input(rdkit_mol)

# ── Aba 2: Gerar ──────────────────────────────────────────────────────────
with tab_generate:
    if "mol_input" not in st.session_state:
        st.info("Carregue uma molécula na aba **Carregar** primeiro.")
    else:
        mol_input = st.session_state["mol_input"]

        c1, c2 = st.columns(2)
        with c1:
            backend_choice = st.selectbox("Backend", ["openconf", "rdkit"])
            preset = st.selectbox(
                "Preset",
                ["rapid", "ensemble", "docking", "macrocycle"],
                disabled=(backend_choice != "openconf"),
            )
            max_confs = st.slider("Máx. confôrmeros", 5, 500, 50, 5)
        with c2:
            energy_window = st.slider("Janela de energia (kcal/mol)", 1.0, 30.0, 10.0, 0.5)
            temperature = st.slider("Temperatura (K)", 200.0, 400.0, 298.0, 5.0)
            random_seed = st.slider("Semente aleatória", 0, 999, 42)

        if st.button("▶ Gerar confôrmeros", type="primary"):
            settings = GenerationSettings(
                preset=preset,
                max_conformers=int(max_confs),
                energy_window_kcal=float(energy_window),
                temperature_k=float(temperature),
                random_seed=int(random_seed),
            )
            with st.spinner("Gerando confôrmeros..."):
                try:
                    if backend_choice == "openconf":
                        from conformerlab.backends.openconf_backend import OpenConfBackend
                        backend = OpenConfBackend()
                        if not backend.is_available():
                            st.error("openconf não instalado. Escolha o backend rdkit.")
                            backend = None
                    else:
                        from conformerlab.backends.rdkit_backend import RDKitBackend
                        backend = RDKitBackend()
                    if backend is not None:
                        from conformerlab.analysis.pipeline import analyze
                        ensemble = backend.generate(mol_input, settings)
                        ensemble = analyze(ensemble)
                        st.session_state["ensemble"] = ensemble
                        st.success(
                            f"{len(ensemble.records)} confôrmeros gerados. "
                            "Acesse a aba **Resultados**."
                        )
                except Exception as exc:
                    st.error(f"Erro na geração: {exc}")

# ── Aba 4: Refinar MLIP ───────────────────────────────────────────────────
with tab_refine:
    if "ensemble" not in st.session_state:
        st.info("Gere confôrmeros na aba **Gerar** primeiro.")
    else:
        # Verifica dependências antes de expor os controles
        try:
            import ase as _ase  # type: ignore  # noqa: F401
            _ase_ok = True
        except ImportError:
            _ase_ok = False

        if not _ase_ok:
            st.error(
                "**ASE não instalado.** O refinamento MLIP requer ASE e pelo menos um "
                "potencial. Instale no ambiente do projeto:\n\n"
                "```bash\n"
                "pip install ase\n"
                "pip install 'aimnet[ase]'   # AIMNet2\n"
                "# ou\n"
                "pip install mace-torch      # MACE-OFF\n"
                "```"
            )
            st.stop()

        st.markdown("Re-optimize energias com um potencial de aprendizado de máquina (CPU local).")

        all_methods = list_methods()
        n_available = len(st.session_state["ensemble"].records)

        mlip_mode = st.radio(
            "Modo",
            ["Minimização geométrica", "Energia (ponto único)"],
            horizontal=True,
            help=(
                "**Minimização geométrica**: otimiza posições atômicas com LBFGS até convergência. "
                "**Energia (ponto único)**: calcula apenas a energia MLIP da geometria atual, sem mover átomos."
            ),
        )
        energy_only = mlip_mode == "Energia (ponto único)"

        r1, r2 = st.columns(2)
        with r1:
            mlip_method = st.selectbox("Método MLIP", all_methods, index=0)
            mlip_fmax = st.slider(
                "Limiar de convergência (eV/Å)", 0.01, 0.5, 0.05, 0.01,
                disabled=energy_only,
            )
            mlip_n = st.slider(
                "Confôrmeros a refinar (top-N por energia)",
                min_value=1, max_value=n_available,
                value=min(20, n_available), step=1,
                help="Serão selecionados os N de menor energia do ensemble original.",
            )
        with r2:
            mlip_steps = st.slider(
                "Máx. passos por confôrmero", 50, 500, 200, 50,
                disabled=energy_only,
            )
            mlip_device = st.selectbox("Dispositivo", ["cpu", "cuda"], index=0)
            mlip_dedup = st.slider(
                "Deduplicar (RMSD Å, 0 = desligado)", 0.0, 1.0, 0.125, 0.025,
                disabled=energy_only,
                help=(
                    "Após otimizar, remove geometrias que convergiram para o mesmo "
                    "mínimo (best-RMSD de átomos pesados abaixo do limiar), mantendo "
                    "a de menor energia."
                ),
            )

        st.caption(
            "O refinamento roda em segundo plano — você pode navegar pelas abas normalmente. "
            "O progresso aparece acima das abas."
        )

        running = st.session_state.get("refine_running", False)
        if st.button("⚡ Refinar confôrmeros", type="primary", disabled=running):
            settings_rf = RefinementSettings(
                method=mlip_method,
                fmax=float(mlip_fmax),
                max_steps=int(mlip_steps),
                device=mlip_device,
                max_conformers=int(mlip_n),
                energy_only=energy_only,
                dedup_rmsd=(float(mlip_dedup) if mlip_dedup > 0 else None),
            )
            st.session_state["refine_running"] = True
            st.session_state["refine_progress"] = (0, int(mlip_n))
            st.session_state["refine_live_trajectories"] = {}
            st.session_state["refine_live_forces"] = {}
            st.session_state["refine_fmax"] = float(mlip_fmax)
            st.session_state["refine_n_requested"] = int(mlip_n)
            st.session_state["refine_pending_rerun"] = True
            st.session_state.pop("ensemble_refined", None)
            st.session_state.pop("refine_error", None)
            from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
            ctx = get_script_run_ctx()
            t = threading.Thread(
                target=_run_refinement,
                args=(st.session_state["ensemble"], settings_rf),
                daemon=True,
            )
            add_script_run_ctx(t, ctx)
            t.start()
            st.rerun()

        if "ensemble_refined" in st.session_state:
            ens_ref = st.session_state["ensemble_refined"]
            method_used = ens_ref.records[0].mlip_method if ens_ref.records else "?"
            st.success(f"✅ Refinamento com **{method_used}** concluído.")

            n_req = st.session_state.get("refine_n_requested")
            n_final = len(ens_ref.records)
            if n_req is not None and n_final < n_req:
                st.caption(
                    f"🔁 Deduplicação: {n_final} estruturas únicas de "
                    f"{n_req} refinadas ({n_req - n_final} duplicatas removidas)."
                )

            st.subheader("Resultados do refinamento")
            df_ref = _ensemble_to_df(ens_ref)
            # Coluna com a posição (rank por energia) que o confôrmero ocupava no
            # ensemble ORIGINAL, antes do refinamento — para comparar reordenações.
            orig = st.session_state.get("ensemble")
            if orig is not None and orig.records:
                orig_sorted = sorted(orig.records, key=lambda r: r.energy_kcal)
                rank_map = {r.conf_id: i + 1 for i, r in enumerate(orig_sorted)}
                df_ref.insert(
                    1, "Rank original", df_ref["conf_id"].map(rank_map)
                )
            st.dataframe(df_ref, use_container_width=True, hide_index=True)

            dl_col, sdf_col = st.columns(2)
            with dl_col:
                st.download_button(
                    "⬇ Baixar tabela CSV",
                    data=df_ref.to_csv(index=False).encode(),
                    file_name=f"refined_{method_used}.csv",
                    mime="text/csv",
                    key="dl_ref_csv",
                )
            with sdf_col:
                mol_ref = ens_ref.mol
                if mol_ref is not None:
                    st.download_button(
                        "⬇ Baixar ensemble SDF",
                        data=_ensemble_to_sdf_bytes(mol_ref, ens_ref.records),
                        file_name=f"refined_{method_used}.sdf",
                        mime="chemical/x-mdl-sdfile",
                        key="dl_ref_sdf_all",
                    )

            st.markdown("**Download individual por confôrmero**")
            if ens_ref.mol is not None and ens_ref.records:
                sel_ref = st.selectbox(
                    "Selecionar confôrmero",
                    options=range(len(ens_ref.records)),
                    format_func=lambda i: (
                        f"#{ens_ref.records[i].conf_id}  "
                        f"E={ens_ref.records[i].energy_kcal:.3f} kcal/mol"
                        + (
                            f"  ΔE={ens_ref.records[i].delta_e_kcal:.2f}"
                            if ens_ref.records[i].delta_e_kcal is not None else ""
                        )
                    ),
                    key="sel_ref_conf",
                )
                rec_sel = ens_ref.records[sel_ref]
                st.download_button(
                    f"⬇ Baixar confôrmero #{rec_sel.conf_id} (SDF)",
                    data=_ensemble_to_sdf_bytes(ens_ref.mol, [rec_sel]),
                    file_name=f"conf_{rec_sel.conf_id}_{method_used}.sdf",
                    mime="chemical/x-mdl-sdfile",
                    key="dl_ref_sdf_single",
                )

            st.divider()
            st.markdown("**Sobreposição dos confôrmeros refinados**")
            if ens_ref.mol is not None and ens_ref.records:
                st.caption(
                    "Confôrmero em destaque em CPK · demais como fantasmas "
                    "alinhados por RMSD"
                )
                _render_3d_overlay(
                    ens_ref.mol, ens_ref.records, key_suffix="_refine_tab"
                )
            else:
                st.info("Geometrias 3D não disponíveis para sobreposição.")



# ── Aba 3: ConformerGen ───────────────────────────────────────────────────
def _show_ensemble_results(ensemble: EnsembleResult, key_suffix: str = "") -> None:
    """Render filters, table, chart and 3D viewers for one ensemble."""
    mol_3d: Chem.Mol = ensemble.mol  # type: ignore[assignment]
    all_records = ensemble.records
    n_total = len(all_records)

    st.subheader(f"{n_total} confôrmeros — backend: {ensemble.backend}")

    with st.expander("Filtros", expanded=False):
        de_vals = [r.delta_e_kcal for r in all_records if r.delta_e_kcal is not None]
        rmsd_vals = [r.rmsd_to_min_ang for r in all_records if r.rmsd_to_min_ang is not None]

        f1, f2, f3 = st.columns(3)
        with f1:
            de_max = st.slider(
                "ΔE máximo (kcal/mol)",
                min_value=0.0,
                max_value=float(max(de_vals)) if de_vals else 10.0,
                value=float(max(de_vals)) if de_vals else 10.0,
                step=0.1,
                key=f"de_max{key_suffix}",
            )
        with f2:
            bw_min_pct = st.slider(
                "Peso Boltzmann mínimo (%)",
                min_value=0.0, max_value=100.0, value=0.0, step=0.1,
                format="%.2f%%",
                key=f"bw_min{key_suffix}",
            )
            bw_min = bw_min_pct / 100.0
        with f3:
            rmsd_max = st.slider(
                "RMSD máximo (Å)",
                min_value=0.0,
                max_value=float(max(rmsd_vals)) if rmsd_vals else 5.0,
                value=float(max(rmsd_vals)) if rmsd_vals else 5.0,
                step=0.1,
                key=f"rmsd_max{key_suffix}",
            )

    records = [
        r for r in all_records
        if (r.delta_e_kcal is None or r.delta_e_kcal <= de_max)
        and (r.boltzmann_weight is None or r.boltzmann_weight >= bw_min)
        and (r.rmsd_to_min_ang is None or r.rmsd_to_min_ang <= rmsd_max)
    ]
    n = len(records)
    if n < n_total:
        st.caption(f"Mostrando **{n}** de {n_total} confôrmeros após filtros.")

    df = _ensemble_to_df(ensemble)
    df_filtered = df[df["conf_id"].isin(r.conf_id for r in records)]

    st.markdown("**Tabela de energias**")
    st.dataframe(df_filtered, use_container_width=True, hide_index=True)

    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            "⬇ Baixar tabela (CSV)",
            data=df_filtered.to_csv(index=False).encode(),
            file_name=f"ensemble_{ensemble.backend}.csv",
            mime="text/csv",
            key=f"dl_csv{key_suffix}",
        )
    with exp2:
        if mol_3d is not None:
            st.download_button(
                "⬇ Baixar ensemble (SDF)",
                data=_ensemble_to_sdf_bytes(mol_3d, records),
                file_name=f"ensemble_{ensemble.backend}.sdf",
                mime="chemical/x-mdl-sdfile",
                key=f"dl_sdf{key_suffix}",
            )

    fig = px.bar(
        df_filtered, x="conf_id", y="ΔE (kcal/mol)",
        color="ΔE (kcal/mol)",
        color_continuous_scale="RdYlGn_r",
        labels={"conf_id": "Confôrmero", "ΔE (kcal/mol)": "ΔE (kcal/mol)"},
    )
    _style_chart(fig, "Energia relativa por confôrmero", height=320)
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    fig.update_traces(marker_line_width=0)
    st.plotly_chart(
        fig, use_container_width=True, theme="streamlit", key=f"chart{key_suffix}"
    )

    st.divider()
    has_refined_overlay = (
        "ensemble_refined" in st.session_state
        and st.session_state["ensemble_refined"].mol is not None
    )
    _tab_names = ["Confôrmero individual", "Sobreposição (RMSD-alinhada)"]
    if has_refined_overlay:
        _tab_names.append("Sobreposição Refinada")
    _viewer_tabs = st.tabs(_tab_names)
    viewer_tab = _viewer_tabs[0]
    overlay_tab = _viewer_tabs[1]
    refined_overlay_tab = _viewer_tabs[2] if has_refined_overlay else None

    with viewer_tab:
        if not records:
            st.warning("Nenhum confôrmero passa nos filtros atuais.")
        else:
            sel_idx = st.selectbox(
                "Selecionar confôrmero",
                options=range(n),
                format_func=lambda i: (
                    f"#{records[i].conf_id}  "
                    f"ΔE={records[i].delta_e_kcal:.2f} kcal/mol  "
                    f"Boltzmann={records[i].boltzmann_weight * 100:.2f}%"
                ),
                key=f"sel_conf{key_suffix}",
            )
            if mol_3d is not None:
                _render_3d_single(mol_3d, conf_id=records[sel_idx].conf_id)
            else:
                st.warning("Geometrias 3D não disponíveis.")

    with overlay_tab:
        st.caption("Confôrmero selecionado em CPK · demais como fantasmas alinhados por RMSD")
        if not records:
            st.warning("Nenhum confôrmero passa nos filtros atuais.")
        elif mol_3d is not None:
            _render_3d_overlay(mol_3d, records, key_suffix=key_suffix)
        else:
            st.warning("Geometrias 3D não disponíveis.")

    if refined_overlay_tab is not None:
        with refined_overlay_tab:
            ens_ref = st.session_state["ensemble_refined"]
            ref_records = ens_ref.records
            ref_mol = ens_ref.mol
            st.caption(
                f"Sobreposição dos confôrmeros refinados ({ens_ref.records[0].mlip_method if ens_ref.records else '?'}) "
                "· confôrmero de menor energia em CPK · demais como fantasmas"
            )
            if ref_records and ref_mol is not None:
                _render_3d_overlay(
                    ref_mol, ref_records, key_suffix=f"{key_suffix}_refined"
                )
            else:
                st.warning("Nenhum confôrmero refinado disponível.")


with tab_results:
    if "ensemble" not in st.session_state:
        st.info("Gere confôrmeros na aba **Gerar** primeiro.")
    else:
        has_refined = "ensemble_refined" in st.session_state
        if has_refined:
            view_choice = st.radio(
                "Mostrar ensemble",
                ["Original", "Refinado (MLIP)"],
                horizontal=True,
                key="results_view",
            )
            active_ensemble = (
                st.session_state["ensemble_refined"]
                if view_choice == "Refinado (MLIP)"
                else st.session_state["ensemble"]
            )
            key_sfx = "_r" if view_choice == "Refinado (MLIP)" else "_o"
        else:
            active_ensemble = st.session_state["ensemble"]
            key_sfx = "_o"

        _show_ensemble_results(active_ensemble, key_suffix=key_sfx)
