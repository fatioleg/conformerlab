"""conformerlab GUI — Fases 1 e 2: carregar, visualizar, gerar e explorar confôrmeros."""

from __future__ import annotations

import base64
import io
import sys
import tempfile
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
from conformerlab.core.types import EnsembleResult, GenerationSettings, MoleculeInput


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



def _render_3d_overlay(mol: Chem.Mol, records, height: int = 520) -> None:
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
            f"Peso = {sorted_recs[i].boltzmann_weight:.3f}"
        ),
        key="overlay_highlight",
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
# Helpers — ensemble
# ---------------------------------------------------------------------------

def _ensemble_to_df(ensemble: EnsembleResult) -> pd.DataFrame:
    rows = []
    for r in ensemble.records:
        rows.append({
            "conf_id": r.conf_id,
            "E (kcal/mol)": round(r.energy_kcal, 4),
            "ΔE (kcal/mol)": round(r.delta_e_kcal, 4) if r.delta_e_kcal is not None else None,
            "Peso Boltzmann": round(r.boltzmann_weight, 4) if r.boltzmann_weight is not None else None,
            "RMSD (Å)": round(r.rmsd_to_min_ang, 3) if r.rmsd_to_min_ang is not None else None,
            "Rg (Å)": round(r.radius_of_gyration_ang, 3) if r.radius_of_gyration_ang is not None else None,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

_icon_path = Path(__file__).parent / "icon.png"
_icon = Image.open(_icon_path)
st.set_page_config(page_title="conformerlab", page_icon=_icon, layout="wide")

_icon_b64 = base64.b64encode(_icon_path.read_bytes()).decode()
st.markdown(
    f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px">'
    f'<img src="data:image/png;base64,{_icon_b64}"'
    f' width="64" style="border-radius:10px;flex-shrink:0"/>'
    f'<h1 style="margin:0;line-height:1">conformerlab</h1>'
    f'</div>',
    unsafe_allow_html=True,
)
st.caption("Geração e análise de confôrmeros — interface local")

tab_load, tab_generate, tab_results = st.tabs([
    "📂 Carregar", "⚙️ Gerar", "📊 Resultados",
])

# ── Aba 1: Carregar ───────────────────────────────────────────────────────
with tab_load:
    input_mode = st.radio("Formato de entrada", ["SMILES", "SDF", "XYZ"], horizontal=True)

    charge = 0
    smiles_val = None
    uploaded = None

    if input_mode == "SMILES":
        smiles_val = st.text_input("SMILES", placeholder="ex: CC(=O)Oc1ccccc1C(=O)O")
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

# ── Aba 4: Resultados ─────────────────────────────────────────────────────
with tab_results:
    if "ensemble" not in st.session_state:
        st.info("Gere confôrmeros na aba **Gerar** primeiro.")
    else:
        ensemble: EnsembleResult = st.session_state["ensemble"]
        mol_3d: Chem.Mol = ensemble.mol  # type: ignore[assignment]
        all_records = ensemble.records
        n_total = len(all_records)

        st.subheader(f"{n_total} confôrmeros — backend: {ensemble.backend}")

        # ── Filtros ───────────────────────────────────────────────────────
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
                )
            with f2:
                bw_min = st.slider(
                    "Peso Boltzmann mínimo",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.001,
                    format="%.3f",
                )
            with f3:
                rmsd_max = st.slider(
                    "RMSD máximo (Å)",
                    min_value=0.0,
                    max_value=float(max(rmsd_vals)) if rmsd_vals else 5.0,
                    value=float(max(rmsd_vals)) if rmsd_vals else 5.0,
                    step=0.1,
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

        # ── Tabela + Export ───────────────────────────────────────────────
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
            )
        with exp2:
            if mol_3d is not None:
                st.download_button(
                    "⬇ Baixar ensemble (SDF)",
                    data=_ensemble_to_sdf_bytes(mol_3d, records),
                    file_name=f"ensemble_{ensemble.backend}.sdf",
                    mime="chemical/x-mdl-sdfile",
                )

        # ── Gráfico de barras ─────────────────────────────────────────────
        st.markdown("**Energia relativa por confôrmero**")
        fig = px.bar(
            df_filtered, x="conf_id", y="ΔE (kcal/mol)",
            color="ΔE (kcal/mol)",
            color_continuous_scale="RdYlGn_r",
            labels={"conf_id": "Confôrmero", "ΔE (kcal/mol)": "ΔE (kcal/mol)"},
            height=300,
        )
        fig.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Viewer 3D ─────────────────────────────────────────────────────
        st.divider()
        viewer_tab, overlay_tab = st.tabs([
            "Confôrmero individual", "Sobreposição (RMSD-alinhada)",
        ])

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
                        f"Boltzmann={records[i].boltzmann_weight:.3f}"
                    ),
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
                _render_3d_overlay(mol_3d, records)
            else:
                st.warning("Geometrias 3D não disponíveis.")
