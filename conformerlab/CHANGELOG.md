# Changelog

Todas as mudanças notáveis neste projeto são registradas aqui.
Formato baseado em [Keep a Changelog](https://keepachangelog.com/);
versionamento [SemVer](https://semver.org/). Uma linha por mudança não trivial.

## [Unreleased]

### Added
- `refine/mlip.py`: padrão de registro de calculadoras (`register_calculator`, `list_methods`, `mlip_available`); `refine_with_mlip` não-destrutivo com `progress_callback`, `step_callback` e `RefinementSettings`.
- `core/types.py`: `RefinementSettings` (method, fmax, max_steps, device, `max_conformers`, `energy_only`) e campos `mlip_method`, `energy_trajectory`, `force_trajectory` em `ConformerRecord`.
- `io/export.py`: `mlip_method` adicionado a `_CSV_COLUMNS` e ao SDF (propriedade opcional).
- `app.py`: aba **⚡ Refinar MLIP** com seleção de método/device/parâmetros, modo Minimização geométrica vs Energia (ponto único) e seletor de quantos confôrmeros refinar (top-N por energia); refinamento em thread de background (`add_script_run_ctx`) com progresso em tempo real.
- `app.py`: gráficos de convergência em tempo real lado a lado (`st.columns`) — energia (kcal/mol) e força máxima (eV/Å) com linha tracejada do `fmax` — atualizados via `@st.fragment(run_every=1)` acima das abas.
- `app.py`: aba Refinar exibe tabela dos confôrmeros refinados, download CSV e SDF do ensemble, download SDF por confôrmero individual e **sobreposição 3D dos refinados** logo após os resultados; aba ConformerGen ganha 3ª sub-aba **Sobreposição Refinada** no viewer 3D (mesma sobreposição acessível pelos dois lugares).
- `app.py`: `_render_3d_overlay` ganhou parâmetro `key_suffix` para reutilização sem colisão de keys do Streamlit entre overlays original, refinado (ConformerGen) e da aba Refinar.
- `refine/mlip.py`: deduplicação pós-refino (`_dedup_records` + `RefinementSettings.dedup_rmsd`) — remove geometrias que convergiram ao mesmo mínimo via best-RMSD simétrico do RDKit, mantendo a de menor energia. Off por padrão na API; UI liga com slider (padrão 0.125 Å).
- `app.py`: tabela de refino ganha coluna **Rank original** (posição por energia no ensemble pré-refino) e nota de quantas duplicatas foram removidas; ao terminar o refino em background a tabela/overlay aparecem sozinhos (`st.rerun(scope="app")` disparado pelo fragment ao concluir).
- `app.py`: caixa de SMILES pré-preenchida com molécula de exemplo.
- `tests/test_refine_mlip.py`: 13 testes — inclui deduplicação (colapsa idênticas, mantém distintas, default off).

### Changed
- `app.py`: gráficos repensados — estilo profissional adaptável a tema claro/escuro (`_style_chart`): fundos transparentes (deixa o tema do Streamlit colorir), moldura completa (box) com eixos espelhados, grade neutra semi-transparente, paleta qualitativa de alto contraste (evita dois azuis idênticos) e `theme="streamlit"`. Removidas as cores hardcoded `#0e1117`/fonte branca.
- `app.py`: pesos de Boltzmann exibidos em porcentagem (ex.: `92.35%`) na tabela, nos seletores de confôrmero e no filtro (slider 0–100%).
- `app/icon.png`: imagem tratada — recortada para preencher o tile, em alta resolução (768px) e com **fundo branco removido (transparente), inclusive nos vãos internos** da molécula (chave de branco puro + preenchimento só de buracos pequenos para preservar os átomos de H). Legível em tema claro e escuro.

### Changed
- `app.py`: layout restruturado em 4 abas principais (Carregar / Gerar / ConformerGen / Refinar MLIP); cada aba exibe mensagem orientadora quando o pré-requisito ainda não foi cumprido.
- Nome de exibição do app alterado para **ConformerMPhysChem** (título da página, header e docstring); pacote Python permanece `conformerlab`.

### Fixed
- `refine/mlip.py`: erro `InvalidCxxCompiler: No working C++ compiler found` (torch._inductor) em ambientes sem `g++` (ex.: WSL sem build-essential). Solução: env vars `TORCHDYNAMO_DISABLE` / `TORCHDYNAMO_SUPPRESS_ERRORS` / `TORCH_COMPILE_DISABLE` definidas no nível do módulo **antes** do primeiro `import torch`, `_suppress_torch_compile()` para o caso de torch já importado, e `mace_off(..., compile=False)`. Não requer instalar g++.
- `refine/mlip.py`: `TypeError: _step_cb() missing 1 required positional argument: 'max_force'` no modo `energy_only` — a chamada single-point agora passa `step_callback(conf_id, 1, e, 0.0)` (4 args).
- `app.py`: `StreamlitDuplicateElementKey` ao abrir a aba ConformerGen com ensemble refinado — `_render_3d_overlay` ganhou parâmetro `key_suffix` para gerar keys únicas entre os overlays original e refinado.

### Added
- `analysis/align.py`: alinhamento de confôrmeros ancorado no **maior sistema de anéis conectado** (fused rings agrupados), com fallback para ligações duplas/aromáticas e depois para todos os pesados; funções públicas `planar_atom_ids`, `align_conformers` e `_largest_ring_system_ids`.
- `tests/test_analysis_align.py`: 7 testes cobrindo detecção de átomos planares e alinhamento (incluindo fallback para moléculas sem features planares).

### Changed
- `app.py` (overlay): substituído `_align_conformers` local por `analysis.align.align_conformers`, movendo a lógica científica para o módulo do pacote.
- `.claude/settings.json`: default de modelo do projeto = `sonnet`.

## [0.1.0]

### Added
- Core inicial: `core` (types, units, molecule, errors), interface
  `ConformerBackend` com backends rdkit/openconf, `analysis`
  (energy, boltzmann, rmsd, geometry, selection, pipeline), `refine.mlip`
  opcional, e `io.export` (CSV/XYZ/SDF/JSON).
- Testes: `test_molecule`, `test_analysis`, `test_backend_rdkit`.
