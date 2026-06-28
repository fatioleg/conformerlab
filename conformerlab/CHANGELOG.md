# Changelog

Todas as mudanças notáveis neste projeto são registradas aqui.
Formato baseado em [Keep a Changelog](https://keepachangelog.com/);
versionamento [SemVer](https://semver.org/). Uma linha por mudança não trivial.

## [Unreleased]

### Changed
- `app.py`: layout restruturado em 4 abas principais (Carregar / Prévia / Gerar / Resultados) em vez de seções sequenciais numa página única; cada aba exibe mensagem orientadora quando o pré-requisito ainda não foi cumprido.

### Added
- `analysis/align.py`: alinhamento de confôrmeros ancorado no **maior sistema de anéis conectado** (fused rings agrupados), com fallback para ligações duplas/aromáticas e depois para todos os pesados; funções públicas `planar_atom_ids`, `align_conformers` e `_largest_ring_system_ids`.
- `tests/test_analysis_align.py`: 7 testes cobrindo detecção de átomos planares e alinhamento (incluindo fallback para moléculas sem features planares).

### Changed
- `app.py` (overlay): substituído `_align_conformers` local por `analysis.align.align_conformers`, movendo a lógica científica para o módulo correto conforme `AGENTS.md`.
- `CHANGELOG.md` e `DECISIONS.md` para rastreabilidade de mudanças e decisões.
- `CLAUDE.md` (raiz): política operacional do Claude Code — política de modelos
  para minimizar tokens e regra de delegação de programação pesada ao Codex.
- `.claude/settings.json`: default de modelo do projeto = `sonnet`.

## [0.1.0]

### Added
- Core inicial: `core` (types, units, molecule, errors), interface
  `ConformerBackend` com backends rdkit/openconf, `analysis`
  (energy, boltzmann, rmsd, geometry, selection, pipeline), `refine.mlip`
  opcional, e `io.export` (CSV/XYZ/SDF/JSON).
- Testes: `test_molecule`, `test_analysis`, `test_backend_rdkit`.
