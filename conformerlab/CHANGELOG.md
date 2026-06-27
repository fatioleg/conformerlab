# Changelog

Todas as mudanças notáveis neste projeto são registradas aqui.
Formato baseado em [Keep a Changelog](https://keepachangelog.com/);
versionamento [SemVer](https://semver.org/). Uma linha por mudança não trivial.

## [Unreleased]

### Added
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
