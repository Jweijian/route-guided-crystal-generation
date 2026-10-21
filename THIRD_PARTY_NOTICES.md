# Third-party notices

- **CrystalFormer-CSP**: the fixed-formula sampling entry point, its imported Python modules, Wyckoff tables and `scripts/awl2struct.py` are included under `vendor/CrystalFormer-CSP`. Source files are copied from the locally tested author distribution. Its Apache-2.0 license is preserved at `vendor/CrystalFormer-CSP/LICENSE`; upstream documentation is preserved. Unused CLI, reinforcement-learning, extensions and postprocessing tools are omitted. The included sampling algorithms are unchanged.
- **CrysVCD elemental language model**: `crystal_pipeline/modules/formula_generation.py` incorporates the electronic-configuration representation and GPT2-based formula generator from CrysVCD. The upstream MIT license is preserved at `third_party/licenses/CrysVCD-MIT.txt`.
- Model weights and the user-supplied synthesis database are not distributed in this code-only repository. Original sizes, hashes and available source information are recorded in `runtime_assets.json`.

Third-party components retain their respective licenses. No additional license grant is made here for the project's original integration code.
