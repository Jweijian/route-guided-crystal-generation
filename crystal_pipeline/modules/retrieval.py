import json
from pathlib import Path

import numpy as np
from pymatgen.core import Composition, Element
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from .chemistry import validate_target, composition_key
from .route_language import route_to_sentence


_BOOL_FEAT_INDICES = [11, 12, 13]

def _safe(x):
    """None → NaN，便于后续 impute。"""
    return np.nan if x is None else float(x)


def element_physical_features(elem: Element) -> np.ndarray:
    """单个元素 → 14 维物理特征向量。"""
    common_ox = elem.common_oxidation_states or ()
    if len(common_ox) == 0:
        common_ox = (0.0,)

    # 优先用 atomic_radius_calculated，兼容新版 pymatgen
    radius = getattr(elem, 'atomic_radius_calculated', None) or elem.atomic_radius

    is_metal = float(elem.is_metal)
    is_metalloid = float(elem.is_metalloid)
    is_nonmetal = float(not elem.is_metal and not elem.is_metalloid)

    feats = np.array([
        _safe(elem.Z),                                      # 0  原子序数
        _safe(elem.X),                                      # 1  电负性
        _safe(elem.atomic_mass),                            # 2  原子质量
        _safe(elem.row),                                    # 3  周期
        _safe(elem.group),                                  # 4  族 (f-区可能 None)
        _safe(radius),                                      # 5  原子半径
        _safe(elem.average_ionic_radius),                   # 6  平均离子半径
        _safe(elem.mendeleev_no),                           # 7  门捷列夫序数
        float(len(common_ox)),                              # 8  常见氧化态个数
        float(np.mean(np.abs(common_ox))),                  # 9  氧化态绝对值均值
        float(np.max(np.abs(common_ox))),                   # 10 氧化态绝对值最大值
        is_metal,                                            # 11 是否金属
        is_metalloid,                                        # 12 是否类金属
        is_nonmetal,                                         # 13 是否非金属
    ], dtype=float)
    return feats


class ChemicalSimilarityEncoder:
    """
    将化学式编码为物理化学向量，计算 cosine similarity。
    fit() 时缓存语料库向量以便后续最近邻查询。
    """
    def __init__(self):
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dim = None
        # 缓存
        self.corpus_formulas = None
        self.corpus_vectors = None
        self.element_features = {}

    # ----- 内部 -----
    def _composition_raw_vector(self, formula: str) -> np.ndarray:
        comp = Composition(formula)
        el_amt = comp.get_el_amt_dict()
        elems = list(el_amt.keys())
        fracs = np.array(list(el_amt.values()), dtype=float)
        fracs /= fracs.sum()

        for element in elems:
            if element not in self.element_features:
                self.element_features[element] = element_physical_features(Element(element))
        elem_feats = np.array([self.element_features[el] for el in elems], dtype=float)

        # 计量加权均值
        mean_feat = np.sum(elem_feats * fracs[:, None], axis=0)

        # 计量加权标准差 —— 排除布尔特征列
        continuous_mask = np.ones(elem_feats.shape[1], dtype=bool)
        continuous_mask[_BOOL_FEAT_INDICES] = False
        elem_cont = elem_feats[:, continuous_mask]
        mean_cont = mean_feat[continuous_mask]
        var_cont = np.sum(((elem_cont - mean_cont) ** 2) * fracs[:, None], axis=0)
        std_cont = np.sqrt(var_cont)

        # 组成统计量
        n_elements = len(elems)
        entropy = -np.sum(fracs * np.log(fracs + 1e-12))
        max_frac = np.max(fracs)
        min_frac = np.min(fracs)
        frac_range = max_frac - min_frac
        frac_l2 = np.sum(fracs ** 2)

        raw_vec = np.concatenate([
            mean_feat,          # 14 维
            std_cont,           # 11 维 (排除了 3 个布尔列)
            np.array([
                n_elements,
                entropy,
                max_frac,
                min_frac,
                frac_range,
                frac_l2,
            ], dtype=float),    # 6 维
        ])
        return raw_vec          # 共 31 维

    # ----- 公开 API -----
    def fit(self, formulas):
        """用化学式集合 fit imputer & scaler，并缓存编码矩阵。"""
        formulas = list(formulas)
        X = np.array([self._composition_raw_vector(f) for f in formulas], dtype=float)
        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)
        self.feature_dim = X.shape[1]
        self.is_fitted = True
        self.corpus_formulas = formulas
        self.corpus_vectors = X
        return self

    def encode(self, formula: str) -> np.ndarray:
        """单个化学式 → 标准化向量。"""
        if not self.is_fitted:
            raise RuntimeError('Encoder not fitted.')
        x = self._composition_raw_vector(formula).reshape(1, -1)
        x = self.imputer.transform(x)
        x = self.scaler.transform(x)
        return x.flatten()

    def top_k_similar(self, formula: str, k: int = 10):
        """在语料库中找 top-k 化学相似材料。返回 (indices, similarities)。"""
        z = self.encode(formula).reshape(1, -1)
        sims = cosine_similarity(z, self.corpus_vectors).flatten()
        top_idx = np.argsort(-sims)[:k]
        return top_idx, sims[top_idx]


class ChemicalSimilaritySearcher:
    def __init__(self, db_json_path: str):
            """
            初始化搜索器并加载数据库建立特征空间
            
            Args:
                db_json_path: JSON 数据库文件的路径
            """
            self.db_path = Path(db_json_path)
            self.db_records = []
            self.encoder = ChemicalSimilarityEncoder()
            
            self._load_database_and_fit()

    def _load_database_and_fit(self):
            """内部方法：加载数据库并训练编码器"""
            if not self.db_path.exists():
                raise FileNotFoundError(f"Database file not found: {self.db_path}")
                
            with open(self.db_path, 'r', encoding='utf-8') as f:
                raw_records = json.load(f)
                
            # 过滤出有实际合成方法的记录
            self.db_records = [
                r for r in raw_records 
                if r.get('synthesis_method') and r.get('synthesis_method') != 'Other'
            ]
            
            db_formulas = [rec['formula'] for rec in self.db_records]
            compositions = {formula: Composition(formula) for formula in set(db_formulas)}
            keys = {formula: composition_key(comp) for formula, comp in compositions.items()}
            elements = {formula: frozenset(comp.get_el_amt_dict()) for formula, comp in compositions.items()}
            self.composition_keys = [keys[formula] for formula in db_formulas]
            self.element_sets = [elements[formula] for formula in db_formulas]
            
            # 训练编码器
            self.encoder.fit(db_formulas)

class Retriever:
    def __init__(self, database_path):
        self.searcher = ChemicalSimilaritySearcher(str(database_path))

    def retrieve(self, formula, oxidation_states, top_n=50, limit=15):
            """合并三路召回，保留原始记录，并按阴离子与方法选择多样化参考。"""
            target = validate_target(formula, oxidation_states)
            comp = Composition(target["formula"])
            target_key = composition_key(comp)
            amounts = comp.get_el_amt_dict()
            anions = {e for e, v in oxidation_states.items() if v < 0}
            variant_keys = set()
            # Generic isovalent anion replacement: no target-specific material rules.
            for old in anions:
                for new in ("N", "P", "O", "S", "Se", "Te", "F", "Cl", "Br", "I"):
                    if (
                        new in amounts
                        or oxidation_states[old] not in Element(new).common_oxidation_states
                    ):
                        continue
                    variant = {e: a for e, a in amounts.items() if e != old}
                    variant[new] = amounts[old]
                    variant_keys.add(composition_key(Composition(variant)))
            indices, similarities = self.searcher.encoder.top_k_similar(
                target["formula"], k=len(self.searcher.db_records)
            )
            similarity_by_index = dict(zip(indices.tolist(), similarities.tolist()))
            nearest = set(indices[:top_n].tolist())
            pool = []
            for i, record in enumerate(self.searcher.db_records):
                key = self.searcher.composition_keys[i]
                channels = []
                if key == target_key:
                    channels.append("exact_composition")
                if i in nearest:
                    channels.append("chemical_similarity")
                if key in variant_keys:
                    channels.append("isovalent_anion_analogy")
                if not channels:
                    continue
                pool.append(
                    {
                        "reference_id": f"DB{i:05d}",
                        "formula": record["formula"],
                        "retrieval_channel": channels,
                        "chemical_similarity": similarity_by_index[i],
                        "same_anion": bool(
                            anions & self.searcher.element_sets[i]
                        ),
                        "structure_description_provenance": "database_reference_not_verified_route_phase",
                        "route_sentence": route_to_sentence(record),
                        "record": record,
                    }
                )
            pool.sort(key=lambda r: (-r["chemical_similarity"], r["reference_id"]))
            chosen = []
            chosen_ids = set()
    
            def add(item):
                if item is not None and item["reference_id"] not in chosen_ids and len(chosen) < limit:
                    chosen.append(item)
                    chosen_ids.add(item["reference_id"])
                return len(chosen) >= limit
    
            for predicate in [lambda r: "exact_composition" in r["retrieval_channel"],
                              lambda r: "isovalent_anion_analogy" in r["retrieval_channel"],
                              lambda r: r["same_anion"]]:
                item = next((r for r in pool if r["reference_id"] not in chosen_ids and predicate(r)), None)
                if add(item):
                    return chosen
            # Prioritize same-anion chemistry; retain method diversity within each channel.
            for subset, budget in [
                ([r for r in pool if r["same_anion"]], 8),
                ([r for r in pool if "isovalent_anion_analogy" in r["retrieval_channel"]], 3),
            ]:
                methods = set()
                for item in subset:
                    method = item["record"]["synthesis_method"].strip().lower()
                    if method not in methods and len(methods) < budget:
                        if add(item):
                            return chosen
                        methods.add(method)
                for item in subset[:budget]:
                    if add(item):
                        return chosen
            for item in pool:
                if add(item):
                    return chosen
            return chosen
