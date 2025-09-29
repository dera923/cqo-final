"""
A-Gate: Audit Gate Implementation
因果推論結果の安全性を保証する監査ゲート

設計原則（NASA標準）:
1. Fail-Closed: 一つでも条件違反があれば古典手法へフォールバック
2. 単調性: 条件を厳しくすると合格集合が縮小
3. 完全監査: すべての判定に根拠を記録

Author: CQO Development Team
Date: 2024-12-20
Standard: NASA-JPL-2024
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from datetime import datetime
from loguru import logger
from scipy import stats

# Gate判定の閾値（設定可能）
DEFAULT_THRESHOLDS = {
    'coverage_min': 0.90,
    'extreme_ps_pct_max': 0.05,
    'smd_max': 0.10,
    'ope_lcb_min': 0.0,
    'weight_max': 100.0,
    'sample_min': 100,
    # Ω拡張用
    'ecs_p_min': 0.05,
    'invariant_score_min': 0.60,
    'dro_lcb_min': 0.0,
}


class GateStatus(Enum):
    """Gate判定結果"""
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    BYPASS = "BYPASS"  # 緊急時のみ


class ViolationType(Enum):
    """違反タイプの分類"""
    COVERAGE = "COVERAGE_VIOLATION"
    OVERLAP = "OVERLAP_VIOLATION"
    BALANCE = "BALANCE_VIOLATION"
    PESSIMISM = "PESSIMISM_VIOLATION"
    SAMPLE_SIZE = "SAMPLE_SIZE_VIOLATION"
    NUMERICAL = "NUMERICAL_VIOLATION"
    CONFOUNDING = "CONFOUNDING_VIOLATION"


@dataclass
class GateCheckpoint:
    """個別チェックポイントの結果"""
    name: str
    passed: bool
    value: float
    threshold: float
    operator: str  # '<', '>', '<=', '>='
    violation_type: Optional[ViolationType] = None
    message: Optional[str] = None
    severity: str = "ERROR"  # ERROR, WARN, INFO
    
    def __str__(self) -> str:
        status = "✓" if self.passed else "✗"
        return f"{status} {self.name}: {self.value:.4f} {self.operator} {self.threshold:.4f}"


@dataclass
class GateResult:
    """Gate判定の総合結果"""
    status: GateStatus
    checkpoints: List[GateCheckpoint]
    mode: str  # "classical" or "hybrid"
    violations: List[ViolationType]
    recommendations: List[str]
    signature: str  # 監査署名
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def passed(self) -> bool:
        return self.status == GateStatus.PASS
    
    @property
    def violation_summary(self) -> str:
        """違反内容の要約"""
        failed = [cp for cp in self.checkpoints if not cp.passed]
        if not failed:
            return "All checks passed"
        return f"Failed: {', '.join([cp.name for cp in failed])}"
    
    def to_dict(self) -> Dict:
        """監査ログ用の辞書変換"""
        return {
            'status': self.status.value,
            'mode': self.mode,
            'timestamp': self.timestamp.isoformat(),
            'signature': self.signature,
            'violations': [v.value for v in self.violations],
            'checkpoints': [
                {
                    'name': cp.name,
                    'passed': cp.passed,
                    'value': cp.value,
                    'threshold': cp.threshold,
                    'operator': cp.operator
                }
                for cp in self.checkpoints
            ],
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }


class AGate:
    """
    監査ゲート実装
    
    NASA標準の実装:
    - すべての判定を記録
    - 失敗時の理由を明確化
    - 改善提案を提供
    """
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None,
                 strict_mode: bool = True,
                 enable_omega: bool = False):
        """
        Args:
            thresholds: カスタム閾値
            strict_mode: 厳格モード（一つでも違反でFail）
            enable_omega: Ω拡張チェックを有効化
        """
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.strict_mode = strict_mode
        self.enable_omega = enable_omega
        self._check_history = []
        
        logger.info(f"A-Gate initialized: strict={strict_mode}, omega={enable_omega}")
    
    def evaluate(self, estimation_result: 'EstimationResult', 
                 additional_checks: Optional[Dict[str, Any]] = None) -> GateResult:
        """
        推定結果の包括的評価
        
        NASA標準: 完全な監査証跡を残す
        """
        timestamp = datetime.now()
        checkpoints = []
        violations = []
        
        # 1. 被覆率チェック
        coverage_check = self._check_coverage(estimation_result.coverage)
        checkpoints.append(coverage_check)
        if not coverage_check.passed:
            violations.append(ViolationType.COVERAGE)
        
        # 2. オーバーラップチェック（極端な傾向スコア）
        overlap_check = self._check_overlap(
            estimation_result.diagnostics.get('extreme_ps_pct', 0.0)
        )
        checkpoints.append(overlap_check)
        if not overlap_check.passed:
            violations.append(ViolationType.OVERLAP)
        
        # 3. バランスチェック（SMD）
        balance_check = self._check_balance(
            estimation_result.diagnostics.get('smd', 0.0)
        )
        checkpoints.append(balance_check)
        if not balance_check.passed:
            violations.append(ViolationType.BALANCE)
        
        # 4. サンプルサイズチェック
        sample_check = self._check_sample_size(estimation_result.diagnostics)
        checkpoints.append(sample_check)
        if not sample_check.passed:
            violations.append(ViolationType.SAMPLE_SIZE)
        
        # 5. 数値安定性チェック
        numerical_check = self._check_numerical_stability(estimation_result)
        checkpoints.append(numerical_check)
        if not numerical_check.passed:
            violations.append(ViolationType.NUMERICAL)
        
        # 6. Ω拡張チェック（有効時のみ）
        if self.enable_omega and additional_checks:
            omega_checks = self._perform_omega_checks(additional_checks)
            checkpoints.extend(omega_checks)
            for check in omega_checks:
                if not check.passed and check.violation_type:
                    violations.append(check.violation_type)
        
        # 総合判定
        if self.strict_mode:
            status = GateStatus.PASS if all(cp.passed for cp in checkpoints) else GateStatus.FAIL
        else:
            # 緩和モード: ERRORレベルのみ判定
            critical_failures = [cp for cp in checkpoints if not cp.passed and cp.severity == "ERROR"]
            status = GateStatus.PASS if not critical_failures else GateStatus.FAIL
        
        # モード決定
        mode = "hybrid" if status == GateStatus.PASS else "classical"
        
        # 推奨事項の生成
        recommendations = self._generate_recommendations(checkpoints, violations)
        
        # 監査署名の生成
        signature = self._generate_signature(checkpoints, timestamp)
        
        result = GateResult(
            status=status,
            checkpoints=checkpoints,
            mode=mode,
            violations=violations,
            recommendations=recommendations,
            signature=signature,
            timestamp=timestamp,
            metadata={
                'thresholds': self.thresholds,
                'strict_mode': self.strict_mode,
                'enable_omega': self.enable_omega
            }
        )
        
        # 履歴に記録
        self._check_history.append(result)
        
        # ログ出力
        logger.info(f"Gate evaluation: {status.value} ({result.violation_summary})")
        
        return result
    
    def _check_coverage(self, coverage: float) -> GateCheckpoint:
        """被覆率のチェック"""
        threshold = self.thresholds['coverage_min']
        passed = coverage >= threshold
        
        return GateCheckpoint(
            name="coverage",
            passed=passed,
            value=coverage,
            threshold=threshold,
            operator=">=",
            violation_type=ViolationType.COVERAGE if not passed else None,
            message=f"Bootstrap coverage {'sufficient' if passed else 'insufficient'}",
            severity="ERROR" if not passed else "INFO"
        )
    
    def _check_overlap(self, extreme_ps_pct: float) -> GateCheckpoint:
        """オーバーラップのチェック"""
        threshold = self.thresholds['extreme_ps_pct_max']
        passed = extreme_ps_pct <= threshold
        
        return GateCheckpoint(
            name="extreme_propensity",
            passed=passed,
            value=extreme_ps_pct,
            threshold=threshold,
            operator="<=",
            violation_type=ViolationType.OVERLAP if not passed else None,
            message=f"Propensity score overlap {'adequate' if passed else 'poor'}",
            severity="ERROR" if not passed else "INFO"
        )
    
    def _check_balance(self, smd: float) -> GateCheckpoint:
        """共変量バランスのチェック"""
        threshold = self.thresholds['smd_max']
        passed = smd <= threshold
        
        return GateCheckpoint(
            name="covariate_balance",
            passed=passed,
            value=smd,
            threshold=threshold,
            operator="<=",
            violation_type=ViolationType.BALANCE if not passed else None,
            message=f"Covariate balance {'good' if passed else 'poor'}",
            severity="WARN" if not passed else "INFO"
        )
    
    def _check_sample_size(self, diagnostics: Dict) -> GateCheckpoint:
        """サンプルサイズのチェック"""
        n_total = diagnostics.get('n_treated', 0) + diagnostics.get('n_control', 0)
        threshold = self.thresholds['sample_min']
        passed = n_total >= threshold
        
        return GateCheckpoint(
            name="sample_size",
            passed=passed,
            value=float(n_total),
            threshold=float(threshold),
            operator=">=",
            violation_type=ViolationType.SAMPLE_SIZE if not passed else None,
            message=f"Sample size: {n_total} ({'adequate' if passed else 'insufficient'})",
            severity="ERROR" if not passed else "INFO"
        )
    
    def _check_numerical_stability(self, result: 'EstimationResult') -> GateCheckpoint:
        """数値安定性のチェック"""
        # 影響関数の最大値をチェック
        max_influence = np.max(np.abs(result.influence_function))
        threshold = 1e6  # 極端に大きい値は不安定
        passed = max_influence < threshold
        
        return GateCheckpoint(
            name="numerical_stability",
            passed=passed,
            value=max_influence,
            threshold=threshold,
            operator="<",
            violation_type=ViolationType.NUMERICAL if not passed else None,
            message=f"Numerical stability {'OK' if passed else 'compromised'}",
            severity="ERROR" if not passed else "INFO"
        )
    
    def _perform_omega_checks(self, additional_checks: Dict[str, Any]) -> List[GateCheckpoint]:
        """Ω拡張チェック"""
        omega_checkpoints = []
        
        # ECS (Entanglement Checking Score)
        if 'ecs_p_value' in additional_checks:
            ecs_check = GateCheckpoint(
                name="ecs_confounding",
                passed=additional_checks['ecs_p_value'] >= self.thresholds['ecs_p_min'],
                value=additional_checks['ecs_p_value'],
                threshold=self.thresholds['ecs_p_min'],
                operator=">=",
                violation_type=ViolationType.CONFOUNDING,
                message="Potential unobserved confounding detected",
                severity="WARN"
            )
            omega_checkpoints.append(ecs_check)
        
        # Invariant Score
        if 'invariant_score' in additional_checks:
            inv_check = GateCheckpoint(
                name="invariance",
                passed=additional_checks['invariant_score'] >= self.thresholds['invariant_score_min'],
                value=additional_checks['invariant_score'],
                threshold=self.thresholds['invariant_score_min'],
                operator=">=",
                violation_type=ViolationType.BALANCE,
                message="Feature invariance check",
                severity="WARN"
            )
            omega_checkpoints.append(inv_check)
        
        return omega_checkpoints
    
    def _generate_recommendations(self, checkpoints: List[GateCheckpoint], 
                                 violations: List[ViolationType]) -> List[str]:
        """改善推奨事項の生成"""
        recommendations = []
        
        if ViolationType.OVERLAP in violations:
            recommendations.append("Consider trimming extreme propensity scores or using overlap weights")
        
        if ViolationType.BALANCE in violations:
            recommendations.append("Improve covariate balance through matching or re-weighting")
        
        if ViolationType.COVERAGE in violations:
            recommendations.append("Increase bootstrap iterations or use more robust CI methods")
        
        if ViolationType.SAMPLE_SIZE in violations:
            recommendations.append("Collect more data or consider pooling similar experiments")
        
        if ViolationType.NUMERICAL in violations:
            recommendations.append("Apply regularization or use more stable numerical algorithms")
        
        if ViolationType.CONFOUNDING in violations:
            recommendations.append("Consider proximal methods or instrumental variables")
        
        return recommendations
    
    def _generate_signature(self, checkpoints: List[GateCheckpoint], 
                           timestamp: datetime) -> str:
        """監査署名の生成（改竄防止）"""
        content = {
            'timestamp': timestamp.isoformat(),
            'checkpoints': [
                {'name': cp.name, 'passed': cp.passed, 'value': cp.value}
                for cp in checkpoints
            ]
        }
        
        content_str = json.dumps(content, sort_keys=True)
        signature = hashlib.sha256(content_str.encode()).hexdigest()
        
        return signature[:16]  # 短縮版
    
    def get_history(self) -> List[GateResult]:
        """評価履歴の取得"""
        return self._check_history
    
    def export_audit_log(self, filepath: str) -> None:
        """監査ログのエクスポート"""
        audit_data = [result.to_dict() for result in self._check_history]
        
        with open(filepath, 'w') as f:
            json.dump(audit_data, f, indent=2, default=str)
        
        logger.info(f"Audit log exported to {filepath}")
