"""命令行入口，所有自编提示均使用中文。"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import platform
import sys
from pathlib import Path

from anitopy_ml import __version__
from anitopy_ml.annotation.seed import seed_from_jsonl
from anitopy_ml.annotation.quality import build_quality_report, write_quality_report
from anitopy_ml.annotation.export import (
    audit_effective_work_groups,
    build_training_splits,
    read_split_assignments,
    read_work_group_ids,
    write_training_splits,
)
from anitopy_ml.annotation.review import review_sample
from anitopy_ml.annotation.store import AnnotationStore
from anitopy_ml.api import MediaParser
from anitopy_ml.data.baseline import read_jsonl_records, write_legacy_baseline
from anitopy_ml.data.grouping import (
    build_group_assignments,
    find_review_candidates,
    write_review_candidates,
    write_split_outputs,
)
from anitopy_ml.data.ingest import ingest_csv, profile_records, write_jsonl
from anitopy_ml.data.manifest import build_manifest, write_manifest
from anitopy_ml.data.synthetic import (
    audit_synthetic_records,
    deduplicate_synthetic_records,
    generate_synthetic_records,
    load_template_data,
    split_synthetic_records,
    write_synthetic_records,
    write_synthetic_splits,
)
from anitopy_ml.errors import AnitopyMlError, ConfigurationError, InputValidationError
from anitopy_ml.legacy import compare_legacy_baseline
from anitopy_ml.messages import message_for


def build_parser() -> argparse.ArgumentParser:
    """构造项目顶层命令与已实现子命令。"""
    parser = argparse.ArgumentParser(
        prog="anitopy-ml",
        description="多语言媒体标题语义解析工具。",
    )
    parser.add_argument("--version", action="version", version=f"anitopy-ml {__version__}")
    subparsers = parser.add_subparsers(dest="command", title="可用命令")

    doctor = subparsers.add_parser("doctor", help="检查运行环境和可选组件。")
    doctor.add_argument("--json", action="store_true", help="以JSON格式输出检查结果。")

    data = subparsers.add_parser("data", help="导入并统计标题数据。")
    data_commands = data.add_subparsers(dest="data_command", title="数据命令")
    ingest = data_commands.add_parser("ingest", help="将CSV导入为可追溯的JSONL。")
    ingest.add_argument("--input", required=True, help="输入CSV文件路径。")
    ingest.add_argument("--column", default="record_title", help="标题列名称。")
    ingest.add_argument("--output", required=True, help="输出JSONL文件路径。")
    ingest.add_argument("--manifest", help="可选的数据清单JSON输出路径。")
    profile = data_commands.add_parser("profile", help="生成CSV标题数据概况。")
    profile.add_argument("--input", required=True, help="输入CSV文件路径。")
    profile.add_argument("--column", default="record_title", help="标题列名称。")
    profile.add_argument("--output", help="可选的JSON报告输出路径。")
    split = data_commands.add_parser("split", help="按作品组稳定划分训练、验证和测试集。")
    split.add_argument("--input", required=True, help="导入JSONL文件路径。")
    split.add_argument("--output", required=True, help="分区JSONL输出路径。")
    split.add_argument("--report", required=True, help="数据划分报告JSON路径。")
    split.add_argument("--seed", type=int, default=20260917, help="稳定划分随机种子。")
    split.add_argument("--database", help="可选SQLite标注库，用于读取人工作品组覆盖。")
    group_audit = data_commands.add_parser("group-audit", help="生成疑似同组的人工复核队列。")
    group_audit.add_argument("--input", required=True, help="导入JSONL文件路径。")
    group_audit.add_argument("--output", required=True, help="候选JSONL输出路径。")
    group_audit.add_argument("--report", required=True, help="候选统计报告JSON路径。")
    group_audit.add_argument("--seed", type=int, default=20260917, help="与冻结分区一致的随机种子。")
    group_audit.add_argument("--database", help="可选SQLite标注库，用于读取人工作品组覆盖。")
    group_audit.add_argument("--similarity-threshold", type=float, default=0.98, help="近重复复核阈值，默认0.98。")
    group_audit.add_argument("--max-similarity-candidates", type=int, default=500, help="近重复候选输出上限，默认500。")
    synthetic = data_commands.add_parser("generate-synthetic", help="按名称模板生成仅供训练的合成BIO数据。")
    synthetic.add_argument("--input", required=True, help="模板数据JSON路径。")
    synthetic.add_argument("--output", required=True, help="合成训练JSONL输出路径。")
    synthetic.add_argument("--count", type=int, default=10000, help="生成样本数量，默认10000。")
    synthetic.add_argument("--seed", type=int, default=20260918, help="随机种子。")
    synthetic.add_argument("--optional-rate", type=float, default=1.0, help="&可选字段的填充比例，默认全部填充。")
    synthetic.add_argument("--season-rate", type=float, default=0.7, help="含season_expr标题的抽样比例，默认0.7。")
    split_synthetic = data_commands.add_parser("split-synthetic", help="按主标题隔离模板训练、验证和测试数据。")
    split_synthetic.add_argument("--input", required=True, help="模板训练JSONL路径。")
    split_synthetic.add_argument("--output-dir", required=True, help="三个分区及清单输出目录。")
    split_synthetic.add_argument("--seed", type=int, default=20260918, help="稳定划分随机种子。")
    audit_synthetic = data_commands.add_parser("audit-synthetic", help="审计模板覆盖、BIO一致性和跨批重复。")
    audit_synthetic.add_argument("--input", required=True, nargs="+", help="一个或多个合成JSONL文件路径。")
    audit_synthetic.add_argument("--output", required=True, help="审计JSON报告路径。")
    dedupe_synthetic = data_commands.add_parser("dedupe-synthetic", help="按文本和BIO标签去重多个合成批次。")
    dedupe_synthetic.add_argument("--input", required=True, nargs="+", help="一个或多个合成JSONL文件路径。")
    dedupe_synthetic.add_argument("--output", required=True, help="去重后的合成JSONL路径。")

    baseline = subparsers.add_parser("baseline", help="比较旧夹具或批量生成旧解析结果。")
    baseline.add_argument("--legacy-source", required=True, help="旧Anitopy项目根目录。")
    baseline.add_argument("--output", required=True, help="基线报告JSON文件路径。")
    baseline.add_argument("--input", help="可选的导入JSONL；指定后逐条输出旧解析结果。")

    annotate = subparsers.add_parser("annotate", help="生成和审核标题标注。")
    annotate_commands = annotate.add_subparsers(dest="annotate_command", title="标注命令")
    seed = annotate_commands.add_parser("seed", help="生成待审核的规则弱标注。")
    seed.add_argument("--input", required=True, help="输入JSONL文件路径。")
    seed.add_argument("--database", required=True, help="SQLite标注库路径。")
    seed.add_argument("--limit", type=int, help="最多处理的标题数量。")
    review = annotate_commands.add_parser("review", help="列出或处理待审核标注。")
    review.add_argument("--database", required=True, help="SQLite标注库路径。")
    review.add_argument("--list", action="store_true", help="列出待审核样本ID。")
    review.add_argument("--limit", type=int, default=20, help="列出样本数量上限。")
    review.add_argument("--sample-id", help="要审核的样本ID。")
    decision = review.add_mutually_exclusive_group()
    decision.add_argument("--approve", action="store_true", help="人工审核通过并升级标注。")
    decision.add_argument("--reject", action="store_true", help="人工审核拒绝并保留审计记录。")
    review.add_argument("--tier", choices=("gold", "silver"), default="gold", help="通过后的标注层级。")
    review.add_argument("--actor", help="审核人员标识。")
    quality = annotate_commands.add_parser("report", help="生成最新标注版本的质量报告。")
    quality.add_argument("--database", required=True, help="SQLite标注库路径。")
    quality.add_argument("--output", required=True, help="Markdown报告输出路径。")
    export_training = annotate_commands.add_parser("export-training", help="导出已审核的字符级BIO训练数据。")
    export_training.add_argument("--database", required=True, help="SQLite标注库路径。")
    export_training.add_argument("--splits", required=True, help="冻结分区JSONL路径。")
    export_training.add_argument("--output-dir", required=True, help="训练数据输出目录。")

    train = subparsers.add_parser("train", help="训练本地语义抽取模型。")
    train_commands = train.add_subparsers(dest="train_command", title="训练命令")
    extractor = train_commands.add_parser("extractor", help="离线训练XLM-R BIO抽取模型。")
    extractor.add_argument("--data-dir", required=True, help="含训练、验证和清单的模板分区目录。")
    extractor.add_argument("--output-dir", required=True, help="模型检查点与训练报告目录。")
    extractor.add_argument("--config", default="configs/抽取训练.json", help="中文训练配置JSON路径。")
    extractor.add_argument("--epochs", type=int, default=8, help="最大训练轮数。")
    extractor.add_argument("--batch-size", type=int, default=4, help="每张设备的批大小。")
    extractor.add_argument("--max-length", type=int, default=256, help="分词后的最大长度。")
    extractor.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="训练设备。")
    extractor.add_argument("--augment-whitespace", action="store_true", help="仅对训练分区追加空白增强样本。")
    extractor.add_argument("--seed", type=int, default=20260918, help="训练随机种子。")
    compare_runs = train_commands.add_parser("compare-runs", help="汇总固定配置下的多随机种子实验。")
    compare_runs.add_argument("--runs", nargs="+", required=True, help="各训练输出目录。")
    compare_runs.add_argument("--output", required=True, help="稳定性报告JSON输出路径。")
    calibrate = subparsers.add_parser("calibrate", help="只用验证集校准本地模型置信度。")
    calibrate.add_argument("--model", required=True, help="待校准的本地模型目录。")
    calibrate.add_argument("--validation", required=True, help="字符级BIO验证集JSONL路径。")
    calibrate.add_argument("--output", required=True, help="校准文件JSON输出路径。")
    calibrate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="校准推理设备。")
    calibrate.add_argument("--bucket-count", type=int, default=10, help="经验校准分桶数。")
    calibrate.add_argument("--minimum-samples", type=int, default=20, help="字段最小验证预测数。")
    policy = subparsers.add_parser("calibration-policy", help="从校准报告生成模型接受策略。")
    policy.add_argument("--calibration", required=True, help="校准文件JSON路径。")
    policy.add_argument("--output", required=True, help="接受策略JSON输出路径。")
    policy.add_argument("--maximum-error-rate", type=float, default=0.05, help="允许的最大验证集经验错误率。")
    evaluate = subparsers.add_parser("evaluate", help="评测固定模型在冻结测试分区上的效果。")
    evaluate.add_argument("--model", required=True, help="已固定并完成校准的本地模型目录。")
    evaluate.add_argument("--test", required=True, help="冻结测试JSONL路径。")
    evaluate.add_argument("--output", required=True, help="评测报告JSON输出路径。")
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="评测推理设备。")

    parse_command = subparsers.add_parser("parse", help="使用本地模型解析单条标题。")
    parse_command.add_argument("title", help="要解析的原始标题。")
    parse_command.add_argument("--model", required=True, help="本地模型目录或字符模型文件。")
    parse_command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="推理设备。")

    batch = subparsers.add_parser("batch", help="使用本地模型按顺序批量解析CSV标题。")
    batch.add_argument("--input", required=True, help="输入CSV文件路径。")
    batch.add_argument("--column", default="record_title", help="标题列名称。")
    batch.add_argument("--model", required=True, help="本地模型目录或字符模型文件。")
    batch.add_argument("--output", required=True, help="JSONL结果输出路径。")
    batch.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="推理设备。")
    batch.add_argument("--on-error", choices=("raise", "record"), default="record", help="单条错误处理策略。")
    return parser


def collect_doctor_report() -> dict[str, object]:
    """收集不含凭证的环境信息。"""
    optional_packages = {
        "tmdb": ("httpx",),
        "train": ("torch", "transformers"),
        "review": ("streamlit",),
        "serve": ("fastapi",),
    }
    available = {
        group: all(importlib.util.find_spec(name) is not None for name in names)
        for group, names in optional_packages.items()
    }
    package_root = Path(__file__).resolve().parent
    return {
        "项目版本": __version__,
        "Python版本": platform.python_version(),
        "Python路径": sys.executable,
        "操作系统": platform.platform(),
        "项目包路径": str(package_root),
        "可选组件": available,
        "说明": "检查结果不会读取或显示任何访问凭证。",
    }


def print_doctor_report(report: dict[str, object]) -> None:
    """以便于终端阅读的中文形式输出环境报告。"""
    print("环境检查结果")
    for key, value in report.items():
        if key == "可选组件":
            print("可选组件：")
            for group, installed in value.items():
                state = "已安装" if installed else "未安装"
                print(f"  - {group}: {state}")
        else:
            print(f"{key}：{value}")


def main(argv: list[str] | None = None) -> int:
    """执行命令行并将可预期错误转换为中文提示。"""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "doctor":
            report = collect_doctor_report()
            if arguments.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print_doctor_report(report)
            return 0
        if arguments.command == "data" and arguments.data_command == "ingest":
            records = ingest_csv(arguments.input, column=arguments.column)
            count = write_jsonl(records, arguments.output)
            manifest_path = arguments.manifest
            if manifest_path:
                write_manifest(
                    build_manifest(records, source_path=arguments.input, title_column=arguments.column),
                    manifest_path,
                )
            print(f"导入完成：已写入 {count} 条标题到 {arguments.output}。")
            if manifest_path:
                print(f"数据清单已写入：{manifest_path}。")
            return 0
        if arguments.command == "data" and arguments.data_command == "profile":
            report = profile_records(ingest_csv(arguments.input, column=arguments.column))
            content = json.dumps(report, ensure_ascii=False, indent=2)
            print(content)
            if arguments.output:
                target = Path(arguments.output)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content + "\n", encoding="utf-8")
                print(f"数据概况已写入：{target}。")
            return 0
        if arguments.command == "data" and arguments.data_command == "split":
            records = list(read_jsonl_records(arguments.input))
            if not records:
                raise InputValidationError("输入JSONL不包含可划分的标题。")
            overrides: dict[str, str] = {}
            if arguments.database:
                with AnnotationStore(arguments.database) as store:
                    overrides = store.work_group_overrides()
            assignments, report = build_group_assignments(
                records,
                seed=arguments.seed,
                overrides=overrides,
            )
            write_split_outputs(
                assignments,
                report,
                output=arguments.output,
                report_output=arguments.report,
                source_sha256=records[0].source_sha256,
            )
            print(
                f"数据划分完成：{report['样本数']} 条样本、{report['作品组数']} 个作品组，"
                f"已写入 {arguments.output}。"
            )
            return 0
        if arguments.command == "data" and arguments.data_command == "group-audit":
            records = list(read_jsonl_records(arguments.input))
            if not records:
                raise InputValidationError("输入JSONL不包含可审核的标题。")
            overrides: dict[str, str] = {}
            if arguments.database:
                with AnnotationStore(arguments.database) as store:
                    overrides = store.work_group_overrides()
            assignments, _ = build_group_assignments(records, seed=arguments.seed, overrides=overrides)
            count = write_review_candidates(
                find_review_candidates(
                    records,
                    assignments,
                    similarity_threshold=arguments.similarity_threshold,
                    max_similarity_candidates=arguments.max_similarity_candidates,
                ),
                output=arguments.output,
                report_output=arguments.report,
            )
            print(f"疑似同组复核队列已生成：{count} 个候选簇。")
            return 0
        if arguments.command == "data" and arguments.data_command == "generate-synthetic":
            records = generate_synthetic_records(
                load_template_data(arguments.input),
                count=arguments.count,
                seed=arguments.seed,
                optional_rate=arguments.optional_rate,
                season_rate=arguments.season_rate,
            )
            manifest = write_synthetic_records(records, arguments.output)
            print(
                f"合成训练数据已生成：{manifest['样本数']} 条，"
                f"仅可用于训练分区，清单已写入 {arguments.output}。"
            )
            return 0
        if arguments.command == "data" and arguments.data_command == "split-synthetic":
            with Path(arguments.input).open(encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            splits, report = split_synthetic_records(records, seed=arguments.seed)
            manifest = write_synthetic_splits(splits, report, arguments.output_dir)
            counts = manifest["分区样本数"]
            print(
                "模板数据划分完成："
                f"训练{counts['train']}条、验证{counts['validation']}条、测试{counts['test']}条。"
            )
            return 0
        if arguments.command == "data" and arguments.data_command == "audit-synthetic":
            records: list[dict[str, object]] = []
            for input_path in arguments.input:
                with Path(input_path).open(encoding="utf-8") as stream:
                    records.extend(json.loads(line) for line in stream if line.strip())
            report = audit_synthetic_records(records)
            target = Path(arguments.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                f"模板数据审计完成：{report['样本数']} 条样本、"
                f"重复样本ID {report['重复样本ID数']} 条、BIO不一致 {report['BIO不一致样本数']} 条。"
            )
            return 0
        if arguments.command == "data" and arguments.data_command == "dedupe-synthetic":
            records: list[dict[str, object]] = []
            for input_path in arguments.input:
                with Path(input_path).open(encoding="utf-8") as stream:
                    records.extend(json.loads(line) for line in stream if line.strip())
            kept, report = deduplicate_synthetic_records(records)
            manifest = write_synthetic_records(kept, arguments.output)
            print(
                f"合成数据去重完成：保留 {manifest['样本数']} 条、"
                f"重复样本ID {report['重复样本ID数']} 条、重复文本 {report['重复文本数']} 条。"
            )
            return 0
        if arguments.command == "baseline":
            if arguments.input:
                report = write_legacy_baseline(
                    read_jsonl_records(arguments.input),
                    legacy_source=arguments.legacy_source,
                    output=arguments.output,
                )
                print(
                    "旧解析器批量基线完成："
                    f"已写入 {report['total']} 条，异常 {report['exception_count']} 条。"
                )
                return 0
            report = compare_legacy_baseline(arguments.legacy_source)
            target = Path(arguments.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                "旧解析器基线完成："
                f"共比较 {report['total']} 条，完全匹配 {report['exact_match']} 条，"
                f"异常 {report['exception_count']} 条。"
            )
            return 0
        if arguments.command == "annotate" and arguments.annotate_command == "seed":
            if arguments.limit is not None and arguments.limit <= 0:
                raise AnitopyMlError("处理数量必须大于0。")
            count = seed_from_jsonl(arguments.input, arguments.database, limit=arguments.limit)
            print(f"规则预标注完成：已生成 {count} 条待审核弱标注。")
            return 0
        if arguments.command == "annotate" and arguments.annotate_command == "review":
            with AnnotationStore(arguments.database) as store:
                if arguments.list:
                    for sample_id in store.pending_sample_ids(limit=arguments.limit):
                        print(sample_id)
                    return 0
                if not arguments.sample_id or not (arguments.approve or arguments.reject) or not arguments.actor:
                    raise ConfigurationError("审核操作需要样本ID、审核人员和通过或拒绝决定。")
                decision = "approve" if arguments.approve else "reject"
                record = review_sample(
                    store,
                    arguments.sample_id,
                    actor=arguments.actor,
                    decision=decision,
                    tier=arguments.tier,
                )
                print(f"审核完成：样本 {record.sample_id} 当前版本为 {record.annotation.version}。")
                return 0
        if arguments.command == "annotate" and arguments.annotate_command == "report":
            with AnnotationStore(arguments.database) as store:
                report = build_quality_report(store.latest_payloads())
            write_quality_report(report, arguments.output)
            print(
                f"标注质量报告已写入：{arguments.output}；"
                f"可训练已接受样本 {report['可训练已接受样本数']} 条。"
            )
            return 0
        if arguments.command == "annotate" and arguments.annotate_command == "export-training":
            assignments = read_split_assignments(arguments.splits)
            work_group_ids = read_work_group_ids(arguments.splits)
            with AnnotationStore(arguments.database) as store:
                work_group_ids.update(store.work_group_overrides())
                audit_effective_work_groups(assignments, work_group_ids)
                splits, skipped = build_training_splits(
                    store.latest_payloads(),
                    assignments,
                    work_group_ids=work_group_ids,
                )
            manifest = write_training_splits(splits, skipped, output_dir=arguments.output_dir)
            total = sum(int(item["样本数"]) for item in manifest["分区"].values())
            print(f"训练数据导出完成：{total} 条合格样本，清单已写入 {arguments.output_dir}。")
            return 0
        if arguments.command == "train" and arguments.train_command == "extractor":
            from anitopy_ml.training.extractor import train_extractor

            report = train_extractor(
                data_dir=arguments.data_dir,
                output_dir=arguments.output_dir,
                training_config_path=arguments.config,
                epochs=arguments.epochs,
                batch_size=arguments.batch_size,
                max_length=arguments.max_length,
                device_name=arguments.device,
                augment_whitespace=arguments.augment_whitespace,
                seed=arguments.seed,
            )
            print(
                "XLM-R抽取训练完成："
                f"最佳验证实体F1为 {report['最佳验证实体F1']:.4f}，"
                f"报告已写入 {arguments.output_dir}。"
            )
            return 0
        if arguments.command == "train" and arguments.train_command == "compare-runs":
            from anitopy_ml.training.experiments import write_repeated_runs_summary

            summary = write_repeated_runs_summary(arguments.runs, arguments.output)
            print(
                f"重复实验汇总完成：{summary['实验数量']} 次，"
                f"平均验证实体F1为 {summary['验证实体F1']['平均值']:.4f}，"
                f"报告已写入 {arguments.output}。"
            )
            return 0
        if arguments.command == "calibrate":
            from anitopy_ml.training.calibrate import calibrate_extractor

            report = calibrate_extractor(
                model_directory=arguments.model,
                validation_path=arguments.validation,
                output_path=arguments.output,
                device=arguments.device,
                bucket_count=arguments.bucket_count,
                minimum_samples=arguments.minimum_samples,
            )
            print(
                f"模型校准完成：已校准字段 {len(report['字段'])} 个，"
                f"未校准字段 {len(report['未校准字段'])} 个，"
                f"结果已写入 {arguments.output}。"
            )
            return 0
        if arguments.command == "calibration-policy":
            from anitopy_ml.inference.calibration import derive_acceptance_policy, write_acceptance_policy

            try:
                payload = json.loads(Path(arguments.calibration).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise InputValidationError(f"无法读取校准文件：{arguments.calibration}。") from error
            policy_payload = derive_acceptance_policy(payload, maximum_error_rate=arguments.maximum_error_rate)
            write_acceptance_policy(policy_payload, arguments.output)
            print(
                f"接受策略已生成：自动接收字段 {len(policy_payload['字段自动接收阈值'])} 个，"
                f"验证集覆盖率 {policy_payload['验证集覆盖率']:.2%}。"
            )
            return 0
        if arguments.command == "evaluate":
            from anitopy_ml.training.evaluate import evaluate_extractor

            report = evaluate_extractor(
                model_directory=arguments.model,
                test_path=arguments.test,
                output_path=arguments.output,
                device=arguments.device,
            )
            print(
                f"冻结测试评测完成：实体F1为 {report['实体指标']['F1']:.4f}，"
                f"整条严格完全正确率为 {report['整条严格完全正确率']:.4f}。"
            )
            return 0
        if arguments.command == "parse":
            result = MediaParser.from_pretrained(arguments.model, device=arguments.device).parse(arguments.title)
            print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
            return 0
        if arguments.command == "batch":
            with Path(arguments.input).open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or arguments.column not in reader.fieldnames:
                    raise InputValidationError(f"输入CSV缺少标题列：{arguments.column}。")
                titles = [str(row.get(arguments.column, "")) for row in reader]
            parser_instance = MediaParser.from_pretrained(arguments.model, device=arguments.device)
            results = parser_instance.parse_batch(titles, on_error=arguments.on_error)
            target = Path(arguments.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8", newline="\n") as stream:
                for result in results:
                    payload = result.model_dump() if hasattr(result, "model_dump") else result
                    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            print(f"批量解析完成：已按输入顺序写入 {len(results)} 条结果到 {target}。")
            return 0
        parser.print_help()
        return 0
    except AnitopyMlError as error:
        message = message_for(error.code)
        detail = str(error).strip()
        suffix = f"{message} {detail}" if detail and detail != message else message
        print(f"错误：{suffix}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
